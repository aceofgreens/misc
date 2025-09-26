import argparse
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


@dataclass
class ExperimentConfig:
    model_name: str = "Qwen/Qwen3-0.6B"
    prefix: str = "The following text talks about cats: "
    raw_text: str = "Cats use their tails for balance while climbing and jumping."
    steps: int = 50
    lr: float = 5e-3
    seed: int = 0
    dtype: str = "float32"
    init_std: float = 0.02
    grad_clip: Optional[float] = 1.0
    objective: str = "max_logprob"
    learnable_length: Optional[int] = None
    entropy_weight: float = 0.
    prefix_prior_weight: float = .02
    anchor_temperature: float = 0.02
    anchor_topk: Optional[int] = 8
    sample_count: int = 3
    sample_max_new_tokens: int = 64
    sample_temperature: float = 0.8
    sample_top_p: float = 0.95



def prepare_tokens(
    tokenizer: AutoTokenizer, prefix: str, raw_text: str, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    prefix_ids = tokenizer(prefix, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
    text_ids = tokenizer(raw_text, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
    bos_id = tokenizer.bos_token_id
    if bos_id is not None:
        bos_token = torch.tensor([[bos_id]], device=device)
        prefix_ids = torch.cat([bos_token, prefix_ids], dim=1)
    labels = torch.cat([prefix_ids, text_ids], dim=1)
    return prefix_ids, text_ids, labels


def compute_sequence_logprob(
    logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100
) -> Tuple[torch.Tensor, torch.Tensor]:
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    log_probs = F.log_softmax(shift_logits, dim=-1)
    mask = shift_labels != ignore_index
    safe_shift_labels = shift_labels.masked_fill(~mask, 0)
    token_log_probs = log_probs.gather(-1, safe_shift_labels.unsqueeze(-1)).squeeze(-1)
    token_log_probs = token_log_probs * mask
    return token_log_probs.sum(dim=-1), mask.sum(dim=-1)


def analyze_learnable_block(
    tokenizer: AutoTokenizer,
    prefix_ids: torch.Tensor,
    final_ids: Optional[torch.Tensor],
    logits: torch.Tensor,
    learnable_length: int,
    final_token_count: int,
    attentions,
    top_k: int = 3,
) -> None:
    with torch.no_grad():
        print("\n=== Learnable Block Next-token Predictions ===")
        learnable_logits = logits[:, prefix_ids.shape[1] : prefix_ids.shape[1] + learnable_length, :]
        probs = F.softmax(learnable_logits, dim=-1)
        log_probs = torch.log_softmax(learnable_logits, dim=-1)
        top_k = min(top_k, probs.shape[-1])
        topk = probs.topk(top_k, dim=-1)
        top_indices = topk.indices.cpu()
        top_scores = topk.values.cpu()

        for pos in range(learnable_length):
            tokens = tokenizer.convert_ids_to_tokens(top_indices[0, pos].tolist())
            top_str = ", ".join(
                f"{tok}:{score:.3f}" for tok, score in zip(tokens, top_scores[0, pos].tolist())
            )
            print(f"learnable[{pos:02d}] next-token top-k: {top_str}")

        if attentions is None:
            print("\n(No attention data captured.)")
            return

        print("\n=== Attention Summary (final iteration) ===")
        att_layers = [layer[0].detach().cpu() for layer in attentions]
        att_stack = torch.stack(att_layers, dim=0)  # (layers, heads, seq, seq)
        att_mean = att_stack.mean(dim=1)  # (layers, seq, seq)

        prefix_tokens = tokenizer.convert_ids_to_tokens(prefix_ids[0].tolist())
        final_tokens = (
            tokenizer.convert_ids_to_tokens(final_ids[0].tolist()) if final_ids is not None else []
        )

        seq_labels = []
        seq_labels.extend(prefix_tokens)
        seq_labels.extend([f"<learn-{i}>" for i in range(learnable_length)])
        seq_labels.extend(final_tokens)

        prefix_length = prefix_ids.shape[1]
        prefix_range = range(prefix_length)
        learnable_range = range(prefix_length, prefix_length + learnable_length)
        final_range = range(prefix_length + learnable_length, prefix_length + learnable_length + final_token_count)

        for layer_idx, layer_att in enumerate(att_mean):
            final_to_prefix = layer_att[list(final_range)][:, list(prefix_range)].mean().item() if prefix_length > 0 else 0.0
            final_to_learnable = layer_att[list(final_range)][:, list(learnable_range)].mean().item()
            final_to_final = layer_att[list(final_range)][:, list(final_range)].mean().item()
            print(
                f"Layer {layer_idx:02d}: final→prefix={final_to_prefix:.3f}, "
                f"final→learnable={final_to_learnable:.3f}, final→final={final_to_final:.3f}"
            )

        # Inspect top sources for the last layer per final token
        last_layer = att_mean[-1]
        print("\nTop attention sources per final token (last layer):")
        for offset, final_idx in enumerate(final_range):
            att_distribution = last_layer[final_idx]
            top_sources = att_distribution.topk(min(5, att_distribution.size(-1)))
            entries = []
            for src_idx, score in zip(top_sources.indices.tolist(), top_sources.values.tolist()):
                label = seq_labels[src_idx] if src_idx < len(seq_labels) else f"idx{src_idx}"
                entries.append(f"{label}:{score:.3f}")
            token_label = final_tokens[offset] if offset < len(final_tokens) else f"final[{offset}]"
            print(f"  {token_label}: {', '.join(entries)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Blend frozen token embeddings with a learnable block of embeddings and optimize them.")
    parser.add_argument("--steps", type=int, default=None, help="Number of SGD steps to run (defaults to config value).")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate for the embedding block optimizer.")
    parser.add_argument("--text", type=str, default=None, help="Custom text segment to optimize embeddings for.")
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["auto", "float32", "float16", "bfloat16"],
        default=None,
        help="Computation dtype for the model (default: float32).",
    )
    parser.add_argument(
        "--init-std",
        type=float,
        default=None,
        help="Standard deviation used to initialize the learnable embedding block.",
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=None,
        help="Max L2 norm for gradient clipping (set <=0 to disable).",
    )
    parser.add_argument(
        "--objective",
        type=str,
        choices=["next_token", "max_logprob"],
        default=None,
        help="Training objective: supervise next tokens or maximize raw log-probability.",
    )
    parser.add_argument(
        "--learnable-length",
        type=int,
        default=None,
        help="Number of learnable embedding vectors to insert (defaults to text token length).",
    )
    parser.add_argument(
        "--entropy-weight",
        type=float,
        default=None,
        help="Weight for entropy regularization on learnable positions (max_logprob only).",
    )
    parser.add_argument(
        "--prefix-prior-weight",
        type=float,
        default=None,
        help="Weight for anchoring learnable embeddings toward real token mixtures (max_logprob only).",
    )
    parser.add_argument(
        "--anchor-temperature",
        type=float,
        default=None,
        help="Temperature for projecting learnable embeddings onto the vocab simplex (max_logprob only).",
    )
    parser.add_argument(
        "--anchor-topk",
        type=int,
        default=None,
        help="Restrict anchor projection to the top-k nearest vocab embeddings (<=0 disables).",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=None,
        help="Number of sequences to sample after optimization.",
    )
    parser.add_argument(
        "--sample-max-new-tokens",
        type=int,
        default=None,
        help="Number of new tokens to autoregressively sample per sequence.",
    )
    parser.add_argument(
        "--sample-temperature",
        type=float,
        default=None,
        help="Sampling temperature for post-optimization generation.",
    )
    parser.add_argument(
        "--sample-top-p",
        type=float,
        default=None,
        help="Top-p nucleus threshold for post-optimization generation.",
    )
    args = parser.parse_args()

    cfg = ExperimentConfig()
    if args.steps is not None:
        cfg.steps = args.steps
    if args.lr is not None:
        cfg.lr = args.lr
    if args.text is not None:
        cfg.raw_text = args.text
    if args.dtype is not None:
        cfg.dtype = args.dtype
    if args.init_std is not None:
        cfg.init_std = args.init_std
    if args.grad_clip is not None:
        cfg.grad_clip = None if args.grad_clip <= 0 else args.grad_clip
    if args.objective is not None:
        cfg.objective = args.objective
    if args.learnable_length is not None:
        cfg.learnable_length = args.learnable_length
    if args.entropy_weight is not None:
        cfg.entropy_weight = args.entropy_weight
    if args.prefix_prior_weight is not None:
        cfg.prefix_prior_weight = args.prefix_prior_weight
    if args.anchor_temperature is not None:
        if args.anchor_temperature <= 0:
            raise ValueError("anchor_temperature must be positive when provided.")
        cfg.anchor_temperature = args.anchor_temperature
    if args.anchor_topk is not None:
        cfg.anchor_topk = None if args.anchor_topk <= 0 else args.anchor_topk
    if args.sample_count is not None:
        cfg.sample_count = max(0, args.sample_count)
    if args.sample_max_new_tokens is not None:
        cfg.sample_max_new_tokens = max(0, args.sample_max_new_tokens)
    if args.sample_temperature is not None:
        cfg.sample_temperature = args.sample_temperature
    if args.sample_top_p is not None:
        cfg.sample_top_p = args.sample_top_p

    torch.manual_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }

    if cfg.dtype == "auto":
        model_dtype = torch.float16 if device.type == "cuda" else torch.float32
    else:
        model_dtype = dtype_map[cfg.dtype]

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        trust_remote_code=True,
        torch_dtype=model_dtype,
    ).to(device)
    model.eval()
    model.requires_grad_(False)

    prefix_ids, text_ids, _ = prepare_tokens(tokenizer, cfg.prefix, cfg.raw_text, device)

    if cfg.learnable_length is not None and cfg.learnable_length <= 0:
        raise ValueError("learnable_length must be positive when provided.")

    embedding_layer = model.get_input_embeddings()
    embedding_weight = embedding_layer.weight
    hidden_size = embedding_layer.embedding_dim

    with torch.no_grad():
        prefix_embeds = embedding_layer(prefix_ids).detach()

    final_ids = None

    if cfg.objective == "next_token":
        text_length = text_ids.shape[1]
        learnable_length = cfg.learnable_length or text_length
        if learnable_length > text_length:
            raise ValueError(
                f"Requested learnable_length={learnable_length} exceeds available text tokens ({text_length})."
            )
        active_target_ids = text_ids[:, :learnable_length]
        target_labels = torch.cat([prefix_ids, active_target_ids], dim=1)
        final_embeds = None
        final_token_count = active_target_ids.shape[1]
    else:
        final_ids = text_ids
        final_length = final_ids.shape[1]
        if final_length == 0:
            raise ValueError("Final answer text produces no tokens; cannot optimize log-probability.")
        learnable_length = cfg.learnable_length or final_length
        total_length = prefix_ids.shape[1] + learnable_length + final_length
        target_labels = torch.full(
            (1, total_length),
            -100,
            dtype=torch.long,
            device=device,
        )
        target_labels[:, prefix_ids.shape[1] + learnable_length :] = final_ids
        with torch.no_grad():
            final_embeds = embedding_layer(final_ids).detach()
        final_token_count = final_length

    learnable_embeddings = torch.nn.Parameter(
        torch.randn(learnable_length, hidden_size, device=device, dtype=prefix_embeds.dtype) * cfg.init_std
    )

    optimizer = torch.optim.Adam([learnable_embeddings], lr=cfg.lr)

    print(f"Device: {device}")
    print(
        f"Prefix tokens: {prefix_ids.shape[1]}, learnable tokens: {learnable_length}, objective: {cfg.objective}"
    )
    if cfg.objective == "max_logprob":
        print(f"Final answer tokens supervised: {final_token_count}")

    for step in range(cfg.steps):
        optimizer.zero_grad()
        if cfg.objective == "next_token":
            blended = torch.cat([prefix_embeds, learnable_embeddings.unsqueeze(0)], dim=1)
            outputs = model(inputs_embeds=blended, labels=target_labels)
            loss = outputs.loss
            seq_logprob_tensor, token_counts = compute_sequence_logprob(
                outputs.logits, target_labels
            )
            avg_seq_logprob = seq_logprob_tensor / token_counts.clamp(min=1).to(
                seq_logprob_tensor.dtype
            )
        else:
            if final_embeds is None:
                raise RuntimeError("final_embeds should be set for max_logprob objective.")
            blended = torch.cat(
                [prefix_embeds, learnable_embeddings.unsqueeze(0), final_embeds], dim=1
            )
            outputs = model(
                inputs_embeds=blended,
                output_attentions=True,
                use_cache=False,
                return_dict=True,
            )
            seq_logprob_tensor, token_counts = compute_sequence_logprob(
                outputs.logits, target_labels
            )
            avg_seq_logprob = seq_logprob_tensor / token_counts.clamp(min=1).to(
                seq_logprob_tensor.dtype
            )
            loss = -(avg_seq_logprob.mean())
            if cfg.entropy_weight > 0:
                learnable_logits = outputs.logits[:, prefix_ids.shape[1] : prefix_ids.shape[1] + learnable_length, :]
                log_probs = torch.log_softmax(learnable_logits, dim=-1)
                probs = log_probs.exp()
                per_token_entropy = -(probs * log_probs).sum(dim=-1)
                entropy_penalty = per_token_entropy.mean()
                loss = loss + cfg.entropy_weight * entropy_penalty
            if cfg.prefix_prior_weight > 0:
                if cfg.anchor_temperature <= 0:
                    raise ValueError("anchor_temperature must be positive.")

                similarities = learnable_embeddings @ embedding_weight.t()
                if (
                    cfg.anchor_topk is not None
                    and 0 < cfg.anchor_topk < similarities.shape[1]
                ):
                    topk = torch.topk(similarities, cfg.anchor_topk, dim=-1)
                    scaled_values = topk.values / cfg.anchor_temperature
                    weights = torch.softmax(scaled_values, dim=-1)
                    topk_embeds = embedding_weight[topk.indices]
                    reconstructed = torch.einsum("lk,lkd->ld", weights, topk_embeds)
                else:
                    scaled_sim = similarities / cfg.anchor_temperature
                    weights = torch.softmax(scaled_sim, dim=-1)
                    reconstructed = weights @ embedding_weight

                anchor_loss = F.mse_loss(learnable_embeddings, reconstructed)
                loss = loss + cfg.prefix_prior_weight * anchor_loss
        loss.backward()

        with torch.no_grad():
            if learnable_embeddings.grad is not None:
                grad_norm = learnable_embeddings.grad.norm(dim=1).mean().item()
                if cfg.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_([learnable_embeddings], cfg.grad_clip)
            else:
                grad_norm = float("nan")
        optimizer.step()

        with torch.no_grad():
            seq_logprob = seq_logprob_tensor[0].item()
            avg_logprob = avg_seq_logprob[0].item()
            token_count = token_counts[0].item()

        print(
            f"step={step:02d} loss={loss.item():.4f} "
            f"seq_logprob={seq_logprob:.2f} avg_logprob/token={avg_logprob:.4f} "
            f"valid_tokens={token_count} mean_grad_norm={grad_norm:.4f}"
        )

    print("Optimization finished.")

    final_norm = learnable_embeddings.norm(dim=1).mean().item()
    print(f"Mean L2 norm of optimized embeddings: {final_norm:.4f}")

    if cfg.objective == "max_logprob":
        analyze_learnable_block(
            tokenizer,
            prefix_ids,
            final_ids,
            outputs.logits.detach(),
            learnable_length=learnable_length,
            final_token_count=final_token_count,
            attentions=outputs.attentions,
        )

    if cfg.sample_count > 0 and cfg.sample_max_new_tokens > 0:
        pad_token_id = tokenizer.eos_token_id
        if pad_token_id is None:
            pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = 0

        with torch.no_grad():
            base_prefix = prefix_embeds.detach()
            prefix_attention_mask = torch.ones(
                base_prefix.shape[:2], dtype=torch.long, device=base_prefix.device
            )
            soft_prompt = learnable_embeddings.detach().unsqueeze(0)
            base_embeds = torch.cat([base_prefix, soft_prompt], dim=1)
            attention_mask = torch.ones(
                base_embeds.shape[:2], dtype=torch.long, device=base_embeds.device
            )
            prefix_text = tokenizer.decode(prefix_ids[0], skip_special_tokens=True)

            print("\n=== Sampled Continuations ===")
            for idx in range(cfg.sample_count):
                seed = cfg.seed + idx

                set_seed(seed)
                baseline_ids = model.generate(
                    inputs_embeds=base_prefix,
                    attention_mask=prefix_attention_mask,
                    do_sample=True,
                    temperature=cfg.sample_temperature,
                    top_p=cfg.sample_top_p,
                    max_new_tokens=cfg.sample_max_new_tokens,
                    pad_token_id=pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                baseline_continuation = tokenizer.decode(
                    baseline_ids[0], skip_special_tokens=True
                )

                set_seed(seed)
                generated_ids = model.generate(
                    inputs_embeds=base_embeds,
                    attention_mask=attention_mask,
                    do_sample=True,
                    temperature=cfg.sample_temperature,
                    top_p=cfg.sample_top_p,
                    max_new_tokens=cfg.sample_max_new_tokens,
                    pad_token_id=pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                continuation = tokenizer.decode(
                    generated_ids[0], skip_special_tokens=True
                )
                print(f"\n-- Sample {idx + 1} (seed={seed}) --")
                print("[Baseline]")
                print(prefix_text + baseline_continuation)
                print("[With Learnable]")
                print(prefix_text + continuation)


if __name__ == "__main__":
    main()
