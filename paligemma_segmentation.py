# VISUALIZATION CODE TAKEN FROM 
# https://github.com/google-research/big_vision/blob/main/big_vision/evaluators/proj/paligemma/transfers/segmentation.py
# AND ADAPTED FOR LEARNING PURPOSES.
import os
import re
import functools
from transformers import PaliGemmaProcessor,PaliGemmaForConditionalGeneration
from transformers.image_utils import load_image

import torch
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
import jax
from tensorflow.io import gfile
import cv2
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt

model_id = "google/paligemma2-3b-pt-448"
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg"
image = load_image(url)
image2 = load_image('penguin.jpg')

model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto").eval()
processor = PaliGemmaProcessor.from_pretrained(model_id)

model_inputs = processor(text='segment the car', images=image, return_tensors="pt").to(torch.bfloat16).to(model.device)
model_inputs2 = processor(text='segment the front penguin', images=image2, return_tensors="pt").to(torch.bfloat16).to(model.device)
input_len = model_inputs["input_ids"].shape[-1]

with torch.inference_mode():
    generation_full = model.generate(**model_inputs, max_new_tokens=100, do_sample=False)
    generation = generation_full[0][input_len:]
    decoded = processor.decode(generation, skip_special_tokens=False)

    generation_full = model.generate(**model_inputs2, max_new_tokens=100, do_sample=False)
    generation = generation_full[0][input_len:]
    decoded2 = processor.decode(generation, skip_special_tokens=False)


def _inrange(a, min_value, max_value):
  return (np.clip(a, min_value, max_value) == a).all()

def _area(y1, x1, y2, x2):
  return max(x2 - x1, 0.0) * max(y2 - y1, 0.0)


_KNOWN_MODELS = {
    # Trained on open images.
    'oi': 'gs://big_vision/paligemma/vae-oid.npz',
}


def _get_params(checkpoint):
    """Converts PyTorch checkpoint to Flax params."""

    def transp(kernel):
        return np.transpose(kernel, (2, 3, 1, 0))

    def conv(name):
        return {
            'bias': checkpoint[name + '.bias'],
            'kernel': transp(checkpoint[name + '.weight']),
        }

    def resblock(name):
        return {
            'Conv_0': conv(name + '.0'),
            'Conv_1': conv(name + '.2'),
            'Conv_2': conv(name + '.4'),
        }

    return {
        '_embeddings': checkpoint['_vq_vae._embedding'],
        'Conv_0': conv('decoder.0'),
        'ResBlock_0': resblock('decoder.2.net'),
        'ResBlock_1': resblock('decoder.3.net'),
        'ConvTranspose_0': conv('decoder.4'),
        'ConvTranspose_1': conv('decoder.6'),
        'ConvTranspose_2': conv('decoder.8'),
        'ConvTranspose_3': conv('decoder.10'),
        'Conv_1': conv('decoder.12'),
    }

def _quantized_values_from_codebook_indices(codebook_indices, embeddings):
    batch_size, num_tokens = codebook_indices.shape
    assert num_tokens == 16, codebook_indices.shape
    unused_num_embeddings, embedding_dim = embeddings.shape

    encodings = jnp.take(embeddings, codebook_indices.reshape((-1)), axis=0)
    encodings = encodings.reshape((batch_size, 4, 4, embedding_dim))
    return encodings


class ResBlock(nn.Module):
  features: int

  @nn.compact
  def __call__(self, x):
    original_x = x
    x = nn.Conv(features=self.features, kernel_size=(3, 3), padding=1)(x)
    x = nn.relu(x)
    x = nn.Conv(features=self.features, kernel_size=(3, 3), padding=1)(x)
    x = nn.relu(x)
    x = nn.Conv(features=self.features, kernel_size=(1, 1), padding=0)(x)
    return x + original_x


class Decoder(nn.Module):
    """Upscales quantized vectors to mask."""

    @nn.compact
    def __call__(self, x):
        num_res_blocks = 2
        dim = 128
        num_upsample_layers = 4

        x = nn.Conv(features=dim, kernel_size=(1, 1), padding=0)(x)
        x = nn.relu(x)

        for _ in range(num_res_blocks):
            x = ResBlock(features=dim)(x)

        for _ in range(num_upsample_layers):
            x = nn.ConvTranspose(
                features=dim,
                kernel_size=(4, 4),
                strides=(2, 2),
                padding=2,
                transpose_kernel=True,
            )(x)
            x = nn.relu(x)
            dim //= 2

        x = nn.Conv(features=1, kernel_size=(1, 1), padding=0)(x)
        return x


@functools.cache
def get_reconstruct_masks(model):
    def reconstruct_masks(codebook_indices):
        quantized = _quantized_values_from_codebook_indices(
            codebook_indices, params['_embeddings']
        )
        return Decoder().apply({'params': params}, quantized)

    with gfile.GFile(_KNOWN_MODELS.get(model, model), 'rb') as f:
        params = _get_params(dict(np.load(f)))

    return jax.jit(reconstruct_masks, backend='cpu')

reconstruct_fn = get_reconstruct_masks('oi')

def extract_and_create_array(pattern: str):
    matches = re.findall(r"<seg(\d{3})>", pattern)
    numbers = [int(match) for match in matches]
    filtered_numbers = [num for num in numbers if 0 <= num <= 127]
    if len(filtered_numbers) < 16:
        filtered_numbers.extend([0] * (16 - len(filtered_numbers)))  # Pad with zeros
    elif len(filtered_numbers) > 16:
        filtered_numbers = filtered_numbers[:16]  # Truncate to size 16
    array = np.array(filtered_numbers, dtype=np.int32)
    return array

def extract_and_scale_coords(coord_string: str, original_max=1024, target_max=448):
    matches = re.findall(r"<loc(\d{4})>", coord_string)
    coords = [int(match) for match in matches]
    if len(coords) != 4:
        raise ValueError("Input string must contain exactly four coordinate values.")
    scaled_coords = [int(coord * target_max / original_max) for coord in coords]
    return np.array(scaled_coords, dtype=np.int32)

def resize_image(image, target_size):
    resized_image = cv2.resize(image[:, :, 0], target_size[::-1])  # OpenCV uses (width, height)    
    return resized_image[:, :, np.newaxis]

def resize_rgb(image, target_size):
    resized_image = cv2.resize(image, target_size[::-1])  # OpenCV uses (width, height)
    return resized_image

B = 1
parts = decoded.split(' ')
parts2 = decoded2.split(' ')

segs = list(filter(lambda x: 'seg' in x, parts))
segs2 = list(filter(lambda x: 'seg' in x, parts2))

codes = extract_and_create_array(segs[0])[None]
codes2 = extract_and_create_array(segs2[0])[None]

masks = reconstruct_fn(codes)
masks2 = reconstruct_fn(codes2)

fig, axs = plt.subplots(1, 2, sharey=True, figsize=(10, 6))

# First image
im = model_inputs['pixel_values'][0].float().cpu().detach().permute(1, 2, 0).numpy()
im_array = resize_rgb(np.asarray(image), (448, 448))
scaled_coords = extract_and_scale_coords(parts[0])
h = scaled_coords[2]-scaled_coords[0]
w = scaled_coords[3] - scaled_coords[1]
axs[0].add_patch(Rectangle(xy=(scaled_coords[1], scaled_coords[0]), 
                           width=scaled_coords[3] - scaled_coords[1], 
                           height=scaled_coords[2]-scaled_coords[0], 
                           facecolor='none', alpha=0.8, edgecolor='#00ee00'))
resized_mask = resize_image(np.array(masks[0]), (h, w))
full = np.ones((448, 448, 1))*(-1)
h_, w_, _ = resized_mask.shape
full[scaled_coords[0]:scaled_coords[0] +h_, scaled_coords[1]:scaled_coords[1]+w_] = resized_mask
full_scaled = (full - full.min()) / (full.max() - full.min())
full_scaled = np.concatenate((full_scaled*2, full_scaled*89, full_scaled*13), axis=-1)
full_scaled = (full_scaled - full_scaled.min()) / (full_scaled.max() - full_scaled.min())
axs[0].imshow(im_array)
axs[0].imshow(full_scaled, alpha=0.5)


# Second image
im = model_inputs2['pixel_values'][0].float().cpu().detach().permute(1, 2, 0).numpy()
im_array = resize_rgb(np.asarray(image2), (448, 448))
scaled_coords = extract_and_scale_coords(parts2[0])
h = scaled_coords[2]-scaled_coords[0]
w = scaled_coords[3] - scaled_coords[1]
scaled_coords2 = extract_and_scale_coords(parts2[6])
h2 = scaled_coords2[2]-scaled_coords2[0]
w2 = scaled_coords2[3] - scaled_coords2[1]

axs[1].add_patch(Rectangle(xy=(scaled_coords[1], scaled_coords[0]), 
                           width=scaled_coords[3] - scaled_coords[1], 
                           height=scaled_coords[2]-scaled_coords[0], 
                           facecolor='none', alpha=0.8, edgecolor='red',))
axs[1].add_patch(Rectangle(xy=(scaled_coords2[1], scaled_coords2[0]), 
                           width=scaled_coords2[3] - scaled_coords2[1], 
                           height=scaled_coords2[2]-scaled_coords2[0], 
                           facecolor='none', alpha=0.8, edgecolor='blue',))
resized_mask = resize_image(np.array(masks2[0]), (h, w))
full = np.ones((448, 448, 1))*(-1)
h_, w_, _ = resized_mask.shape
full[scaled_coords[0]:scaled_coords[0] +h_, scaled_coords[1]:scaled_coords[1]+w_] = resized_mask
full_scaled = (full - full.min()) / (full.max() - full.min())
full_scaled = np.concatenate((full_scaled, np.zeros((448, 448, 2))), axis=-1)
axs[1].imshow(im_array)
axs[1].imshow(full_scaled, alpha=0.5)
plt.subplots_adjust(hspace=0, wspace=0.125)
plt.savefig("paligemma_segmentation.png", dpi=250, bbox_inches='tight')