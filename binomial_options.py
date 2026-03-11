# Run using streamlit run binomial_options.py
# Requires matplotlib, pandas, streamlit modules.
import math
from dataclasses import dataclass

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


@dataclass
class BinomialResult:
    price: float
    delta_t: float
    up_factor: float
    down_factor: float
    probability_up: float
    expected_payoff: float
    stock_tree: list[list[float]]
    option_tree: list[list[float]]


def build_stock_tree(spot: float, up_factor: float, down_factor: float, steps: int) -> list[list[float]]:
    tree: list[list[float]] = []
    for step in range(steps + 1):
        level = []
        for down_moves in range(step + 1):
            up_moves = step - down_moves
            price = spot * (up_factor ** up_moves) * (down_factor ** down_moves)
            level.append(price)
        tree.append(level)
    return tree


def payoff(option_type: str, stock_price: float, strike: float) -> float:
    if option_type == "Call":
        return max(stock_price - strike, 0.0)
    return max(strike - stock_price, 0.0)


def expected_terminal_payoff(
    stock_tree: list[list[float]],
    strike: float,
    probability_up: float,
    option_type: str,
) -> float:
    steps = len(stock_tree) - 1
    terminal_payoffs = stock_tree[-1]
    expected_value = 0.0

    for down_moves, stock_price in enumerate(terminal_payoffs):
        up_moves = steps - down_moves
        path_probability = math.comb(steps, up_moves) * (probability_up ** up_moves) * (
            (1.0 - probability_up) ** down_moves
        )
        expected_value += path_probability * payoff(option_type, stock_price, strike)

    return expected_value


def price_option(
    spot: float,
    strike: float,
    rate: float,
    maturity: float,
    steps: int,
    probability_up: float,
    up_move_pct: float,
    option_type: str,
    exercise_style: str,
) -> BinomialResult:
    delta_t = maturity / steps
    up_factor = 1.0 + up_move_pct
    down_factor = 1.0 / up_factor
    discount = math.exp(-rate * delta_t)
    total_discount = math.exp(-rate * maturity)

    stock_tree = build_stock_tree(spot, up_factor, down_factor, steps)
    option_tree: list[list[float]] = [[0.0] * (step + 1) for step in range(steps + 1)]

    for node, stock_price in enumerate(stock_tree[-1]):
        option_tree[-1][node] = payoff(option_type, stock_price, strike)

    for step in range(steps - 1, -1, -1):
        for node in range(step + 1):
            continuation = discount * (
                probability_up * option_tree[step + 1][node]
                + (1.0 - probability_up) * option_tree[step + 1][node + 1]
            )
            if exercise_style == "American":
                intrinsic = payoff(option_type, stock_tree[step][node], strike)
                option_tree[step][node] = max(continuation, intrinsic)
            else:
                option_tree[step][node] = continuation

    expected_payoff = expected_terminal_payoff(stock_tree, strike, probability_up, option_type)
    if exercise_style == "European":
        option_tree[0][0] = total_discount * expected_payoff

    return BinomialResult(
        price=option_tree[0][0],
        delta_t=delta_t,
        up_factor=up_factor,
        down_factor=down_factor,
        probability_up=probability_up,
        expected_payoff=expected_payoff,
        stock_tree=stock_tree,
        option_tree=option_tree,
    )


def plot_tree(
    tree: list[list[float]],
    title: str,
    value_format: str,
    node_color: str = "#0f766e",
    text_color: str = "white",
    edge_color: str = "white",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 6))

    for step, level in enumerate(tree):
        for node, value in enumerate(level):
            y = step - 2 * node

            if step < len(tree) - 1:
                ax.plot([step, step + 1], [y, y + 1], color="#94a3b8", linewidth=1.1)
                ax.plot([step, step + 1], [y, y - 1], color="#94a3b8", linewidth=1.1)

            ax.scatter(step, y, s=1100, color=node_color, edgecolor=edge_color, linewidth=1.5, zorder=3)
            ax.text(
                step,
                y,
                format(value, value_format),
                color=text_color,
                fontsize=7,
                ha="center",
                va="center",
                zorder=4,
            )

    ax.set_title(title, fontsize=14, weight="bold")
    ax.set_xlabel("Step")
    ax.set_yticks([])
    ax.grid(axis="x", linestyle="--", alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(-0.5, len(tree) - 0.5)

    return fig


def tree_to_frame(tree: list[list[float]], label: str) -> pd.DataFrame:
    rows = []
    for step, level in enumerate(tree):
        for down_moves, value in enumerate(level):
            rows.append(
                {
                    "step": step,
                    "down_moves": down_moves,
                    label: value,
                }
            )
    return pd.DataFrame(rows)


st.set_page_config(
    page_title="Binomial Option Tree Explorer",
    page_icon="🌲",
    layout="wide",
)

st.title("Binomial Option Tree Explorer")
st.caption("Visualize stock-price and option-value trees with a binomial model.")

DEFAULT_SPOT = 100.0
DEFAULT_STRIKE = 100.0
DEFAULT_MATURITY = 1.0
DEFAULT_RATE = 0.05
with st.sidebar:
    st.header("Inputs")
    steps = st.slider("Tree steps", min_value=1, max_value=12, value=4, step=1)
    probability_up = st.slider("Probability of up", min_value=0.01, max_value=0.99, value=0.50, step=0.01)
    up_move_pct = st.slider("Up move (%)", min_value=1.0, max_value=100.0, value=10.0, step=1.0) / 100.0
    rate = st.slider("Risk-free rate (%)", min_value=0.0, max_value=20.0, value=5.0, step=0.25) / 100.0
    option_type = st.radio("Option type", options=["Call", "Put"], horizontal=True)
    exercise_style = st.radio("Exercise style", options=["European", "American"], horizontal=True)

try:
    result = price_option(
        spot=DEFAULT_SPOT,
        strike=DEFAULT_STRIKE,
        rate=rate,
        maturity=DEFAULT_MATURITY,
        steps=steps,
        probability_up=probability_up,
        up_move_pct=up_move_pct,
        option_type=option_type,
        exercise_style=exercise_style,
    )
except ValueError as exc:
    st.error(str(exc))
    st.stop()

metric_1, metric_2, metric_3, metric_4 = st.columns(4)
metric_1.metric("Option value", f"{result.price:.4f}")
metric_2.metric("u", f"{result.up_factor:.4f}")
metric_3.metric("d", f"{result.down_factor:.4f}")
metric_4.metric("p(up)", f"{result.probability_up:.2f}")
if exercise_style == "European":
    st.caption(
        f"Expected terminal payoff = {result.expected_payoff:.4f}; discounted present value = "
        f"{math.exp(-rate * DEFAULT_MATURITY) * result.expected_payoff:.4f}"
    )
else:
    st.caption(
        f"Expected terminal payoff = {result.expected_payoff:.4f}; American pricing still uses "
        "backward induction to allow early exercise."
    )

st.markdown(
    f"""
    `Δt = {result.delta_t:.4f}` years, `N = {steps}`, `S0 = {DEFAULT_SPOT:.0f}`, `K = {DEFAULT_STRIKE:.0f}`,
    `r = {rate:.2%}`, `up = {up_move_pct:.0%}`, `down = {(result.down_factor - 1.0):.1%}`, `T = {DEFAULT_MATURITY:.2f}`
    """
)

chart_col, table_col = st.columns([1.6, 1.0])

with chart_col:
    left, right = st.columns(2)
    with left:
        stock_fig = plot_tree(result.stock_tree, "Stock Price Tree", ".2f")
        st.pyplot(stock_fig, use_container_width=True)
        plt.close(stock_fig)
    with right:
        option_fig = plot_tree(
            result.option_tree,
            f"{exercise_style} {option_type} Value Tree",
            ".3f",
            node_color="#fef3c7",
            text_color="#9a3412",
            edge_color="#b45309",
        )
        st.pyplot(option_fig, use_container_width=True)
        plt.close(option_fig)

with table_col:
    st.subheader("Tree data")
    tree_kind = st.selectbox("Table view", options=["Option values", "Stock prices"])
    if tree_kind == "Option values":
        st.dataframe(tree_to_frame(result.option_tree, "option_value"), use_container_width=True)
    else:
        st.dataframe(tree_to_frame(result.stock_tree, "stock_price"), use_container_width=True)

with st.expander("Model notes"):
    st.write(
        "This app uses a binomial tree with fixed defaults for spot, strike, and maturity. "
        "The `Probability of up` slider controls the continuation-value weighting during backward "
        "induction, `Up move (%)` sets the size of each up step, and `Risk-free rate (%)` controls "
        "discounting. The down step is set as the reciprocal move `d = 1 / u`. For European options, "
        "the root value is shown as the discounted expected mean of terminal payoffs."
    )
