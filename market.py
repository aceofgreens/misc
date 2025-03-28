import matplotlib.pyplot as plt
from jax import vmap, grad, jit
import jax.numpy as jnp
import jax
import optax
import numpy as np
import flax.linen as nn
from functools import partial

def demand(price, buyer_A, buyer_B):
    if len(price.shape) >= 1:
        return jnp.maximum(buyer_A[:, None] - buyer_B[:, None] * price[None], 0)
    return jnp.maximum(buyer_A - buyer_B * price, 0)

def wtp(quantity, buyer_A, buyer_B):
    return buyer_A/buyer_B - quantity/buyer_B 

def step(actions, rng, buyer_A, buyer_B, marginal_cost, num_interactions=10_000):
    # actions is (N, 2), N sellers, (p, q) each
    num_sellers = actions.shape[0]
    num_buyers = buyer_A.shape[0]
    seller_prices, produced = actions[:, 0], actions[:, 1]
    
    demand_at_p = demand(seller_prices, buyer_A, buyer_B)
    inventory = produced.copy()  # initial inventory equals produced
    sales = jnp.zeros(num_sellers)
    buyer_buys = jnp.zeros(num_buyers)
    WTPs = wtp(demand_at_p, buyer_A[:, None], buyer_B[:, None]) # (num_buyers, num_sellers)
    avg_demand = demand(seller_prices.mean(), buyer_A, buyer_B)
    epsilon = 0.5  # epsilon bound for randomness (adjust as needed)

    # Loop state: (i, inventory, sales, buyer_buys, rng)
    init_state = (0, inventory, sales, buyer_buys, rng)
    
    def cond_fun(state):
        i, inventory, sales, buyer_buys, rng = state
        # Continue if there are remaining interactions and some seller has supply.
        return (i < num_interactions) & (jnp.any(inventory >= 1))
    
    def loop_body(state):
        i, inventory, sales, buyer_buys, rng = state
        buyer_idx = jnp.mod(i, num_buyers)
        
        def skip_sale(_):
            return (i + 1, inventory, sales, buyer_buys, rng)
        
        def attempt_sale(_):
            available_mask = (inventory >= 1)  # shape: (num_sellers,)
            candidate_mask = (WTPs[buyer_idx, :] >= (seller_prices - 1e-5)) & available_mask

            def no_candidate(rng_operand):
                # If no candidate is found, simply pass along the RNG.
                return (i + 1, inventory, sales, buyer_buys, rng_operand)

            def candidate_found(rng_operand):
                # Use the RNG passed in as operand.
                rng_local = rng_operand
                candidate_prices = jnp.where(candidate_mask, seller_prices, jnp.inf)
                prob = jax.nn.softmax(-candidate_prices/1) # Softmax
                rng_local, rng_choice = jax.random.split(rng_local)
                chosen = jax.random.choice(rng_choice, a=jnp.arange(num_sellers), p=prob)
                inventory_new = inventory.at[chosen].add(-1)
                sales_new = sales.at[chosen].add(1)
                buyer_buys_new = buyer_buys.at[buyer_idx].add(1)
                return (i + 1, inventory_new, sales_new, buyer_buys_new, rng_local)

            return jax.lax.cond(jnp.any(candidate_mask),
                                candidate_found,
                                no_candidate,
                                operand=rng)
                
        return jax.lax.cond(buyer_buys[buyer_idx] >= avg_demand[buyer_idx],
                            lambda _: skip_sale(None),
                            lambda _: attempt_sale(None),
                            operand=None)

    final_state = jax.lax.while_loop(cond_fun, loop_body, init_state)
    _, final_inventory, final_sales, final_buyer_buys, _ = final_state
    # Compute per-seller profits.
    current_profit = final_sales * seller_prices - marginal_cost * produced
    return current_profit, final_sales, produced



# ---------------------
# Simulation Parameters
# ---------------------
num_sellers = 2 
num_buyers  = 10    
num_interactions = 10_000
num_iterations = 200
batch_size = 64
p_range = jnp.array([0, 20])
q_range = jnp.array([0, 1100])

# ----------------------------
# Initialize Sellers and Buyers
# ----------------------------
rng = jax.random.PRNGKey(42)
rngs = jax.random.split(rng, 5)
rng = rngs[0]

# Symmetric firms with constant marginal cost
marginal_cost = jnp.ones((num_sellers,)) * 1.5

# Buyers: each has a linear demand function: Q_d = max(A - B * price, 0).
buyer_A = jax.random.uniform(key=rngs[3],shape=(num_buyers,)) * 20 + 100        # Demand intercepts in [100, 120]
buyer_B = jax.random.uniform(key=rngs[4], shape=(num_buyers,)) * 0.4 + 4.8        # Demand slopes in [4.8, 5.2]



# ---------------------
# Plots
# ---------------------
K1 = 301
K2 = 301
P1 = 0
P2 = 20
Q1 = 0
Q2 = 1100

P_fixed = 14
Q_fixed = 700

from matplotlib.patches import Circle, Rectangle
from matplotlib.legend_handler import HandlerPatch
from mpl_toolkits.axes_grid1 import make_axes_locatable

def make_legend_circle(legend, orig_handle, xdescent, ydescent, width, height, fontsize):
    # Return a properly placed and scaled circle
    return Circle((width / 2, height / 2 - 5), radius=min(width, height) / 2,
                  facecolor=orig_handle.get_facecolor(),
                  edgecolor=orig_handle.get_edgecolor())


def plot_br(filename=None, legend=True, firm_1_fixed=True):
    P, Q = jnp.meshgrid(np.linspace(P1, P2, K1, endpoint=True), np.linspace(Q1, Q2, K2, endpoint=True), indexing='ij')
    if firm_1_fixed:
        plt.scatter([P_fixed], [Q_fixed], marker='s', c='C1', s=150, edgecolors='k')
    else:
        plt.scatter([P_fixed], [Q_fixed], marker='s', c='C0', s=150, edgecolors='k')
    fixed = jnp.array([P_fixed, Q_fixed])[None, None].repeat(K1, 0).repeat(K2, 1)[:, :, None]
    pairs = jnp.stack([P, Q], axis=-1)[:, :, None] # (K1, K2, 1, 2)
    pairs = jnp.concatenate((pairs, fixed), axis=2)
    pairs_flat = pairs.reshape((-1, 2, 2))
    rngs = jax.random.split(rng, pairs_flat.shape[0])
    profit, sales, produced = jax.vmap(step, in_axes=(0, 0, None, None, None))(pairs_flat, rngs, buyer_A, buyer_B, marginal_cost)
    profit_ = profit[:, 0]
    sales_ = sales[:, 0]
    max_profit = profit_.argmax()
    i, j = max_profit // K1, max_profit % K2
    best_p = pairs[i, j, 0, 0]
    best_q = pairs[i, j, 0, 1]
    if firm_1_fixed:
        plt.gca().scatter([best_p], [best_q], s=150, c='C0', edgecolors='k', zorder=10)
    else:
        plt.gca().scatter([best_p], [best_q], s=150, c='C1', edgecolors='k', zorder=10)
    im = plt.imshow(profit_.reshape(K1, K2).T, cmap='magma', extent=[P1, P2, Q2, Q1], aspect='auto')
    plt.xticks(np.linspace(P1, P2, 11, endpoint=True))  # Adjust tick locations for X
    plt.yticks(np.linspace(Q1, Q2, 11, endpoint=True))  # Adjust tick locations for Y
    plt.ylabel("Quantity supplied")
    plt.xlabel("Price")

    if legend:
        white_circle = Circle((0, 0), radius=5, facecolor='white', edgecolor='black')
        rect_c0 = Rectangle((0, 0), width=10, height=10, facecolor='C0', edgecolor='none')
        rect_c1 = Rectangle((0, 0), width=10, height=10, facecolor='C1', edgecolor='none')
        white_square = Rectangle((0, 0), width=2, height=10, facecolor='white', edgecolor='black')
        plt.legend(handles=[rect_c1, rect_c0, white_circle, white_square],
                labels=['Firm 1', 'Firm 2', 'Best response', 'Fixed strategy'],
                title="Legend", loc='upper left', handleheight=2,
                handler_map={white_circle: HandlerPatch(patch_func=make_legend_circle)})

    # divider = make_axes_locatable(plt.gca())
    # cax = divider.append_axes('right', size='5%', pad=0.05)
    # plt.colorbar(im, cax=cax, orientation='vertical')

    if filename is not None:
        plt.savefig(filename, dpi=200, bbox_inches='tight')
    plt.show()
    return pairs[i, j, 0, 0], pairs[i, j, 0, 1]

plot_br(filename=None, legend=True, firm_1_fixed=False)
