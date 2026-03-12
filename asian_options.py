import math
import numpy as np
from numba import njit, prange


@njit(parallel=True)
def asian_option_mc_numba(
    s0,
    k,
    r,
    sigma,
    t,
    n_fixings,
    n_paths,
    is_call=True,
    seed=12345,
):
    np.random.seed(seed)

    dt = t / n_fixings
    sqrt_dt = math.sqrt(dt)
    drift = (r - 0.5 * sigma * sigma) * dt
    vol = sigma * sqrt_dt
    disc = math.exp(-r * t)

    sum_payoff = 0.0
    sum_payoff2 = 0.0

    for i in prange(n_paths):
        s = s0
        running_sum = 0.0

        for _ in range(n_fixings):
            z = np.random.normal()
            s *= math.exp(drift + vol * z)
            running_sum += s

        avg = running_sum / n_fixings

        if is_call:
            payoff = max(avg - k, 0.0)
        else:
            payoff = max(k - avg, 0.0)

        pv = disc * payoff

        sum_payoff += pv
        sum_payoff2 += pv * pv

    mean = sum_payoff / n_paths
    var = max(sum_payoff2 / n_paths - mean * mean, 0.0)
    stderr = math.sqrt(var / n_paths)

    return mean, stderr


# Example run
price, stderr = asian_option_mc_numba(
    s0=100.0,
    k=100.0,
    r=0.05,
    sigma=0.2,
    t=1.0,
    n_fixings=252,
    n_paths=1_000_000,
)

print("Price:", price)
print("Std error:", stderr)
