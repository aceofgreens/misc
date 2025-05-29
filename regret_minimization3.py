import numpy as np
import matplotlib.pyplot as plt

# Setup
np.random.seed(42)
num_players = 3
actions = ['H', 'T']
num_actions = len(actions)
T = 10000  # number of iterations
eta = 0.1  # learning rate

# Payoff matrix
payoffs = np.zeros((num_players, num_actions, num_actions, num_actions))
for i, a in enumerate(actions):
    for j, b in enumerate(actions):
        for k, c in enumerate(actions):
            choices = [a, b, c]
            unique = set(choices)
            if len(unique) == 1:
                payoff = [0, 0, 0]
            elif choices.count(a) == 1:
                payoff = [1, -0.5, -0.5]
            elif choices.count(b) == 1:
                payoff = [-0.5, 1, -0.5]
            else:
                payoff = [-0.5, -0.5, 1]
            payoffs[:, i, j, k] = payoff

# Initialize
probs = np.ones((num_players, num_actions)) / num_actions
regret_history = []
prob_history = np.zeros((num_players, num_actions, T))
cumulative_regrets = np.zeros((num_players, num_actions))
per_action_regret_history = np.zeros((num_players, num_actions, T))

# Run no-regret dynamics
for t in range(T):
    a_idxs = [np.random.choice(num_actions, p=probs[p]) for p in range(num_players)]
    for p in range(num_players):
        prob_history[p, :, t] = probs[p]

    realized_payoffs = payoffs[:, a_idxs[0], a_idxs[1], a_idxs[2]]

    for p in range(num_players):
        for a in range(num_actions):
            idxs = a_idxs.copy()
            idxs[p] = a
            hypothetical_payoff = payoffs[p, idxs[0], idxs[1], idxs[2]]
            regret = hypothetical_payoff - realized_payoffs[p]
            cumulative_regrets[p, a] += regret
            per_action_regret_history[p, a, t] = cumulative_regrets[p, a] / (t + 1)

    for p in range(num_players):
        weights = np.exp(eta * cumulative_regrets[p])
        probs[p] = weights / weights.sum()

    avg_regret = cumulative_regrets.max(axis=1).mean() / (t + 1)
    regret_history.append(avg_regret)

# Plot average regret and action probabilities
fig, axes = plt.subplots(num_players + 1, 1, figsize=(10, 10), sharex=True)
axes[0].plot(regret_history)
axes[0].set_title("Average Regret Over Time")
axes[0].set_ylabel("Avg Regret")

for p in range(num_players):
    for a in range(num_actions):
        axes[p + 1].plot(prob_history[p, a, :], label=f'P{p} {actions[a]}')
    axes[p + 1].set_ylabel(f'Player {p} Prob')
    axes[p + 1].legend()

axes[-1].set_xlabel("Iterations")
plt.tight_layout()
plt.show()

# Plot per-action regret
fig, axes = plt.subplots(num_players, num_actions, figsize=(12, 8), sharex=True)
for p in range(num_players):
    for a in range(num_actions):
        axes[p, a].plot(per_action_regret_history[p, a, :])
        axes[p, a].set_title(f'Player {p} Regret for {actions[a]}')
        axes[p, a].set_ylabel("Regret")
        if p == num_players - 1:
            axes[p, a].set_xlabel("Iterations")

plt.tight_layout()
plt.show()

