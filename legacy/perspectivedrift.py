# Fixing the shape mismatch in forward_models by matching dimensions for prediction

import numpy as np

class PerspectiveModule:
    def __init__(self, names=('optimistic', 'pessimistic', 'neutral')):
        self.names = names
        self.num = len(names)
        self.posterior = np.ones(self.num) / self.num
        self.history = []

    def update(self, obs, preds):
        likelihood = []
        for p in range(self.num):
            error = np.linalg.norm(obs - preds[p])
            likelihood.append(np.exp(-error))
        self.posterior *= likelihood
        self.posterior /= np.sum(self.posterior)
        self.history.append(self.get_dominant())

    def get_dominant(self):
        return int(np.argmax(self.posterior))

    def get_name(self):
        return self.names[self.get_dominant()]


class PerspectiveAgent:
    def __init__(self, state_dim=8, hidden_dim=8, action_dim=8):
        self.h = np.zeros((hidden_dim,))
        self.Wxh = np.random.randn(hidden_dim, state_dim)
        self.Whh = np.random.randn(hidden_dim, hidden_dim)
        self.h_policies = {
            'optimistic': np.random.randn(action_dim, hidden_dim),
            'pessimistic': np.random.randn(action_dim, hidden_dim),
            'neutral': np.random.randn(action_dim, hidden_dim),
        }
        self.forward_models = {
            'optimistic': lambda s, a: s + 0.1 * a,
            'pessimistic': lambda s, a: s - 0.1 * a,
            'neutral': lambda s, a: s,
        }
        self.perspective = PerspectiveModule()
        self.prev_state = None
        self.prev_action = None

    def step(self, state):
        self.h = np.tanh(self.Wxh @ state + self.Whh @ self.h)

        preds = []
        for name in self.perspective.names:
            if self.prev_state is not None and self.prev_action is not None:
                pred = self.forward_models[name](self.prev_state, self.prev_action)
            else:
                pred = state
            preds.append(pred)

        self.perspective.update(state, preds)

        dominant = self.perspective.get_name()
        policy = self.h_policies[dominant]
        action_logits = policy @ self.h
        action = np.tanh(action_logits)

        self.prev_state = state
        self.prev_action = action
        return action, dominant


def simulate_development(steps=25):
    agent = PerspectiveAgent()
    trace = []
    for t in range(steps):
        state = np.random.randn(8)
        action, perspective = agent.step(state)
        trace.append(perspective)
    return trace


trace = simulate_development()

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 2))
plt.plot(trace, marker='o')
plt.yticks(ticks=[0, 1, 2], labels=['optimistic', 'pessimistic', 'neutral'])
plt.xlabel("Timestep")
plt.title("Perspective Drift over Time (Development)")
plt.grid(True)
plt.tight_layout()
plt.show()
