import gymnasium as gym
from gym import spaces
import numpy as np


class PerspectiveEnv(gym.Env):
    """
    3-Zone environment with volatility in red zone.
    - 0: Green (low reward, stable)
    - 1: Neutral (medium reward, stable)
    - 2: Red (high reward, volatile: may backfire)
    """
    def __init__(self, max_steps=50, p_counter=0.3):
        super().__init__()
        self.max_steps = max_steps
        self.p_counter = p_counter
        self.zones = {
            0: {"reward": 0.3, "volatility": False},
            1: {"reward": 0.5, "volatility": False},
            2: {"reward": 1.0, "volatility": True},
        }
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Discrete(3)
        self.reset()

    def reset(self):
        self.current_step = 0
        self.zone = 0
        return self.zone

    def step(self, action):
        self.zone = action
        info = self.zones[self.zone]
        reward = info["reward"]
        if info["volatility"] and np.random.rand() < self.p_counter:
            reward = -1.0  # counterattack
        self.current_step += 1
        done = self.current_step >= self.max_steps
        return self.zone, reward, done, {"volatility": info["volatility"]}


class PerspectiveAgent:
    """
    Agent with latent perspective P:
    0: short-term (greedy), 1: uncertainty-averse, 2: exploratory
    """
    def __init__(self):
        self.perspectives = ["short-term", "uncertainty-averse", "exploratory"]
        self.posterior = np.ones(3) / 3  # uniform prior

    def update_posterior(self, volatility):
        likelihood = np.array([
            1.0,  # short-term unaffected
            2.0 if volatility else 0.5,  # uncertainty-averse reacts
            1.0  # exploratory neutral
        ])
        self.posterior *= likelihood
        self.posterior /= self.posterior.sum()

    def select_action(self, obs):
        p = np.argmax(self.posterior)
        if p == 0:
            return 2  # short-term: always Red
        elif p == 1:
            return np.random.choice([0, 1])  # avoid Red
        else:
            return np.random.choice([0, 1, 2])  # exploratory
        
        
def evaluate(agent, env, phi_target=10.0, lambda_reg=0.1, verbose=False):
    obs = env.reset()
    F_env, F_phi = 0.0, 0.0
    done = False

    while not done:
        action = agent.select_action(obs)
        obs, reward, done, info = env.step(action)
        F_env += reward

        desired_p = 1 if info["volatility"] else 0
        F_phi += agent.posterior[desired_p]  # higher when perspective matches

        agent.update_posterior(info["volatility"])

        if verbose:
            print(f"Step: {env.current_step}, Action: {action}, Zone: {obs}, "
                  f"Volatile: {info['volatility']}, Reward: {reward:.2f}, "
                  f"P: {agent.posterior.round(2)}")

    F_total = F_env - lambda_reg * (phi_target - F_phi) ** 2
    return F_env, F_phi, F_total


if __name__ == "__main__":
    env = PerspectiveEnv()
    agent = PerspectiveAgent()
    F_env, F_phi, F_total = evaluate(agent, env, lambda_reg=0.1, phi_target=10.0, verbose=True)
    print(f"\nFinal Scores — F_env: {F_env:.2f}, F_phi: {F_phi:.2f}, F_total: {F_total:.2f}")
















