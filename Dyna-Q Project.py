import numpy as np
import gym

class DynaQAgent:
    def __init__(self, state_size, action_size, alpha=0.1, gamma=0.95, epsilon=0.1, planning_steps=5):
        self.state_size = state_size
        self.action_size = action_size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.planning_steps = planning_steps
        self.q_table = np.zeros((state_size, action_size))
        self.model = {}

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state, best_next_action]
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error

        # 模型學習
        if state not in self.model:
            self.model[state] = {}
        self.model[state][action] = (reward, next_state)

        # 模型規劃
        for _ in range(self.planning_steps):
            s = np.random.choice(list(self.model.keys()))
            a = np.random.choice(list(self.model[s].keys()))
            r, s_next = self.model[s][a]
            best_next_action = np.argmax(self.q_table[s_next])
            td_target = r + self.gamma * self.q_table[s_next, best_next_action]
            td_error = td_target - self.q_table[s, a]
            self.q_table[s, a] += self.alpha * td_error

# 創建環境
env = gym.make('FrozenLake-v0')

# 設置參數
state_size = env.observation_space.n
action_size = env.action_space.n
agent = DynaQAgent(state_size, action_size)
episodes = 1000

# 訓練Dyna-Q
for episode in range(episodes):
    state = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = agent.choose_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.learn(state, action, reward, next_state)
        state = next_state
        total_reward += reward
    print(f"Episode: {episode+1}/{episodes}, Total Reward: {total_reward}")

print("Learned Q-Table:")
print(agent.q_table)