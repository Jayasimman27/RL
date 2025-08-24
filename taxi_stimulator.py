# taxi learning
import gymnasium as gym
import numpy as np
import random
import time
from pathlib import Path

#env
train_env = gym.make("Taxi-v3")

state_n  = train_env.observation_space.n      # kept 500 state
action_n = train_env.action_space.n           # 6 total actions are possible 
Q = np.zeros((state_n, action_n), dtype=np.float32)

#params
alpha = 0.1           
gamma = 0.9            # discount factor
epsilon = 1.0          # start exploring a lot
epsilon_min = 0.01
epsilon_decay = 0.995
episodes = 5000      
#train
for ep in range(episodes):
    state, _ = train_env.reset()
    done = False
    while not done:
        # ε-greedy
        if random.random() < epsilon:
            action = train_env.action_space.sample()
        else:
            action = int(np.argmax(Q[state]))

        next_state, reward, terminated, truncated, _ = train_env.step(action)
        done = terminated or truncated

        # updation of the q learning for the current state
        best_next = np.max(Q[next_state])
        #Q(s,a)←Q(s,a)+α[r+γa′max​Q(s′,a′)−Q(s,a)]
        Q[state, action] += alpha * (reward + gamma * best_next - Q[state, action]) # pyright: ignore[reportOperatorIssue]

        state = next_state

    # decay exploration
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay


out = Path("qtable_taxi.npy")
np.save(out, Q)
print(f"Q-table saved to {out.resolve()}")

play_env = gym.make("Taxi-v3", render_mode="human")
state, _ = play_env.reset()
done = False
total_reward = 0
steps = 0

print("\n Playing with trained policy...\n")
while not done:
    action = int(np.argmax(Q[state]))   # greedy action from trained Q
    state, reward, terminated, truncated, _ = play_env.step(action)
    done = terminated or truncated
    total_reward += reward # type: ignore
    steps += 1

    play_env.render()
    time.sleep(0.25)  # slow down so you can see moves

print(f"\n🏁 Finished! Steps: {steps}, Total reward: {total_reward}")
play_env.close()
train_env.close()
