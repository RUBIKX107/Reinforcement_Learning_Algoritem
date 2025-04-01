# Reinforcement_Learning_Algoritem

📂 Reinforcement_Learning_Algorithms │ 
├── 📜 README.md  # Detailed explanation of Q-learning & SARSA
├── 📂 notebooks │  
├── 📜 q_learning_example.ipynb  # Jupyter Notebook with step-by-step guide for Q-learning │  
├── 📜 sarsa_example.ipynb  # Jupyter Notebook with step-by-step guide for SARSA ├── 📂 code │   
├── 📜 q_learning.py  # Full Python implementation of Q-learning │  
├── 📜 sarsa.py  # Full Python implementation of SARSA 
├── 📂 results │   
├── 📜 q_learning_rewards.csv  # CSV file storing rewards per episode for Q-learning │   
├── 📜 sarsa_rewards.csv  # CSV file storing rewards per episode for SARSA │   ├── 📜 training_plots.png  # Visualization of the training process └── 📜 .gitignore  # Ignore unnecessary files like pycache and checkpoints


import gym
import numpy as np 
import random 
import matplotlib.pyplot as plt
import pandas as pd

#Initialize environment

env = gym.make("FrozenLake-v1", is_slippery=True, render_mode=None) num_states = env.observation_space.n num_actions = env.action_space.n

#Q-table initialization

q_table = np.zeros((num_states, num_actions))

#Hyperparameters

alpha = 0.1  # Learning rate 
gamma = 0.99  # Discount factor 
epsilon = 1.0  # Exploration rate epsilon_decay = 0.995
epsilon_min = 0.01 
num_episodes = 10000

#Tracking rewards

rewards_per_episode = []

#Q-learning algorithm

for episode in range(num_episodes): 
  state = env.reset()
  total_reward = 0 done = False

while not done:
    # Choose action (epsilon-greedy strategy)
    if random.uniform(0, 1) < epsilon:
        action = env.action_space.sample()  # Explore
    else:
        action = np.argmax(q_table[state, :])  # Exploit
    
    # Take action
    
    next_state, reward, done, info = env.step(action)
    
    # Q-learning update rule
    
    q_table[state, action] = q_table[state, action] + alpha * (reward + gamma * np.max(q_table[next_state, :]) - q_table[state, action])
    
    state = next_state
    total_reward += reward

# Decay epsilon
epsilon = max(epsilon * epsilon_decay, epsilon_min)
rewards_per_episode.append(total_reward)

if episode % 1000 == 0:
    print(f"Episode {episode}, Reward: {total_reward}")

#Save results

pd.DataFrame(rewards_per_episode, columns=["Reward"]).to_csv("results/q_learning_rewards.csv", index=False)

#Plot rewards

plt.plot(rewards_per_episode) plt.xlabel("Episodes") plt.ylabel("Reward") plt.title("Q-learning Training Performance") plt.savefig("results/training_plots.png") plt.show()

print("Training completed!")

