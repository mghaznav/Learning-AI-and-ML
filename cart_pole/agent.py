import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from network import Network
from agent_memory import AgentMemory


class Agent:
    def __init__(
        self,
        alpha: float,
        gamma: float,
        epsilon_max: float,
        epsilon_min: float,
        epsilon_decrement: float,
        num_actions: int,
        input_dims: int,
    ) -> None:
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount Factor
        self.epsilon = epsilon_max  # Using Epsilon greedy for exploration
        self.epsilon_min = epsilon_min
        self.epsilon_decrement = epsilon_decrement
        self.num_actions = num_actions
        self.input_dims = input_dims
        self.action_space = [i for i in range(num_actions)]

        self.DQL = Network(alpha, num_actions, input_dims)

        self.memory = AgentMemory(10000)
        self.device = self.DQL.device

    def choose_action(self, observation):
        # Choosing between explore on exploit using epsilon greedy
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            # Converting observation to the correct tensor type (cpu vs gpu)
            state = T.tensor(observation, device=self.device, dtype=T.float)
            actions = self.DQL.forward(state)
            action = T.argmax(actions).item()

        return action

    def update(self, last_state, last_reward, last_action, new_state):
        # self.DQL.optimizer.zero_grad()

        # Converting data from numpy arrays to tensors
        states = T.tensor(last_state, device=self.device, dtype=T.float).unsqueeze(0)
        actions = T.tensor([last_action], device=self.device)
        rewards = T.tensor([last_reward], device=self.device)
        new_states = T.tensor(new_state, device=self.device, dtype=T.float).unsqueeze(0)

        # Add transition to agent memory
        self.memory.update((states, actions, rewards, new_states))

        # Begin learning as soon as we have enough memory
        if len(self.memory.memory) > 128:
            (
                batch_state,
                batch_action,
                batch_reward,
                batch_new_state,
            ) = self.memory.sample(128)

            self.learn(batch_state, batch_action, batch_reward, batch_new_state)

        # Update the network without experience replay
        # q_pred = self.DQL.forward(states)[actions]
        # q_next = self.DQL.forward(new_states).max()
        # q_target = last_reward + self.gamma * q_next

        # loss = self.DQL.loss(q_target, q_pred).to(self.device)
        # loss.backward()
        # self.DQL.optimizer.step()

        # Decrement epsilon each time
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decrement

    def learn(self, batch_state, batch_action, batch_reward, batch_new_state):
        outputs = self.DQL(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        next_outputs = self.DQL(batch_new_state).detach().max(1)[0]
        target = self.gamma * next_outputs + batch_reward
        loss = self.DQL.loss(target, outputs).to(self.device)
        self.DQL.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.DQL.optimizer.step()
