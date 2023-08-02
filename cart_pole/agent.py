import os
import numpy as np
import torch as T
from network import Network
from agent_memory import AgentMemory, Transition


class Agent:
    def __init__(
        self,
        alpha: float,
        gamma: float,
        tau: float,
        epsilon_max: float,
        epsilon_min: float,
        epsilon_decrement: float,
        num_actions: int,
        input_dims: int,
    ) -> None:
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount Factor
        self.tau = tau  # Update rate
        self.epsilon = epsilon_max  # Using Epsilon greedy for exploration
        self.epsilon_min = epsilon_min
        self.epsilon_decrement = epsilon_decrement
        self.num_actions = num_actions
        self.input_dims = input_dims
        self.action_space = [i for i in range(num_actions)]

        self.q_policy = Network(alpha, num_actions, input_dims)
        self.q_target = Network(alpha, num_actions, input_dims)
        self.q_target.load_state_dict(self.q_policy.state_dict())

        self.memory = AgentMemory(1000)
        self.batch_size = 128
        self.device = self.q_policy.device

    def choose_action(self, observation: np.array) -> T.int64:
        # Choosing between explore or exploit using epsilon greedy
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            # Converting observation to the correct tensor type (cpu vs gpu)
            state = T.tensor(observation, device=self.device, dtype=T.float)
            actions = self.q_policy.forward(state)
            action = T.argmax(actions).item()

        return action

    def update(
        self,
        last_state: np.array,
        last_reward: np.float32,
        last_action: T.int64,
        new_state: np.array,
    ) -> None:
        # Converting data from numpy arrays to tensors
        states = T.tensor(last_state, device=self.device, dtype=T.float).unsqueeze(0)
        actions = T.tensor([last_action], device=self.device)
        rewards = T.tensor([last_reward], device=self.device)
        new_states = (
            T.tensor(new_state, device=self.device, dtype=T.float).unsqueeze(0)
            if new_state is not None
            else None
        )

        # Add transition to agent memory
        self.memory.update(states, actions, rewards, new_states)

        # Begin learning as soon as we have enough memory
        self.learn()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = self.q_target.state_dict()
        policy_net_state_dict = self.q_policy.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[
                key
            ] * self.tau + target_net_state_dict[key] * (1 - self.tau)
        self.q_target.load_state_dict(target_net_state_dict)

        # Decrement epsilon each time
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decrement

    def learn(self) -> None:
        if len(self.memory.memory) < self.batch_size:
            return

        batch = Transition(*zip(*self.memory.sample(self.batch_size)))

        # Create a mask of all non final new states
        non_final_mask = T.tensor(
            tuple(map(lambda s: s is not None, batch.new_state)),
            device=self.device,
            dtype=bool,
        )
        non_final_new_states = T.cat([s for s in batch.new_state if s is not None])

        # Collect all the other batch's
        batch_state = T.cat(batch.state)
        batch_action = T.cat(batch.action)
        batch_reward = T.cat(batch.reward)

        # Get Q-values for selected actions
        outputs = (
            self.q_policy(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        )

        # Get Max Q-values for the new states setting final states to 0
        next_outputs = T.zeros(self.batch_size, device=self.device)
        with T.no_grad():
            next_outputs[non_final_mask] = (
                self.q_target(non_final_new_states).detach().max(1)[0]
            )

        # Get a 1D array of target values
        target = self.gamma * next_outputs + batch_reward

        # Calculate the loss and back propogate
        loss = self.q_policy.loss(target, outputs).to(self.device)
        self.q_policy.optimizer.zero_grad()
        loss.backward()
        self.q_policy.optimizer.step()

    def save(self, filename):
        file_count = self.file_count()
        T.save(
            {
                "policy_state_dict": self.q_policy.state_dict(),
                "target_state_dict": self.q_target.state_dict(),
                "policy_optimizer": self.q_policy.optimizer.state_dict(),
                "target_optimizer": self.q_target.optimizer.state_dict(),
            },
            f"./data/{file_count}_{filename}.pth",
        )
        return f"{file_count}_{filename}.pth"

    def load(self, filename):
        if os.path.isfile(f"./data/{filename}.pth"):
            print("=> loading checkpoint... ")
            checkpoint = T.load(f"./data/{filename}.pth")
            self.q_policy.load_state_dict(checkpoint["policy_state_dict"])
            self.q_policy.optimizer.load_state_dict(checkpoint["policy_optimizer"])
            self.q_target.load_state_dict(checkpoint["target_state_dict"])
            self.q_target.optimizer.load_state_dict(checkpoint["target_optimizer"])
            print("=> checkpoint loaded !")
        else:
            print("no checkpoint found...")

    def file_count(self):
        # folder path
        dir_path = "./data/"
        count = 0
        # Iterate directory
        for path in os.listdir(dir_path):
            # check if current path is a file
            if os.path.isfile(os.path.join(dir_path, path)):
                count += 1
        return count
