import numpy as np


class Agent:
    def __init__(
        self,
        alpha: float,
        gamma: float,
        epsilon_max: float,
        epsilon_min: float,
        epsilon_decrement: float,
        num_actions: int,
        num_states: int,
    ) -> None:
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount Factor
        self.epsilon = epsilon_max  # Using Epsilon greedy for exploration
        self.epsilon_min = epsilon_min
        self.num_actions = num_actions
        self.num_states = num_states
        self.epsilon_decrement = epsilon_decrement

        self.q_table = np.zeros((self.num_states, self.num_actions))

    def choose_action(self, state: np.array) -> int:
        # Choosing between explore on exploit using epsilon greedy
        if np.random.random() < self.epsilon:
            action = np.random.choice([i for i in range(self.num_actions)])
        else:
            actions = np.array(
                [self.q_table[state, a] for a in range(self.num_actions)]
            )
            action = np.argmax(actions)

        return action

    def update(
        self,
        last_state: np.array,
        last_reward: float,
        last_action: int,
        new_state: np.array,
    ) -> None:
        actions = np.array(
            [self.q_table[new_state, a] for a in range(self.num_actions)]
        )
        max_action = np.argmax(actions)

        # TD(a, s) = R(s, a) + (df * max(Q(s', a'))) - Q(s, a)
        # Q(s, a) = Q(s, a) + (lr * TD(a, s))
        temporal_diff = (
            last_reward
            + (self.gamma * self.q_table[new_state, max_action])
            - self.q_table[last_state, last_action]
        )
        self.q_table[last_state, last_action] += self.alpha * temporal_diff

        # Decrement the epsilon value
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decrement
