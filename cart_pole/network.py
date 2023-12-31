import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch as T


class Network(nn.Module):
    def __init__(self, alpha: float, num_actions: int, input_dims: list) -> None:
        super(Network, self).__init__()

        self.fc1 = nn.Linear(*input_dims, 128)
        self.fc2 = nn.Linear(128, num_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.loss = nn.MSELoss()

        # Use the GPU if available, otherwise default to CPU
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state: list) -> T.tensor:
        layer1 = F.relu(self.fc1(state))
        actions = self.fc2(layer1)

        return actions
