import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)     
        
        # define dimensions of hidden layers
        hidden_size1 = 128
        hidden_size2 = 64
        hidden_size3 = 32
        self.fc1 = nn.Linear(state_size,hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)      
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)      
        self.fc4 = nn.Linear(hidden_size3, action_size)

        

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        # final layer without relu activation (regression problem)
        x = self.fc4(x)
        return x