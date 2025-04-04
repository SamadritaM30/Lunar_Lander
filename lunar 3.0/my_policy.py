import numpy as np
import torch
import torch.nn as nn

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim=8, output_dim=4):
        super(PolicyNetwork, self).__init__()
        hidden_size=256
        self.actor=nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_dim),
            nn.Softmax(dim=-1)
        )
    def forward(self, x):
        return self.actor(x)

def load_policy_from_numpy(params):
    model=PolicyNetwork()
    idx= 0
    with torch.no_grad():
        for param in model.parameters():
            num_params =param.numel()
            param.copy_(torch.tensor(params[idx:idx + num_params]).view_as(param))
            idx += num_params
    model.eval()
    return model

def policy_action(policy, observation):
    model= load_policy_from_numpy(policy)
    state_tensor=torch.FloatTensor(observation).unsqueeze(0)
    with torch.no_grad():
        action_probs= model(state_tensor)
    return torch.argmax(action_probs, dim=-1).item()
