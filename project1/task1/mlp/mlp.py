import torch
import torch.nn as nn
from typing import List, Tuple


# append path to parent folder to allow imports from utils folder
import sys

sys.path.append("..")


class MLP(nn.Module):
    def __init__(
        self,
        in_size: int,
        out_size: int = 2,
        hidden_profile: List[int] = [1024],
        dropout: float = 0.0,
    ):
        super(MLP, self).__init__()

        self.layers = []
        # add first hidden layer
        self.layers.append(nn.Linear(in_size, hidden_profile[0]))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Dropout(dropout))

        # loop through hidden profile and add hidden layers
        for hidden_in_size, hidden_out_size in zip(
            hidden_profile[:-1], hidden_profile[1:]
        ):            
            self.layers.append(nn.Linear(hidden_in_size, hidden_out_size))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout))

        # add output layer
        self.layers.append(nn.Linear(hidden_profile[-1], out_size))

        self.nn = nn.Sequential(*self.layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.nn(x)
        return x.squeeze()  # Squeeze the output here to ensure correct shape for BCEWithLogitsLoss
    


    
