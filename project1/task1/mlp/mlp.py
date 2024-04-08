import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from typing import List, Tuple


# append path to parent folder to allow imports from utils folder
import sys

sys.path.append("..")
from utils.utils import ExULayer, ReLUn


class MLP(nn.Module):
    def __init__(
        self,
        in_size: int,
        out_size: int = 2,
        hidden_profile: List[int] = [1024],
        #use_exu: bool = True,
        #use_relu_n: bool = True,
        dropout: float = 0.0,
    ):
        super(MLP, self).__init__()

        #self.hidden_unit = ExULayer if use_exu else nn.Linear
        #self.activation = ReLUn if use_relu_n else nn.ReLU

        self.layers = []
        # add first hidden layer
        self.layers.append(nn.Linear(in_size, hidden_profile[0]))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Dropout(dropout))


        # loop through hidden profile and add hidden layers
        for hidden_in_size, hidden_out_size in zip(
            hidden_profile[:-1], hidden_profile[1:]
        ):
            #self.layers.append(self.hidden_unit(hidden_in_size, hidden_out_size))
            #self.layers.append(self.activation())
            #self.layers.append(nn.Dropout(dropout))
            
            self.layers.append(nn.Linear(hidden_in_size, hidden_out_size))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout))

        # add output layer
        self.layers.append(nn.Linear(hidden_profile[-1], out_size))

        self.nn = nn.Sequential(*self.layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # no sigmoid for numerical stability and use nn.BCEWithLogitsLoss
        # for integrated log-sum-exp trick
        return self.nn(x.unsqueeze(1))
    
