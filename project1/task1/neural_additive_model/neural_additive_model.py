import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from typing import List


# append path to parent folder to allow imports from utils folder
import sys

sys.path.append("..")
from utils.utils import ExULayer, ReLUn


class FeatureNet(nn.Module):
    def __init__(
        self,
        in_size: int,
        out_size: int = 2,
        hidden_profile: List[int] = [1024],
        use_exu: bool = True,
        use_relu_n: bool = True,
        dropout: float = 0.0,
    ):
        super(FeatureNet, self).__init__()

        self.hidden_unit = ExULayer if use_exu else nn.Linear
        self.activation = ReLUn if use_relu_n else nn.ReLU

        self.layers = []
        # add first hidden layer
        self.layers.append(self.hidden_unit(in_size, hidden_profile[0]))
        self.layers.append(self.activation())
        self.layers.append(nn.Dropout(dropout))

        # loop through hidden profile and add hidden layers
        for hidden_in_size, hidden_out_size in zip(
            hidden_profile[:-1], hidden_profile[1:]
        ):
            self.layers.append(self.hidden_unit(hidden_in_size, hidden_out_size))
            self.layers.append(self.activation())
            self.layers.append(nn.Dropout(dropout))

        # add output layer
        self.layers.append(nn.Linear(hidden_profile[-1], out_size))

        self.nn = nn.Sequential(*self.layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # no sigmoid for numerical stability and use nn.BCEWithLogitsLoss
        # for integrated log-sum-exp trick
        return self.nn(x.unsqueeze(1))


class NAM(nn.Module):
    def __init__(
        self,
        n_features: int,
        in_size: int,
        out_size: int = 1,
        hidden_profile: List[int] = [1024],
        use_exu: bool = True,
        use_relu_n: bool = True,
        within_feature_dropout: float = 0.2,
        feature_dropout: float = 0.0,
    ):
        super(NAM, self).__init__()

        self.feature_nets = nn.ModuleList(
            [
                FeatureNet(
                    in_size=in_size,
                    out_size=out_size,
                    hidden_profile=hidden_profile,
                    use_exu=use_exu,
                    use_relu_n=use_relu_n,
                    dropout=within_feature_dropout,
                )
                for _ in range(n_features)
            ]
        )

        # dropout layer for features
        self.feature_dropout = nn.Dropout(feature_dropout)

        # bias term to add
        self.bias = Parameter(torch.Tensor(1))
        nn.init.constant_(self.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # !!! forward method inspired by:
        # https://github.com/kherud/neural-additive-models-pt/blob/master/nam/model.py
        single_logits = [
            feature_net(input) for feature_net, input in zip(self.feature_nets, x.T)
        ]
        # concatenate logits and add feature dropout
        concat_logits = self.feature_dropout(torch.concat(single_logits, dim=1))

        return torch.sum(concat_logits, dim=1) + self.bias
