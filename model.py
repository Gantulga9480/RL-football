from torch import nn


DROP_OUT = 0.3
acivation = nn.Tanh


class DQN(nn.Module):

    def __init__(self, input_shape, output_shape) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_shape, 64),
            acivation(),
            nn.Dropout(DROP_OUT),
            nn.Linear(64, 32),
            acivation(),
            nn.Dropout(DROP_OUT),
            nn.Linear(32, 16),
            acivation(),
            nn.Dropout(DROP_OUT),
            nn.Linear(16, output_shape)
        )

    def forward(self, x):
        return self.model(x)
