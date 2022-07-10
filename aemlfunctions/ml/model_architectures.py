"""

model_architectures

Architecture parameters are varied here. To help track, label ## each change.

Nick Tulshibagwale

"""
from torch import nn

class NeuralNetwork_01(nn.Module): 
    # classification
    def __init__(self, feature_dim, classes):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(feature_dim, 20),
            nn.ReLU(),
            nn.Linear(20,classes)
        )

    def forward(self, x):
        z = self.layers(x)
        return z # predictions

class NeuralNetwork_02(nn.Module): 
    # regression
    def __init__(self, feature_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(feature_dim, 50),
            nn.ReLU(),
            nn.Linear(50,1)
        )
        
    def forward(self, x):
        z = self.layers(x)
        z = z.flatten()
        return z # predictions   