"""

model_architectures

Architecture parameters are varied here. To help track, label ## each change.

Nick Tulshibagwale

"""
from torch import nn

class NeuralNetwork_01(nn.Module): # 1 layer
    # classification
    def __init__(self, feature_dim, classes, hidden_units):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(feature_dim, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units,classes)
        )

    def forward(self, x):
        z = self.layers(x)
        return z # predictions

class NeuralNetwork_02(nn.Module): # 2 layer
    # classification
    def __init__(self, feature_dim, classes, hidden_units):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(feature_dim, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units,classes)
        )

    def forward(self, x):
        z = self.layers(x)
        return z # predictions

class NeuralNetwork_03(nn.Module): # 3 layer
    # classification
    def __init__(self, feature_dim, classes, hidden_units):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(feature_dim, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units,classes)
        )

    def forward(self, x):
        z = self.layers(x)
        return z # predictions
    
class NeuralNetwork_04(nn.Module): 
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