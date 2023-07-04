import torch as T

class Network(T.nn.Module):
    def __init__(self):
        super(Network, self).__init__()

    def forward(self, image, segbox):
        pass

    def train_step(self, image, segment, criterion, segbox):
        pass