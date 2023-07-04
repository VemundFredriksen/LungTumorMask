import torch as T
import torch.nn as nn
from Networks.network import Network

class UNet5_Teacher(Network):
    def __init__(self, base, expansion):
        super(UNet5_Teacher, self).__init__()
        self._build(base, expansion)

    def forward(self, image, segbox):
        x_new = T.cat((image, segbox), dim = 1)
        layer_0, layer_1, layer_2, layer_3 = self._analysis(x_new)

        return self._synthesis(layer_0, layer_1, layer_2, layer_3, self._bridge(layer_3))

    def train_step(self, image, segment, criterion, segbox = None):
        output = self.forward(image, segbox)

        loss = criterion(output, segment)
        return loss

    def _analysis(self, x):
        layer_0 = self.analysis_0(x)
        layer_1 = self.analysis_1(layer_0)
        layer_2 = self.analysis_2(layer_1)
        layer_3 = self.analysis_3(layer_2)

        return layer_0, layer_1, layer_2, layer_3

    def _bridge(self, layer_3):
        return self.bridge(layer_3)

    def _synthesis(self, l0, l1, l2, l3, l4):
        c_3 = T.cat((l3, l4), dim = 1)
        c_2 = T.cat((l2, self.synthesis_3(c_3)), dim = 1)
        c_1 = T.cat((l1, self.synthesis_2(c_2)), dim = 1)
        c_0 = T.cat((l0, self.synthesis_1(c_1)), dim = 1)

        return self.synthesis_0(c_0)

    def _build(self, base, expansion):
        fl_0 = int(base)
        fl_1 = int(base * expansion)
        fl_2 = int(base * (expansion ** 2))
        fl_3 = int(base * (expansion ** 3))
        fl_4 = int(base * (expansion ** 4))

        self.analysis_0 = nn.Sequential(
            nn.Conv3d(2, fl_0, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Conv3d(fl_0, fl_0, 3, 1, 1),
            nn.LeakyReLU(),
        )

        self.analysis_1 = nn.Sequential(
            nn.MaxPool3d(2),
            nn.Conv3d(fl_0, fl_1, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Conv3d(fl_1, fl_1, 3, 1, 1),
            nn.LeakyReLU(),
        )

        self.analysis_2 = nn.Sequential(
            nn.MaxPool3d(2),
            nn.Conv3d(fl_1, fl_2, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Conv3d(fl_2, fl_2, 3, 1, 1),
            nn.LeakyReLU(),
        )

        self.analysis_3 = nn.Sequential(
            nn.MaxPool3d(2),
            nn.Conv3d(fl_2, fl_3, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Conv3d(fl_3, fl_3, 3, 1, 1),
            nn.LeakyReLU(),
        )

        self.bridge = nn.Sequential(
            nn.MaxPool3d(2),
            nn.Conv3d(fl_3, fl_3, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Conv3d(fl_3, fl_3, 3, 1, 1),
            nn.LeakyReLU(),
            nn.ConvTranspose3d(fl_3, fl_3, 2, 2, 0),
        )

        self.synthesis_3 = nn.Sequential(
            nn.Conv3d(fl_3 + fl_3, fl_2, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Conv3d(fl_2, fl_2, 3, 1, 1),
            nn.LeakyReLU(),
            nn.ConvTranspose3d(fl_2, fl_2, 2, 2, 0),
        )

        self.synthesis_2 = nn.Sequential(
            nn.Conv3d(fl_2 + fl_2, fl_1, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Conv3d(fl_1, fl_1, 3, 1, 1),
            nn.LeakyReLU(),
            nn.ConvTranspose3d(fl_1, fl_1, 2, 2, 0),
        )

        self.synthesis_1 = nn.Sequential(
            nn.Conv3d(fl_1 + fl_1, fl_0, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Conv3d(fl_0, fl_0, 3, 1, 1),
            nn.LeakyReLU(),
            nn.ConvTranspose3d(fl_0, fl_0, 2, 2, 0),
        )

        self.synthesis_0 = nn.Sequential(
            nn.Conv3d(fl_0 + fl_0, fl_0, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Conv3d(fl_0, fl_0, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Conv3d(fl_0, 1, 3, 1, 1),
            nn.Conv3d(1, 1, 1, 1, 0),
            nn.Sigmoid()
        )
    
