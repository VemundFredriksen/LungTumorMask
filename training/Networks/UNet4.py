import torch as T
import torch.nn as nn
from Networks.network import Network

class UNet4(Network):
    def __init__(self, base, expansion):
        super(UNet4, self).__init__()
        self._build(base, expansion)

    def forward(self, image, segbox=None):
        layer_0, layer_1, layer_2 = self._analysis(image)
        return self._synthesis(layer_0, layer_1, layer_2, self._bridge(layer_2))

    def train_step(self, image, segment, criterion, segbox = None):
        output = self.forward(image)

        loss = criterion(output, segment)
        return loss

    def _analysis(self, x):
        layer_0 = self.analysis_0(x)
        layer_1 = self.analysis_1(layer_0)
        layer_2 = self.analysis_2(layer_1)

        return layer_0, layer_1, layer_2

    def _bridge(self, layer_2):
        return self.bridge(layer_2)

    def _synthesis(self, layer_0, layer_1, layer_2, layer_3):
        concat_2 = T.cat((layer_2, layer_3), dim = 1)
        concat_1 = T.cat((layer_1, self.synthesis_2(concat_2)), dim = 1)
        concat_0 = T.cat((layer_0, self.synthesis_1(concat_1)), dim = 1)

        return self.synthesis_0(concat_0)

    def _build(self, base, expansion):
        layer_0_count = int(base)
        layer_1_count = int(base * expansion)
        layer_2_count = int(base * (expansion ** 2))
        layer_3_count = int(base * (expansion ** 3))

        self.analysis_0 = nn.Sequential(
            nn.Conv3d(1, layer_0_count, 3, 1, 1),
            nn.BatchNorm3d(layer_0_count),
            nn.LeakyReLU(),
            nn.Conv3d(layer_0_count, layer_0_count, 3, 1, 1),
            nn.BatchNorm3d(layer_0_count),
            nn.LeakyReLU(),
        )

        self.analysis_1 = nn.Sequential(
            nn.MaxPool3d(2),
            nn.Conv3d(layer_0_count, layer_1_count, 3, 1, 1),
            nn.BatchNorm3d(layer_1_count),
            nn.LeakyReLU(),
            nn.Conv3d(layer_1_count, layer_1_count, 3, 1, 1),
            nn.BatchNorm3d(layer_1_count),
            nn.LeakyReLU(),
        )

        self.analysis_2 = nn.Sequential(
            nn.MaxPool3d(2),
            nn.Conv3d(layer_1_count, layer_2_count, 3, 1, 1),
            nn.BatchNorm3d(layer_2_count),
            nn.LeakyReLU(),
            nn.Conv3d(layer_2_count, layer_2_count, 3, 1, 1),
            nn.BatchNorm3d(layer_2_count),
            nn.LeakyReLU(),
        )

        self.bridge = nn.Sequential(
            nn.MaxPool3d(2),
            nn.Conv3d(layer_2_count, layer_3_count, 3, 1, 1),
            nn.BatchNorm3d(layer_3_count),
            nn.LeakyReLU(),
            nn.Conv3d(layer_3_count, layer_3_count, 3, 1, 1),
            nn.BatchNorm3d(layer_3_count),
            nn.LeakyReLU(),
            nn.ConvTranspose3d(layer_3_count, layer_3_count, 2, 2, 0)
        )

        self.synthesis_2 = nn.Sequential(
            nn.Conv3d(layer_2_count + layer_3_count, layer_2_count, 3, 1, 1),
            nn.BatchNorm3d(layer_2_count),
            nn.LeakyReLU(),
            nn.Conv3d(layer_2_count,layer_2_count, 3, 1, 1),
            nn.BatchNorm3d(layer_2_count),
            nn.LeakyReLU(),
            nn.ConvTranspose3d(layer_2_count, layer_2_count, 2, 2, 0)
        )

        self.synthesis_1 = nn.Sequential(
            nn.Conv3d(layer_1_count + layer_2_count, layer_1_count, 3, 1, 1),
            nn.BatchNorm3d(layer_1_count),
            nn.LeakyReLU(),
            nn.Conv3d(layer_1_count,layer_1_count, 3, 1, 1),
            nn.BatchNorm3d(layer_1_count),
            nn.LeakyReLU(),
            nn.ConvTranspose3d(layer_1_count, layer_1_count, 2, 2, 0)
        )

        self.synthesis_0 = nn.Sequential(
            nn.Conv3d(layer_0_count + layer_1_count, layer_0_count, 3, 1, 1),
            nn.BatchNorm3d(layer_0_count),
            nn.LeakyReLU(),
            nn.Conv3d(layer_0_count,layer_0_count, 3, 1, 1),
            nn.BatchNorm3d(layer_0_count),
            nn.LeakyReLU(),
            nn.Conv3d(layer_0_count, layer_0_count, 3, 1, 1),
            nn.BatchNorm3d(layer_0_count),
            nn.LeakyReLU(),
            nn.Conv3d(layer_0_count, 1, 3, 1, 1),
            nn.Sigmoid()
        )
