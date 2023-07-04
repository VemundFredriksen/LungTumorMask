import torch as T
import torch.nn as nn
from Networks.network import Network

class UNet5_decoder_teacher(Network):
    def __init__(self, base, expansion):
        super(UNet5_decoder_teacher, self).__init__()
        self._build(base, expansion)

    def forward(self, image, segbox):
        layer_0, layer_1, layer_2, layer_3 = self._analysis(image)
        s_0, s_1, s_2, s_3 = self._box_scale(segbox)

        inp_layer_0 = T.cat((layer_0, self.analyse_box_0(s_0)), dim = 1)
        inp_layer_1 = T.cat((layer_1, self.analyse_box_1(s_1)), dim = 1)
        inp_layer_2 = T.cat((layer_2, self.analyse_box_2(s_2)), dim = 1)
        inp_layer_3 = T.cat((layer_3, self.analyse_box_3(s_3)), dim = 1)

        return self._synthesis(inp_layer_0, inp_layer_1, inp_layer_2, inp_layer_3, self._bridge(layer_3))

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

    def _box_scale(self, b):
        s_0 = b
        s_1 = nn.functional.interpolate(s_0, scale_factor = (0.5, 0.5, 0.5))
        s_2 = nn.functional.interpolate(s_1, scale_factor = (0.5, 0.5, 0.5))
        s_3 = nn.functional.interpolate(s_2, scale_factor = (0.5, 0.5, 0.5))
        return s_0, s_1, s_2, s_3

    def _build(self, base, expansion):
        fl_0 = int(base)
        fl_1 = int(base * expansion)
        fl_2 = int(base * (expansion ** 2))
        fl_3 = int(base * (expansion ** 3))

        self.analyse_box_0 = nn.Sequential(
            nn.Conv3d(1, 64, 3, 1, 1),
            nn.BatchNorm3d(64),
            nn.Sigmoid(),
        )

        self.analyse_box_1 = nn.Sequential(
            nn.Conv3d(1, 128, 3, 1, 1),
            nn.BatchNorm3d(128),
            nn.Sigmoid(),
        )

        self.analyse_box_2 = nn.Sequential(
            nn.Conv3d(1, 256, 3, 1, 1),
            nn.BatchNorm3d(256),
            nn.Sigmoid(),
        )

        self.analyse_box_3 = nn.Sequential(
            nn.Conv3d(1, 512, 3, 1, 1),
            nn.BatchNorm3d(512),
            nn.Sigmoid(),
        )

        self.analysis_0 = nn.Sequential(
            nn.Conv3d(1, 64, 3, 1, 1),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(),
            nn.Conv3d(64, 64, 3, 1, 1),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(),
        )

        self.analysis_1 = nn.Sequential(
            nn.MaxPool3d(2),
            nn.Conv3d(64, 128, 3, 1, 1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(),
            nn.Conv3d(128, 128, 3, 1, 1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(),
        )

        self.analysis_2 = nn.Sequential(
            nn.MaxPool3d(2),
            nn.Conv3d(128, 256, 3, 1, 1),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(),
            nn.Conv3d(256, 256, 3, 1, 1),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(),
        )

        self.analysis_3 = nn.Sequential(
            nn.MaxPool3d(2),
            nn.Conv3d(256, 512, 3, 1, 1),
            nn.BatchNorm3d(512),
            nn.LeakyReLU(),
            nn.Conv3d(512, 512, 3, 1, 1),
            nn.BatchNorm3d(512),
            nn.LeakyReLU(),
        )

        self.bridge = nn.Sequential(
            nn.MaxPool3d(2),
            nn.Conv3d(512, 512, 3, 1, 1),
            nn.BatchNorm3d(512),
            nn.LeakyReLU(),
            nn.Conv3d(512, 512, 3, 1, 1),
            nn.BatchNorm3d(512),
            nn.LeakyReLU(),
            nn.ConvTranspose3d(512, 512, 2, 2, 0),
        )

        self.synthesis_3 = nn.Sequential(
            nn.Conv3d(512 + 512 + 512, 256, 3, 1, 1),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(),
            nn.Conv3d(256, 256, 3, 1, 1),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(),
            nn.ConvTranspose3d(256, 256, 2, 2, 0),
        )

        self.synthesis_2 = nn.Sequential(
            nn.Conv3d(256 + 256 + 256, 128, 3, 1, 1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(),
            nn.Conv3d(128, 128, 3, 1, 1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(),
            nn.ConvTranspose3d(128, 128, 2, 2, 0),
        )

        self.synthesis_1 = nn.Sequential(
            nn.Conv3d(128 + 128 + 128, 64, 3, 1, 1),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(),
            nn.Conv3d(64, 64, 3, 1, 1),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(),
            nn.ConvTranspose3d(64, 64, 2, 2, 0),
        )

        self.synthesis_0 = nn.Sequential(
            nn.Conv3d(64 + 64 + 64, 64, 3, 1, 1),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(),
            nn.Conv3d(64, 64, 3, 1, 1),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(),
            nn.Conv3d(64, 1, 3, 1, 1),
            nn.Sigmoid()
        )

if __name__ == "__main__":
    s = UNet_teacher()
    f = T.randn((1,1,64,64,64))
    a = s._box_scale(f)

    print(a[3].shape)
    
