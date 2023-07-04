import torch as T
import torch.nn as nn
from Networks.network import Network
from monai.networks.nets.unet import UNet

class UNet_student(Network):
    def __init__(self):
        super(UNet_student, self).__init__()
        #self._build()
        self.model = T.nn.Sequential(UNet(3, 1, 1, (32, 64, 128, 256), (2, 2, 2)), T.nn.Sigmoid())

    def forward(self, image):
        return self.model(image)
    
    def train_step(self, image, segment, criterion, segbox):
        fo = self.forward(image)
        loss = criterion(image, segment)
        return loss


    """
    def forward(self, image, segbox = None, eval_mode = True):


        layer_0, layer_1, layer_2, layer_3 = self._analysis(image)

        inter_output, s_3, s_2, s_1 = self._synthesis(layer_0, layer_1, layer_2, layer_3, self._bridge(layer_3))
        seg_out = self.output_0(inter_output)

        u_0 = self._upsample(s_3, s_2, s_1, inter_output)
        box_out = self.output_1(u_0)

        if (eval_mode):
            return seg_out
        return seg_out, box_out

    def train_step(self, image, segment, criterion, segbox = None):
        seg_out, box_out = self.forward(image, eval_mode=False)

        seg_loss = criterion(seg_out, segment)
        box_loss = criterion(box_out, segbox)
        return seg_loss + box_loss

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
        s_3 = self.synthesis_3(c_3)
        c_2 = T.cat((l2, self.up_conv_3(s_3)), dim = 1)
        s_2 = self.synthesis_2(c_2)
        c_1 = T.cat((l1, self.up_conv_2(s_2)), dim = 1)
        s_1 = self.synthesis_1(c_1)
        c_0 = T.cat((l0, self.up_conv_1(s_1)), dim = 1)

        return self.synthesis_0(c_0), s_3, s_2, s_1

    def _upsample(self, s_3, s_2, s_1, inter):
        u_3 = self.up_sample_3(s_3)
        u_2 = self.up_sample_2(s_2)
        u_1 = self.up_sample_1(s_1)

        return T.cat((u_3, u_2, u_1, inter), dim = 1)

    def _build(self):
        self.analysis_0 = nn.Sequential(
            nn.Conv3d(1, 16, 3, 1, 1),
            nn.BatchNorm3d(16),
            nn.LeakyReLU(),
            nn.Conv3d(16, 24, 3, 1, 1),
            nn.BatchNorm3d(24),
            nn.LeakyReLU(),
        )

        self.analysis_1 = nn.Sequential(
            nn.MaxPool3d(2),
            nn.Conv3d(24, 40, 3, 1, 1),
            nn.BatchNorm3d(40),
            nn.LeakyReLU(),
            nn.Conv3d(40, 48, 3, 1, 1),
            nn.BatchNorm3d(48),
            nn.LeakyReLU(),
        )

        self.analysis_2 = nn.Sequential(
            nn.MaxPool3d(2),
            nn.Conv3d(48, 64, 3, 1, 1),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(),
            nn.Conv3d(64, 86, 3, 1, 1),
            nn.BatchNorm3d(86),
            nn.LeakyReLU(),
        )

        self.analysis_3 = nn.Sequential(
            nn.MaxPool3d(2),
            nn.Conv3d(86, 110, 3, 1, 1),
            nn.BatchNorm3d(110),
            nn.LeakyReLU(),
            nn.Conv3d(110, 128, 3, 1, 1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(),
        )

        self.bridge = nn.Sequential(
            nn.MaxPool3d(2),
            nn.Conv3d(128, 128, 3, 1, 1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(),
            nn.Conv3d(128, 128, 3, 1, 1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(),
            nn.ConvTranspose3d(128, 128, 2, 2, 0),
        )

        self.synthesis_3 = nn.Sequential(
            nn.Conv3d(128 + 128, 110, 3, 1, 1),
            nn.BatchNorm3d(110),
            nn.LeakyReLU(),
            nn.Conv3d(110, 86, 3, 1, 1),
            nn.BatchNorm3d(86),
            nn.LeakyReLU(),
        )

        self.synthesis_2 = nn.Sequential(
            nn.Conv3d(86 + 86, 64, 3, 1, 1),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(),
            nn.Conv3d(64, 48, 3, 1, 1),
            nn.BatchNorm3d(48),
            nn.LeakyReLU(),
        )

        self.synthesis_1 = nn.Sequential(
            nn.Conv3d(48 + 48, 40, 3, 1, 1),
            nn.BatchNorm3d(40),
            nn.LeakyReLU(),
            nn.Conv3d(40, 24, 3, 1, 1),
            nn.BatchNorm3d(24),
            nn.LeakyReLU(),
        )

        self.up_conv_3 = nn.Sequential(
            nn.ConvTranspose3d(86, 86, 2, 2, 0),
        )

        self.up_conv_2 = nn.Sequential(
            nn.ConvTranspose3d(48, 48, 2, 2, 0),
        )

        self.up_conv_1 = nn.Sequential(
            nn.ConvTranspose3d(24, 24, 2, 2, 0),
        )

        self.synthesis_0 = nn.Sequential(
            nn.Conv3d(24 + 24, 16, 3, 1, 1),
            nn.BatchNorm3d(16),
            nn.LeakyReLU(),
            nn.Conv3d(16, 16, 3, 1, 1),
            nn.BatchNorm3d(16),
            nn.LeakyReLU(),
        )

        self.output_0 = nn.Sequential(
            nn.Conv3d(16, 8, 3, 1, 1),
            nn.BatchNorm3d(8),
            nn.LeakyReLU(),
            nn.Conv3d(8, 1, 3, 1, 1),
            nn.Sigmoid()
        )

        self.up_sample_3 = nn.Sequential(
            nn.ConvTranspose3d(86, 32, 8, 8, 0),
        )

        self.up_sample_2 = nn.Sequential(
            nn.ConvTranspose3d(48, 16, 4, 4, 0),
        )

        self.up_sample_1 = nn.Sequential(
            nn.ConvTranspose3d(24, 8, 2, 2, 0),
        )

        self.output_1 = nn.Sequential(
            nn.Conv3d(72, 64, 3, 1, 1),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(),
            nn.Conv3d(64, 64, 3, 1, 1),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(),
            nn.Conv3d(64, 32, 3, 1, 1),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(),
            nn.Conv3d(32, 1, 3, 1, 1),
            nn.Sigmoid()
        )
    """