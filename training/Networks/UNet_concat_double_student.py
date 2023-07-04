import torch as T
import torch.nn as nn
from Networks.network import Network

class UNet_concat_student(Network):
    def __init__(self, base, expansion):
        super(UNet_concat_student, self).__init__()
        self._build(base, expansion)

    def forward(self, image, segbox = None,  eval_mode = True):
        layer_0, layer_1, layer_2, layer_3 = self._analysis(image)

        inter_output, s_3, s_2, s_1 = self._synthesis(layer_0, layer_1, layer_2, layer_3, self._bridge(layer_3))
        output_0 = self.output_0(inter_output)

        u_0 = self._upsample(s_3, s_2, s_1, inter_output)
        output_1 = self.output_1(u_0)

        if (eval_mode):
            return output_0
        return output_0, output_1

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

    def _build(self, base, expansion):
        fl_i1 = int(base / (expansion ** 2))
        fl_i0 = int(base / expansion)
        fl_0 = int(base)
        fl_1 = int(base * expansion)
        fl_2 = int(base * (expansion ** 2))
        fl_3 = int(base * (expansion ** 3))

        self.analysis_0 = nn.Sequential(
            nn.Conv3d(1, fl_0, 3, 1, 1),
            nn.BatchNorm3d(fl_0),
            nn.LeakyReLU(),
            nn.Conv3d(fl_0, fl_0, 3, 1, 1),
            nn.BatchNorm3d(fl_0),
            nn.LeakyReLU(),
        )

        self.analysis_1 = nn.Sequential(
            nn.MaxPool3d(2),
            nn.Conv3d(fl_0, fl_1, 3, 1, 1),
            nn.BatchNorm3d(fl_1),
            nn.LeakyReLU(),
            nn.Conv3d(fl_1, fl_1, 3, 1, 1),
            nn.BatchNorm3d(fl_1),
            nn.LeakyReLU(),
        )

        self.analysis_2 = nn.Sequential(
            nn.MaxPool3d(2),
            nn.Conv3d(fl_1, fl_2, 3, 1, 1),
            nn.BatchNorm3d(fl_2),
            nn.LeakyReLU(),
            nn.Conv3d(fl_2, fl_2, 3, 1, 1),
            nn.BatchNorm3d(fl_2),
            nn.LeakyReLU(),
        )

        self.analysis_3 = nn.Sequential(
            nn.MaxPool3d(2),
            nn.Conv3d(fl_2, fl_3, 3, 1, 1),
            nn.BatchNorm3d(fl_3),
            nn.LeakyReLU(),
            nn.Conv3d(fl_3, fl_3, 3, 1, 1),
            nn.BatchNorm3d(fl_3),
            nn.LeakyReLU(),
        )

        self.bridge = nn.Sequential(
            nn.MaxPool3d(2),
            nn.Conv3d(fl_3, fl_3, 3, 1, 1),
            nn.BatchNorm3d(fl_3),
            nn.LeakyReLU(),
            nn.Conv3d(fl_3, fl_3, 3, 1, 1),
            nn.BatchNorm3d(fl_3),
            nn.LeakyReLU(),
            nn.ConvTranspose3d(fl_3, fl_3, 2, 2, 0),
        )

        self.synthesis_3 = nn.Sequential(
            nn.Conv3d(fl_3 + fl_3, fl_2, 3, 1, 1),
            nn.BatchNorm3d(fl_2),
            nn.LeakyReLU(),
            nn.Conv3d(fl_2, fl_2, 3, 1, 1),
            nn.BatchNorm3d(fl_2),
            nn.LeakyReLU(),
        )

        self.synthesis_2 = nn.Sequential(
            nn.Conv3d(fl_2 + fl_2, fl_1, 3, 1, 1),
            nn.BatchNorm3d(fl_1),
            nn.LeakyReLU(),
            nn.Conv3d(fl_1, fl_1, 3, 1, 1),
            nn.BatchNorm3d(fl_1),
            nn.LeakyReLU(),
        )

        self.synthesis_1 = nn.Sequential(
            nn.Conv3d(fl_1 + fl_1, fl_0, 3, 1, 1),
            nn.BatchNorm3d(fl_0),
            nn.LeakyReLU(),
            nn.Conv3d(fl_0, fl_0, 3, 1, 1),
            nn.BatchNorm3d(fl_0),
            nn.LeakyReLU(),
        )

        self.up_conv_3 = nn.Sequential(
            nn.ConvTranspose3d(fl_2, fl_2, 2, 2, 0),
        )

        self.up_conv_2 = nn.Sequential(
            nn.ConvTranspose3d(fl_1, fl_1, 2, 2, 0),
        )

        self.up_conv_1 = nn.Sequential(
            nn.ConvTranspose3d(fl_0, fl_0, 2, 2, 0),
        )

        self.synthesis_0 = nn.Sequential(
            nn.Conv3d(fl_0 + fl_0, fl_0, 3, 1, 1),
            nn.BatchNorm3d(fl_0),
            nn.LeakyReLU(),
            nn.Conv3d(fl_0, fl_0, 3, 1, 1),
            nn.BatchNorm3d(fl_0),
            nn.LeakyReLU(),
        )

        self.output_0 = nn.Sequential(
            nn.Conv3d(fl_0, fl_i0, 3, 1, 1),
            nn.BatchNorm3d(fl_i0),
            nn.LeakyReLU(),
            nn.Conv3d(fl_i0, 1, 3, 1, 1),
            nn.Sigmoid()
        )

        self.up_sample_3 = nn.Sequential(
            nn.ConvTranspose3d(fl_2, fl_0, 8, 8, 0),
        )

        self.up_sample_2 = nn.Sequential(
            nn.ConvTranspose3d(fl_1, fl_i0, 4, 4, 0),
        )

        self.up_sample_1 = nn.Sequential(
            nn.ConvTranspose3d(fl_0, fl_i1, 2, 2, 0),
        )

        self.output_1 = nn.Sequential(
            nn.Conv3d(fl_0 + fl_0 + fl_i0 + fl_i1, fl_0, 3, 1, 1),
            nn.BatchNorm3d(fl_0),
            nn.LeakyReLU(),
            nn.Conv3d(fl_0, fl_0, 3, 1, 1),
            nn.BatchNorm3d(fl_0),
            nn.LeakyReLU(),
            nn.Conv3d(fl_0, fl_i0, 3, 1, 1),
            nn.BatchNorm3d(fl_i0),
            nn.LeakyReLU(),
            nn.Conv3d(fl_i0, 1, 3, 1, 1),
            nn.Sigmoid()
        )
    
