import math
import torch
from torch import nn
import numpy as np
from modules.netvlad import NetVLADLoupe
from torchvision.models.resnet import resnet18


class Naive_Discriminator(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=1, padding=0)
        )

    def forward(self, x):
        x = self.conv(x)
        x = torch.mean(x, (1, 2, 3))
        return x


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd, **kwargs: None):
        ctx.lambd = lambd
        return x.view_as(x)

    # @staticmethod
    # def backward(ctx, *grad_output):
    #     return grad_output[0] * -ctx.lambd, None

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * -ctx.lambd, None


gradient_scalar = GradReverse.apply


class SPR(nn.Module):
    def __init__(self, pretrained):
        super().__init__()
        self.conv1_i = pretrained.conv1
        self.bn1_i = pretrained.bn1
        self.relu = pretrained.relu
        self.maxpool = pretrained.maxpool
        self.layer1_i = pretrained.layer1
        self.layer2_i = pretrained.layer2
        self.layer3_i = pretrained.layer3
        self.layer4_i = pretrained.layer4
        self.v_conv_i = VertConv(in_channels=512, mid_channels=256, out_channels=256)
        self.netvlad_i = NetVLADLoupe(feature_size=256, max_samples=132, cluster_size=32, output_dim=256)
        self.eca = eca_layer(256)
        self.attention = MultiHeadAttention(d_model=256, n_head=4)  # 保留注意力机制
        self.final_discriminator = Naive_Discriminator(256, 1)

    def forward(self, x_i):
        # x_i: B x N x C x Hi x Wi
        B, N, C, Hi, Wi = x_i.shape
        x_i = x_i.view(B * N, C, Hi, Wi)
        x_i = self.conv1_i(x_i)
        x_i = self.bn1_i(x_i)
        x_i = self.relu(x_i)
        x_i = self.maxpool(x_i)
        x_i = self.layer1_i(x_i)
        x_i = self.layer2_i(x_i)
        x_i = self.layer3_i(x_i)
        x_i = self.layer4_i(x_i)

        x_i = paronamic_concat(x_i, N=N)  # B x C x Hi x NWi
        x_i = self.v_conv_i(x_i)  # B x C x NWi
        x_i = x_i.unsqueeze(2)  # B x C x 1 x NWi

        x_i = x_i.squeeze(2)  # B x C x NWi -> B x C x NWi (去掉第3维)
        x_i = x_i.permute(0, 2, 1)  # B x NWi x C
        x_i = x_i + self.attention(x_i, x_i, x_i)  # 自注意力
        x_i = x_i.permute(0, 2, 1)  # B x C x NWi

        x_i_n = self.netvlad_i(x_i.unsqueeze(2))  # B x 256
        x_i_n = x_i_n.unsqueeze(-1).unsqueeze(-1)
        x_i_n = self.eca(x_i_n)
        x_i_n = x_i_n.squeeze(-1).squeeze(-1)
        descriptors = nn.functional.normalize(x_i_n, dim=-1)

        # x_final_reverse = gradient_scalar(x_i, 1.0)
        # domain_output = self.final_discriminator(x_final_reverse)
        x_final_reverse = gradient_scalar(x_i, 1.0)
        x_final_reverse = x_final_reverse.permute(0, 2, 1)  # [B, NWi, C]
        x_final_reverse = x_final_reverse.unsqueeze(2)  # [B, NWi, 1, C]
        x_final_reverse = x_final_reverse.permute(0, 3, 2, 1)  # [B, C, 1, NWi]
        domain_output = self.final_discriminator(x_final_reverse)

        return descriptors, domain_output

    @classmethod
    def create(cls, weights=None):
        if weights is not None:
            pretrained = resnet18(weights=weights)
        else:
            pretrained = resnet18()
        model = cls(pretrained)
        return model


class eca_layer(nn.Module):
    def __init__(self, channel, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


def paronamic_concat(x, N):
    # x: BN x C x H x W
    BN, C, H, W = x.shape
    B = int(BN / N)
    x = x.view(B, N, C, H, W)
    x = x.permute(0, 2, 3, 1, 4)  # B x C x H x N x W
    x = x.reshape(B, C, H, N * W)  # B x C x H x NW
    return x


class VertConv(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=True)
        )

        self.reduce_conv = nn.Sequential(
            nn.Conv1d(
                in_channels,
                mid_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True)
        )

        self.conv = nn.Sequential(
            nn.Conv1d(
                mid_channels,
                mid_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                mid_channels,
                mid_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
        )

        self.out_conv = nn.Sequential(
            nn.Conv1d(
                mid_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                bias=True,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.input_conv(x)
        x = x.max(2)[0]
        x = self.reduce_conv(x)
        x = self.conv(x) + x
        x = self.out_conv(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.attention = ScaleDotProductAttention()
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        # 1. dot product with weight matrices
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        # 2. split tensor by number of heads
        q, k, v = self.split(q), self.split(k), self.split(v)

        # 3. do scale dot product to compute similarity
        out, attention = self.attention(q, k, v, mask=mask)

        # 4. concat and pass to linear layer
        out = self.concat(out)
        out = self.w_concat(out)

        return out

    def split(self, tensor):
        batch_size, length, d_model = tensor.size()

        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)

        return tensor

    def concat(self, tensor):
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor

        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor


class ScaleDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None, e=1e-12):
        batch_size, head, length, d_tensor = k.size()

        k_t = k.transpose(2, 3)
        score = (q @ k_t) / math.sqrt(d_tensor)

        if mask is not None:
            score = score.masked_fill(mask == 0, -10000)

        score = self.softmax(score)

        v = score @ v

        return v, score
