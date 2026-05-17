import math
import torch
from torch import nn
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


class TPR(nn.Module):
    def __init__(self, pretrained):
        super().__init__()
        self.conv_l = nn.Conv2d(1, 64, kernel_size=1)
        self.layer1_l = pretrained.layer1
        self.layer2_l = pretrained.layer2
        self.layer3_l = pretrained.layer3
        self.layer4_l = pretrained.layer4
        self.v_conv_l = VertConv(in_channels=512, mid_channels=256, out_channels=256)
        self.netvlad_l = NetVLADLoupe(feature_size=256, max_samples=132, cluster_size=32, output_dim=256)
        self.eca = eca_layer(256)
        self.attention = MultiHeadAttention(d_model=256, n_head=4)  # 保留注意力机制
        self.final_discriminator = Naive_Discriminator(256, 1)

    def forward(self, x_l):
        # x_l: B x C x Hl x Wl
        x_l = self.conv_l(x_l)
        x_l = self.layer1_l(x_l)
        x_l = self.layer2_l(x_l)
        x_l = self.layer3_l(x_l)
        x_l = self.layer4_l(x_l)

        x_l = self.v_conv_l(x_l)  # B x C x Wl
        x_l = x_l.unsqueeze(2)  # B x C x 1 x Wl

        # 应用注意力机制
        x_l = x_l.squeeze(2)  # B x C x Wl -> B x C x Wl (去掉第3维)
        x_l = x_l.permute(0, 2, 1)  # B x Wl x C
        x_l = x_l + self.attention(x_l, x_l, x_l)  # 应用自注意力
        x_l = x_l.permute(0, 2, 1)  # B x C x Wl

        x_l_n = self.netvlad_l(x_l.unsqueeze(2))  # B x 256
        x_l_n = x_l_n.unsqueeze(-1).unsqueeze(-1)
        x_l_n = self.eca(x_l_n)
        x_l_n = x_l_n.squeeze(-1).squeeze(-1)
        descriptors = nn.functional.normalize(x_l_n, dim=-1)

        x_final_reverse = gradient_scalar(x_l, 1.0)
        x_final_reverse = x_final_reverse.permute(0, 2, 1)  # [B, Wl, C]
        x_final_reverse = x_final_reverse.unsqueeze(2)  # [B, Wl, 1, C]
        x_final_reverse = x_final_reverse.permute(0, 3, 2, 1)  # [B, C, 1, Wl]
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
