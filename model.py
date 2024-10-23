import torch
import torch.nn as nn
import numpy
from timm.models.layers import DropPath

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Stem(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_dim, out_dim // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim // 2),
            nn.GELU(),
            nn.Conv2d(out_dim // 2, out_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.GELU(),
        )

    def forward(self, x):
        return self.stem(x)


class FFN(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Conv2d(in_dim, in_dim * 4, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_dim * 4),
            nn.GELU(),
            nn.Conv2d(in_dim * 4, in_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_dim),
        )

    def forward(self, x):
        return self.ffn(x)


class InvertedResidual(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.DeepWiseSeparate = nn.Sequential(
            nn.Conv2d(in_dim, in_dim * 4, kernel_size=1),
            nn.BatchNorm2d(in_dim * 4),
            nn.GELU(),

            nn.Conv2d(in_dim * 4, in_dim * 4, kernel_size=3, stride=1, padding=1, groups=in_dim * 4),
            nn.BatchNorm2d(in_dim * 4),
            nn.GELU(),

            nn.Conv2d(in_dim * 4, in_dim, kernel_size=1),
            nn.BatchNorm2d(in_dim),
        )

    def forward(self, x):
        return x + self.DeepWiseSeparate(x)


class DownSample(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.ds = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim)
        )

    def forward(self, x):
        return self.ds(x)


class DyMRconv(nn.Module):
    def __init__(self, K, in_dim, out_dim):
        super().__init__()
        self.num_ones = None
        self.mean = 0
        self.std = 0
        self.K = K
        self.nn = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 1),
            nn.BatchNorm2d(out_dim),
            nn.GELU()
        )

    def forward(self, x):

        B, C, H, W = x.shape
        x_j = x - x
        x_m = x - x
        x_rolled = torch.cat([x[:, :, -H // 2:, :], x[:, :, :-H // 2, :]], dim=2)
        x_rolled = torch.cat([x_rolled[:, :, :, -W // 2:], x_rolled[:, :, :, :-W // 2]], dim=3)

        norm = torch.norm(x - x_rolled, p=1, dim=1, keepdim=True)
        self.mean = torch.mean(norm, dim=[2, 3], keepdim=True)
        self.std = torch.std(norm, dim=[2, 3], keepdim=True)
        mask_sum = norm - norm
        for i in range(0, H, self.K):
            x_rolled = torch.cat([x[:, :, -i:, :], x[:, :, :-i, :]], dim=2)
            dist = torch.norm(x - x_rolled, p=1, dim=1, keepdim=True)
            mask = torch.where(dist < self.mean - self.std, 1, 0)
            x_d = (x_rolled - x) * mask
            #x_m = x_m + x_rolled * mask
            x_j = torch.max(x_j, x_d)
            mask_sum = mask_sum + mask

        for j in range(0, W, self.K):
            x_rolled = torch.cat([x[:, :, -j:, :], x[:, :, :-j, :]], dim=2)
            dist = torch.norm(x - x_rolled, p=1, dim=1, keepdim=True)
            mask = torch.where(dist < self.mean - self.std, 1, 0)
            x_d = (x_rolled - x) * mask
            #x_m = x_m + x_rolled * mask
            x_j = torch.max(x_j, x_d)
            mask_sum = mask_sum + mask

        mask_min = mask_sum.amin(dim=(1, 2, 3), keepdim=True)  # 找到每个batch的最小值
        mask_max = mask_sum.amax(dim=(1, 2, 3), keepdim=True)
        #mask_weight = (mask_sum - mask_min) / ((mask_max - mask_min)+1e-10)
        #mask_sum = mask_sum+1e-10
        #x_m = x_m/mask_sum
        x_weighted = torch.where(mask_sum < (mask_max - mask_min)*0.4, 1, 0)
        x = torch.cat([x, x_j, x*x_weighted], dim=1)
        #x = torch.cat([x, x_j, x_m], dim=1)  # changed
        return self.nn(x)


class PEG(nn.Module):
    def __init__(self, in_dim, kernel_size=3):
        super().__init__()
        self.peg = nn.Conv2d(in_dim, in_dim, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, groups=in_dim)

    def forward(self, x):
        return x + self.peg(x)


class DyGrapher(nn.Module):
    def __init__(self, in_dim, K):
        super().__init__()
        self.cpe = PEG(in_dim)
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_dim)
        )
        self.grapher = DyMRconv(K, in_dim * 3, in_dim)
        self.fc2 = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_dim)
        )

    def forward(self, x):
        x = self.cpe(x)
        x = self.fc1(x)
        x = self.grapher(x)
        x = self.fc2(x)
        return x


class DAGC(nn.Module):
    def __init__(self, in_dim, K, drop_path=0.):
        super().__init__()
        self.Dygrapher = DyGrapher(in_dim, K)
        self.ffn = FFN(in_dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.Dygrapher(x))
        x = x + self.drop_path(self.ffn(x))
        return x


class GreedyViG(torch.nn.Module):
    def __init__(self, blocks, channels,
                 dropout=0., drop_path=0., emb_dims=512,
                 K=2, num_classes=500):
        super(GreedyViG, self).__init__()

        self.stage_names = ['stem', 'local_1', 'local_2', 'local_3', 'global']

        n_blocks = sum([sum(x) for x in blocks])
        dpr = [x.item() for x in torch.linspace(0, drop_path, n_blocks)]  # stochastic depth decay rule
        dpr_idx = 0

        self.stem = Stem(in_dim=3, out_dim=channels[0])

        self.backbone = []
        for i in range(len(blocks)):
            stage = []
            local_stages = blocks[i][0]
            global_stages = blocks[i][1]
            if i > 0:
                stage.append(DownSample(channels[i - 1], channels[i]))
            for _ in range(local_stages):
                stage.append(InvertedResidual(in_dim=channels[i]))
            for _ in range(global_stages):
                stage.append(DAGC(channels[i], K=K[i]))
                dpr_idx += 1
            self.backbone.append(nn.Sequential(*stage))

        self.backbone = nn.Sequential(*self.backbone)

        self.prediction = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                        nn.Conv2d(channels[-1], emb_dims, kernel_size=1, bias=True),
                                        nn.BatchNorm2d(emb_dims),
                                        nn.GELU(),
                                        nn.Dropout(dropout))

        self.head = nn.Conv2d(emb_dims, num_classes, kernel_size=1, bias=True)

        self.model_init()

    def model_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, inputs):
        x = self.stem(inputs)
        B, C, H, W = x.shape
        x = self.backbone(x)

        x = self.prediction(x)

        x = self.head(x).squeeze(-1).squeeze(-1)
        return x

# dim = x.shape[1]
# a = InvertedResidual(dim)


# b = torch.randn(32, 3, 224, 224)
#
# a = DyMRconv(2,6,64)
# c = a(b)
# print(c.shape)
# if torch.cuda.is_available():
#     print("CUDA is available. Current device:", torch.cuda.get_device_name(torch.cuda.current_device()))
# else:
#     print("CUDA is not available.")
