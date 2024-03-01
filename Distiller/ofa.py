import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.vision_transformer import Block
from ._base import BaseDistiller
from .registry import register_distiller
from .utils import GAP1d, get_module_dict, init_weights, is_cnn_model, PatchMerging, SepConv, set_module_dict, \
    TokenFilter, TokenFnContext

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def ofa_loss(features_student, features_teacher, eps=1e-6,temperature = 1.):
    # 假设 features_student 和 features_teacher 都是特征图
    # 计算 MSE 损失
    print("Student features shape:", features_student.size())
    print("Teacher features shape:", features_teacher.size())
    loss = F.mse_loss(features_student, features_teacher, reduction='mean')
    return loss

class Adapter(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(Adapter, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=1)  # 1x1卷积用于通道数适配
        self.bn = nn.BatchNorm2d(output_channels)  # 可选，但有助于提高稳定性
        self.relu = nn.ReLU()  # 可选，增加非线性

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

@register_distiller
class OFA(BaseDistiller):
    requires_feat = True

    def __init__(self, student, teacher, criterion, args, **kwargs):
        super(OFA, self).__init__(student, teacher, criterion, args)

        if len(self.args.ofa_eps) == 1:
            eps = [self.args.ofa_eps[0] for _ in range(len(self.args.ofa_stage) + 1)]
            self.args.ofa_eps = eps

        assert len(self.args.ofa_stage) + 1 == len(self.args.ofa_eps)  # +1 for logits

        self.projector = nn.ModuleDict()

        is_cnn_student = is_cnn_model(student)

        _, feature_dim_t = self.teacher.stage_info(-1)
        _, feature_dim_s = self.student.stage_info(-1)
        #change
        self.adapters = nn.ModuleDict()
        for stage, num_channels in enumerate([64, 128, 256, 512], start=1):  # 学生模型的各阶段通道数
            self.adapters[str(stage)] = Adapter(num_channels, 1280).to(device)  # 教师模型的通道数是1280   
        #change
        for stage in self.args.ofa_stage:
            _, size_s = self.student.stage_info(stage)

            if is_cnn_student:
                in_chans, _, _ = size_s

                if stage != 4:
                    down_sample_blk_num = 4 - stage
                    down_sample_blks = []
                    for i in range(down_sample_blk_num):
                        if i == down_sample_blk_num - 1:
                            if isinstance(feature_dim_s, tuple):
                                feature_dim_s = feature_dim_s[0]  # 假设通道数是元组中的第一个元素
                                out_chans = max(feature_dim_s, feature_dim_t)
                        else:
                            out_chans = in_chans * 2
                        down_sample_blks.append(SepConv(in_chans, out_chans))
                        in_chans *= 2
                else:
                    down_sample_blks = [nn.Conv2d(in_chans, max(feature_dim_s, feature_dim_t), 1, 1, 0)]

                projector = nn.Sequential(
                    *down_sample_blks,
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.Linear(max(feature_dim_s, feature_dim_t), 256)  # new change
                )
            else:
                patch_num, embed_dim = size_s
                token_num = getattr(student, 'num_tokens', 0)  # cls tokens

                final_patch_grid = 7  # finally there are 49 patches
                patch_grid = int(patch_num ** .5)
                merge_num = max(int(np.log2(patch_grid / final_patch_grid)), 0)
                merger_modules = []
                for i in range(merge_num):
                    if i == 0:  # proj to feature_dim_s
                        merger_modules.append(
                            PatchMerging(input_resolution=(patch_grid // 2 ** i, patch_grid // 2 ** i),
                                         dim=embed_dim,
                                         out_dim=feature_dim_s,
                                         act_layer=nn.GELU))
                    else:
                        merger_modules.append(
                            PatchMerging(input_resolution=(patch_grid // 2 ** i, patch_grid // 2 ** i),
                                         dim=feature_dim_s,
                                         out_dim=feature_dim_s,
                                         act_layer=nn.GELU if i != merge_num - 1 else nn.Identity))
                patch_merger = nn.Sequential(*merger_modules)
                blocks = nn.Sequential(
                    *[Block(dim=feature_dim_s, num_heads=4) for _ in range(max(4 - stage, 1))]  # todo: check this
                )
                if token_num != 0:
                    get_feature = nn.Sequential(
                        TokenFilter(token_num, remove_mode=False),  # todo: token_num > 1
                        nn.Flatten()
                    )
                else:
                    get_feature = GAP1d()
                projector = nn.Sequential(
                    TokenFnContext(token_num, patch_merger),
                    blocks,
                    get_feature,
                    nn.Linear(feature_dim_s, 256)  # todo: cifar100
                )
            set_module_dict(self.projector, stage, projector)
        self.projector.apply(init_weights)
        # print(self.projector)  # for debug

    def forward(self, images, target_feats, **kwargs):
        # 不再需要 label
        with torch.no_grad():
            self.teacher.eval()
            _, feats_teacher = self.teacher(images, requires_feat=True)

        _, feats_student = self.student(images, requires_feat=True)

        # 现在我们比较的是特征而不是 logits
        total_loss = 0
        for stage in self.args.ofa_stage:
            idx_t, _ = self.teacher.stage_info(stage)
            idx_s, _ = self.student.stage_info(stage)
            feat_t = feats_teacher[idx_t]
            feat_s = feats_student[idx_s]
            # 检查特征图的空间维度是否需要调整以匹配
            if feat_s.size(2) != feat_t.size(2) or feat_s.size(3) != feat_t.size(3):
                # 使用双线性插值进行空间尺寸调整
                feat_s = F.interpolate(feat_s, size=(feat_t.size(2), feat_t.size(3)), mode='bilinear', align_corners=False)
            
            # 检查并适配特征图的通道数
            adapter = self.adapters[str(stage)]
            feat_s = adapter(feat_s)  # 使用适配器层调整学生特征图的通道数


            # 计算 OFA 损失
            total_loss += ofa_loss(feat_s, feat_t)

        return total_loss