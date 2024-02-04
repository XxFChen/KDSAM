import torch
from edge_sam.build_sam import build_sam_vit_h

# 指定预训练权重的路径
checkpoint_path = '/root/autodl-tmp/sam_vit_h_4b8939.pth'

# 创建模型实例并加载预训练权重
model = build_sam_vit_h(checkpoint=checkpoint_path)

# 初始化一个空的权重字典
image_encoder_state_dict = {}

# 遍历完整模型权重字典，提取 Image Encoder 部分的权重
for key, value in model.state_dict().items():
    if 'image_encoder.' in key:
        # 移除键名中的 'image_encoder.' 前缀
        new_key = key.replace('image_encoder.', '')
        image_encoder_state_dict[new_key] = value

# 保存 Image Encoder 部分的权重为新的 checkpoint 文件
torch.save(image_encoder_state_dict, '/root/autodl-tmp/image_encoder_weights.pth')