# 文件路径: /root/autodl-tmp/KDSAM/edge_sam/extract_encoder_weights.py

import torch
from build_sam import build_sam_vit_h

def extract_image_encoder_weights(checkpoint_path, save_path):
    # 加载完整的SAM模型
    sam_h_model = build_sam_vit_h(checkpoint=checkpoint_path)

    # 访问并提取Image Encoder的权重
    image_encoder_state_dict = sam_h_model.image_encoder.state_dict()

    # 保存Image Encoder的状态字典
    torch.save(image_encoder_state_dict, save_path)
    print(f"Image Encoder weights saved to {save_path}")

if __name__ == "__main__":
    checkpoint_path = '/root/autodl-tmp/sam_vit_h_4b8939.pth'
    save_path = '/root/autodl-tmp/sam_vit_h_4b8939_image_encoder.pth'
    extract_image_encoder_weights(checkpoint_path, save_path)
