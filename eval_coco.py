import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import time
import json
import torch
from edge_sam.modeling.rep_vit import RepViT
from pycocotools.coco import COCO
from pycocotools.mask import decode, frPyObjects
from dataset import transform
from train import customized_mseloss
from edge_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

def parse_option():
    parser = argparse.ArgumentParser('argument for evaluation')

    parser.add_argument('--device', type=str, default='cuda', help='device')

    # eval dataset settings
    parser.add_argument('--dataset_path', type=str, default="/root/autodl-tmp/annotations_trainval2014/annotations/instances_val2014.json", help='root path of dataset')
    parser.add_argument('--eval_num', type=int, default=20)
    parser.add_argument('--data_idx_offset', type=int, default=1532781)
    parser.add_argument('--image_dir', type=str, default="/root/autodl-tmp/val2014/val2014", help='directory containing COCO images')   
    # our mobile sam model
    parser.add_argument('--ckpt', type=str, default="/root/autodl-tmp/KDSAM/test/ckpt/iter_50000.pth")

    # the given mobile sam model
    parser.add_argument('--mobile_sam_type', type=str, default="edge_sam")
    parser.add_argument('--mobile_sam_ckpt', type=str, default="/root/autodl-tmp/sam_vit_h_4b8939.pth")

    # sam model 
    parser.add_argument('--sam_type', type=str, default="vit_h")
    parser.add_argument('--sam_ckpt', type=str, default="/root/autodl-tmp/sam_vit_h_4b8939.pth")

    # visualization
    parser.add_argument('--vis', type=bool, default=True, help='whether to visualize the segment results')
    parser.add_argument('--vis_dir', type=str, default="vis", help='root path of dataset')
    # miou
    parser.add_argument('--miou', type=bool, default=True, help='whether to output the miou')
    parser.add_argument('--point_num_h', type=int, default=5)
    parser.add_argument('--point_num_w', type=int, default=5)

    # paths
    parser.add_argument('--work_dir', type=str, default="./work_dir", help='work dir')

    args = parser.parse_args()
    return args

def eval_miou(pred_masks, target_masks):
    print(pred_masks.shape)
    assert len(pred_masks.shape) == 2 or len(pred_masks.shape) == 3
    if len(pred_masks.shape) == 2:
        return (pred_masks & target_masks).sum() / ((pred_masks | target_masks).sum() + 1e-10)
    return [(pred_mask & target_mask).sum() / ((pred_mask | target_mask).sum() + 1e-10) for pred_mask, target_mask in zip(pred_masks, target_masks)]

def get_coco_ground_truth(coco, img_id):


if __name__ == "__main__":
    args = parse_option()

    coco = COCO(args.dataset_path)
    img_ids = coco.getImgIds()[:args.eval_num]

    if args.vis and not os.path.exists(os.path.join(args.work_dir, args.vis_dir)):
        os.makedirs(os.path.join(args.work_dir, args.vis_dir))

    # sam = sam_model_registry['vit_h'](checkpoint=args.ckpt)  # Update this to your SAM model type
    # sam.to(device=args.device)
    # sam.eval()
    original_sam = sam_model_registry[args.sam_type](checkpoint=args.sam_ckpt)
    # 创建 RepVIT 实例并加载训练后的权重
    repvit = RepViT(arch="m3", img_size=1024, upsample_mode="bicubic")  # 或根据您的模型具体情况调整参数
    repvit.load_state_dict(torch.load('/root/autodl-tmp/KDSAM/test/ckpt/non_att_iter_30000.pth'))
    original_sam.image_encoder = repvit
    original_sam.to(device=args.device)
    sam = original_sam
    sam.eval()

    if args.vis:
        sam_mask_generator = SamAutomaticMaskGenerator(sam)

    if args.miou:
        sam_predictor = SamPredictor(sam)

    ious = []

    # Iterate through selected images for evaluation
    for img_id in img_ids:
        img_data = coco.loadImgs(img_id)[0]
        img_path = os.path.join(args.image_dir, img_data['file_name'])
        test_img = cv2.imread(img_path)
        test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
        
        sam_predictor.set_image(test_img)

        # Get ground truth mask from COCO annotations
        target_mask = get_coco_ground_truth(coco, img_id)
        h, w, c = test_img.shape

        point_num_h, point_num_w = args.point_num_h, args.point_num_w
        margin_h, margin_w = h // point_num_h, w // point_num_w
        start_point_pos = (margin_w // 2, margin_h // 2)
        input_label = np.array([1])  # Update this based on your use case

        for point_h in range(point_num_h):
            for point_w in range(point_num_w):
                input_point = np.array([[start_point_pos[0] + point_w * margin_w, start_point_pos[1] + point_h * margin_h]])
                pred_masks = sam_predictor.predict(
                        point_coords=input_point,
                        point_labels=input_label,
                    )

                # Here, you should adjust the evaluation based on how your predictions are structured
                for pred_mask in pred_masks:  # This line assumes 'pred_masks' is iterable; adjust if not
                    iou = eval_miou(pred_mask, target_mask)
                    ious.append(iou)

    if args.miou:
        print(f"Mean IoU: {np.mean(ious) * 100:.2f}%")


