import os
import numpy as np
import argparse
import random
import cv2
import json

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter  

from dataset import transform, sa1b_dataset
from edge_sam.modeling.rep_vit import RepViT as StudentModel
from edge_sam.build_sam import build_sam_vit_h 

# from Distiller.ofa import OFA

# from torch import distributed as dist
# from torch.utils.data.distributed import DistributedSampler



def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    # dataset paths
    parser.add_argument('--dataset_path', type=str, default="/root/autodl-tmp/SA-1B", help='root path of dataset')

    # training epochs, batch size and so on
    parser.add_argument('--epochs', type=int, default=8, help='number of training epochs')
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers to use')
    parser.add_argument('--batch_size', type=int, default=8, help='batch_size')

    # multi gpu settings
    parser.add_argument("--local_rank", type=int, default=-1)

    # cuda settings
    # parser.add_argument('--device', type=str, default='cuda', help='device')
    parser.add_argument('--seed', type=int, default=1234, help='seed')
    parser.add_argument('--deterministic', type=bool, default=True, help='deterministic')
    parser.add_argument('--benchmark', type=bool, default=False, help='benchmark')

    # learning process settings
    parser.add_argument('--optim', type=str, default='sgd', choices=['adam', 'sgd', 'adamw'])
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # print and evaluate frequency during training
    parser.add_argument('--print_iters', type=int, default=200, help='print loss iterations')
    parser.add_argument('--eval_nums', type=int, default=200, help='evaluation numbers')
    parser.add_argument('--eval_iters', type=int, default=500, help='evaluation iterations')

    # file and folder paths
    parser.add_argument('--root_path', type=str, default="/root/autodl-tmp/KDSAM", help='root path')
    parser.add_argument('--work_dir', type=str, default="work_dir", help='work directory')
    parser.add_argument('--save_dir', type=str, default="ckpt", help='save directory')
    parser.add_argument('--log_dir', type=str, default="log", help='save directory')
    parser.add_argument('--save_iters', type=int, default=30000, help='save iterations')

    # # 添加OFA特定的参数
    # parser.add_argument('--gt-loss-weight', default=1., type=float, help='Ground truth loss weight')
    # parser.add_argument('--kd-loss-weight', default=1., type=float, help='Knowledge Distillation loss weight')
    # parser.add_argument('--ofa-eps', default=[1], nargs='+', type=float, help='OFA epsilon values for each stage')
    # parser.add_argument('--ofa-stage', default=[1, 2, 3, 4], nargs='+', type=int, help='OFA stages to apply distillation')
    # parser.add_argument('--ofa-loss-weight', default=1, type=float, help='OFA loss weight')
    # parser.add_argument('--ofa-temperature', default=1, type=float, help='Temperature for OFA loss calculation')

    args = parser.parse_args()
    return args



def get_optimizer(args, model):
    if args.optim == 'adam':
        return optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optim == 'sgd':
        return optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optim == 'adamw':
        return optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        raise NotImplementedError(args.optim)

def get_scheduler(args, optimizer):
    return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.5)
    
def customized_mseloss(pred_feats, target_feats):
    # return (0.5 * (pred_feats - target_feats) ** 2).sum(1).mean()
    return ((pred_feats - target_feats) ** 2).sum(1).mean().sqrt()
            
def test(args, model, test_loader):
    model.eval()
    test_loss = 0
    arch = 'm3'  # 或 'm2' 或 'm3'，取决于你想使用哪种配置
    student_model = StudentModel(arch=arch)
    student_model.to(device)

    with torch.no_grad():
        for idx, (imgs, target_feats, mask_paths) in enumerate(test_loader):
            imgs, target_feats = imgs.to(device), target_feats.to(device)
            pred_feats = student_model(imgs)
            test_loss += customized_mseloss(pred_feats, target_feats).item()

    return test_loss / len(test_loader)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




def main(args):


    

    # file folder creating
    if args.local_rank == 0:
        if not os.path.exists(os.path.join(args.root_path, args.work_dir, args.save_dir)):
            os.makedirs(os.path.join(args.root_path, args.work_dir, args.save_dir))
        
        if not os.path.exists(os.path.join(args.root_path, args.work_dir, args.log_dir)):
            os.makedirs(os.path.join(args.root_path, args.work_dir, args.log_dir))

    # seed setting
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cudnn.deterministic = args.deterministic
        cudnn.benchmark = args.benchmark
    
    # dataset
    train_dirs = ["sa_" + str(i).zfill(6) for i in range(20, 29)]
    val_dirs = ['sa_000137']
    train_dataset = sa1b_dataset(args.dataset_path, train_dirs, transform)
    val_dataset = sa1b_dataset(args.dataset_path, val_dirs, transform, args.eval_nums)

    # data loader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    writer = SummaryWriter(os.path.join(args.root_path, args.work_dir, args.log_dir))

    # model
    teacher_checkpoint = '/root/autodl-tmp/sam_vit_h_4b8939.pth'  # 需要被替换为实际的路径
    
    # Full_model = build_sam_vit_h()
    # Full_model.load_state_dict(trch.load(teacher_checkpoint))
    # teacher_model = Full_model.image_encoder
    # teacher_model.to(device)
    # teacher_model.eval()

    arch = 'm3'  # 或 'm2' 或 'm3'，取决于你想使用哪种配置
    student_model = StudentModel(arch=arch)
    student_model.to(device)


    criterion = nn.CrossEntropyLoss()  # 对于分类任务，使用交叉熵损失


    
    # optimizer and scheduler
    optimizer = get_optimizer(args, student_model)
    scheduler = get_scheduler(args, optimizer)

    total_iters = 0

    for epoch in range(1, args.epochs + 1):
        # training
        student_model.train()
        for batch_idx, (imgs, target_feats, mask_paths) in enumerate(train_loader):
            total_iters += 1
            
            imgs, target_feats = imgs.to(device), target_feats.to(device)
            optimizer.zero_grad()
            pred_feats = student_model(imgs)
            

             # change
            # global_avg_pool = torch.nn.AdaptiveAvgPool2d(1)
            
            # t_attention_map = global_avg_pool(target_feats)
            # t_attention_map = t_attention_map.view(target_feats.size(0), target_feats.size(1))
            # t_attention_map_expanded = t_attention_map.unsqueeze(-1).unsqueeze(-1).expand_as(target_feats)
            # alpha = student_model.weights(student_model.alpha)
            # t_enhanced_feature_map =  target_feats   + alpha * t_attention_map_expanded
            

          

            # s_attention_map = global_avg_pool(pred_feats)
            # s_attention_map = s_attention_map.view(pred_feats.size(0), pred_feats.size(1))
            # s_attention_map_expanded = s_attention_map.unsqueeze(-1).unsqueeze(-1).expand_as(pred_feats)
            # beta = student_model.weights(student_model.beta)
            # s_enhanced_feature_map =  pred_feats   +  beta * s_attention_map_expanded

            
            # change
            
            
            
            
            # loss = customized_mseloss(s_enhanced_feature_map, t_enhanced_feature_map)
            loss = customized_mseloss(pred_feats, target_feats)
            loss.backward()
            optimizer.step()
            

            
            
            # if is master process
                # print training info
            if (batch_idx + 1) % args.print_iters == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tMSE Loss: {:.6f}'.format(
                    epoch, batch_idx * len(imgs), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.item()))
                writer.add_scalar("mse_loss", loss.item(), total_iters)
                
                # save model
            if total_iters % args.save_iters == 0:
                save_path = os.path.join(args.root_path, args.work_dir, args.save_dir, "iter_" + str(total_iters) + ".pth")
                print("save model to {}".format(save_path))
                torch.save(student_model.state_dict(), save_path)

                # evaluation
            
            # if total_iters % args.eval_iters == 0:
            #     test_loss = test(args, student_model, val_loader)
            #     print('\nTest set: Average loss: {:.4f}\n'.format(test_loss))
            #     writer.add_scalar("eval_mse_loss", test_loss, total_iters)
            

        scheduler.step()
    # save final model
    if args.local_rank == 0:
        torch.save(student_model.state_dict(), os.path.join(args.root_path, args.work_dir, args.save_dir, "iter_final.pth"))
        writer.close()

if __name__ == "__main__":
    args = parse_option()
    main(args)