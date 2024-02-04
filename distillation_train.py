import torch
from Distiller.DKD import DKD,dkd_loss
from data_preprocess import CustomDataset
from edge_sam.build_sam import build_sam_vit_h as TeacherModel
from edge_sam.modeling.rep_vit import RepViT as StudentModel
from torch.optim import Adam
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from statistics import mean

arch = 'm1'
img_size = 1024
#加载权重
checkpoint_path = '/root/autodl-tmp/sam_vit_h_4b8939.pth'

# 初始化模型
teacher_model = TeacherModel(checkpoint=checkpoint_path)
student_model = StudentModel(arch, img_size)

# 加载教师模型预训练权重
teacher_model.load_state_dict(torch.load('/root/autodl-tmp/image_encoder_weights.pth'), strict=False)

teacher_model.eval()
student_model.train()

# 配置蒸馏的超参数
cfg = {
    'DKD': {
        'CE_WEIGHT': 0.5,  # 交叉熵损失的权重
        'ALPHA': 0.9,      # DKD损失的权重
        'BETA': 0.1,       # DKD损失的第二部分的权重
        'T': 2.0,          # 温度参数
        'WARMUP': 5        # Warmup周期数
    }
}

# 实例化DKD类
distiller = DKD(student_model, teacher_model, cfg)

# 设置优化器
optimizer = torch.optim.Adam(student_model.parameters(), lr=0.001)


num_epochs = 100
device = "cuda" if torch.cuda.is_available() else "cpu"
teacher_model.to(device).eval()  # 先移动到设备，再设置为评估模式
student_model.to(device).train()  # 先移动到设备，再设置为训练模式

# 图像转换
transform = transforms.Compose([
    transforms.Resize((200, 200)),  # 根据ImageEncoderViT类的要求调整
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 实例化数据集
root_dir = '/root/autodl-tmp/SA-1B_dataset'  # 更改为实际路径
dataset = CustomDataset(root_dir=root_dir, transform=transform)

# # 创建数据加载器
data_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)

# # 测试数据集是否可用
# print(f"Dataset size: {len(dataset)}")

# # 尝试获取第一个样本
# if len(dataset) > 0:
#     sample = dataset[10]
#     print("First sample loaded successfully")
# else:
#     print("Dataset is empty. Check your dataset path and loading logic.")


# 训练循环
for epoch in range(num_epochs):
    epoch_losses = []
    for batch in enumerate(data_loader):
        # 从batch获取输入和目标
        inputs = batch["inputs"].to(device)
        targets = batch["targets"].to(device)

        # 前向传播 - 学生模型
        logits_student = student_model(inputs)

        # 不计算教师模型的梯度
        with torch.no_grad():
            # 前向传播 - 教师模型
            logits_teacher = teacher_model(inputs)

        # 计算DKD损失
        loss_dkd = dkd_loss(
            logits_student, logits_teacher, targets,
            alpha=cfg['DKD']['ALPHA'], beta=cfg['DKD']['BETA'], temperature=cfg['DKD']['T']
        )

        # 反向传播和优化
        optimizer.zero_grad()
        loss_dkd.backward()
        optimizer.step()
        
        # 记录损失
        epoch_losses.append(loss_dkd.item())

    # 输出日志
    print(f'EPOCH: {epoch}')
    print(f'Mean Loss: {mean(epoch_losses)}')
