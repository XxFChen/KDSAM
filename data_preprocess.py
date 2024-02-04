import os
import json
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        # 获取文件夹内的图像和注释文件
        self.image_annotations_pairs = []
        for filename in os.listdir(root_dir):
            if filename.endswith('.jpg'):
                img_path = os.path.join(root_dir, filename)
                ann_path = os.path.join(root_dir, filename.replace('.jpg', '.json'))
                if os.path.isfile(ann_path):  # 确保注释文件存在
                    self.image_annotations_pairs.append((img_path, ann_path))

    def __len__(self):
        return len(self.image_annotations_pairs)

    def __getitem__(self, idx):
        img_path, ann_path = self.image_annotations_pairs[idx]
        image = Image.open(img_path).convert('RGB')
        with open(ann_path, 'r') as f:
            annotation = json.load(f)

        if self.transform:
            image = self.transform(image)

        return image, annotation

# # 图像转换
# transform = transforms.Compose([
#     transforms.Resize((1024, 1024)),  # 根据ImageEncoderViT类的要求调整
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])

# # 实例化数据集
# root_dir = '/root/autodl-tmp/SA-1B_dataset'  # 更改为实际路径
# dataset = CustomDataset(root_dir=root_dir, transform=transform)

# # # 创建数据加载器
# data_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

# # 测试数据集是否可用
# print(f"Dataset size: {len(dataset)}")

# # 尝试获取第一个样本
# if len(dataset) > 0:
#     sample = dataset[0]
#     print("First sample loaded successfully")
# else:
#     print("Dataset is empty. Check your dataset path and loading logic.")

