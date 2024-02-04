import tarfile
import os

def extract_all_tarfiles(tar_dir, extract_path):
    """
    解压tar_dir目录下的所有.tar文件到extract_path目录。
    Args:
    - tar_dir: 包含.tar文件的目录路径。
    - extract_path: 解压目标路径。
    """
    # 检查解压目标路径是否存在，如果不存在就创建它
    if not os.path.exists(extract_path):
        os.makedirs(extract_path)
    
    # 遍历tar_dir目录中的所有文件
    for tar_file in os.listdir(tar_dir):
        # 只处理.tar文件
        if tar_file.endswith('.tar'):
            tar_path = os.path.join(tar_dir, tar_file)
            # 打开.tar文件
            with tarfile.open(tar_path, 'r:*') as tar:
                # 解压全部内容
                tar.extractall(path=extract_path)
            print(f"Extracted tar file {tar_path} to directory {extract_path}")

# 调用函数以解压所有.tar文件
tar_files_dir = '/root/autodl-tmp'  # .tar文件的目录路径
destination_path = '/root/autodl-tmp/SA-1B_dataset'  # 解压目标路径
extract_all_tarfiles(tar_files_dir, destination_path)
