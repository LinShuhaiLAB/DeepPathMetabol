import torchvision.models as models
from torchvision import transforms
import numpy as np
import torch
import torch.nn as nn

def extract_patches_from_adata(adata,spot_diameter=None):
    """
    从AnnData对象中提取每个spot的高分辨率图像补丁。

    参数:
    adata -- 包含空间信息和图像数据的AnnData对象。

    返回:
    patches -- 包含所有spot补丁的列表。
    """
    # 从 adata 中提取高分辨率图像和spot直径的一半
    id = list(adata.uns['spatial'].keys())[0]
    hires_image = adata.uns['spatial'][id]['images']['hires']
    if spot_diameter is not None:
        half_diameter = spot_diameter
    else:
        spot_diameter = adata.uns['spatial'][id]['scalefactors']['spot_diameter_fullres']
        half_diameter = spot_diameter // 1

    spot_x = adata.obsm['spatial'][:, 1]  # 假设 x 坐标在第二列
    spot_y = adata.obsm['spatial'][:, 0]  # 假设 y 坐标在第一列

    patches = []

    # 提取每个 spot 的补丁
    for x, y in zip(spot_x, spot_y):
        # 计算补丁的边界
        x_start = int(x - half_diameter)
        x_end = int(x + half_diameter)
        y_start = int(y - half_diameter)
        y_end = int(y + half_diameter)

        # # 确保边界在图像范围内
        # x_start = max(0, x_start)
        # x_end = min(hires_image.shape[0], x_end)
        # y_start = max(0, y_start)
        # y_end = min(hires_image.shape[1], y_end)

        # 提取补丁并添加到列表
        patch = hires_image[x_start:x_end, y_start:y_end]
        patches.append(patch)

    return patches


def process_patches_with_model(patches):
    """
    使用 torchvision.models 中的预训练模型处理 patches，并进行转换和归一化。

    参数:
    patches -- 包含所有spot补丁的列表。

    返回:
    processed_patches -- 模型处理后的patches列表。
    """
    # 确保 patches 是 uint8 类型
    patches = [np.array(patch, dtype=np.uint8) if isinstance(patch, np.ndarray) else patch for patch in patches]


    # 定义转换和归一化
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128, 128)),  # 调整大小到512x512
        transforms.Lambda(lambda x: np.array(x)[..., :3] if x.mode == 'RGBA' else np.array(x)),  # 去除透明度通道
        transforms.ToTensor(),  # 转换为Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
    ])

    # 将 patches 转换为张量并应用转换和归一化
    patches_tensor = torch.stack([transform(patch).float() for patch in patches])  # 确保为 Float 类型

    # 选择一个预训练模型，例如 resnet18
    model = models.resnet18(weights='IMAGENET1K_V1')  # 使用新的 weights 参数

    # # 修改模型的最后一层，将其输出改为6
    # num_features = model.fc.in_features
    # model.fc = nn.Linear(num_features, 6)

    # 确保模型处于评估模式
    model.eval()

    # 使用模型处理 patches
    with torch.no_grad():
        processed_patches = model(patches_tensor)

    return processed_patches