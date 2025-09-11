import json
import cv2
import numpy as np
from PIL import Image
import pycocotools.mask as mask_util

# 路径配置
image_base = "sofa.jpg"
image_name = image_base.split(".")[0]
img_path = f"data/{image_base}"
json_path = "Grounded-SAM-2/outputs/grounded_sam2_hf_demo/grounded_sam2_hf_model_demo_results.json"
output_path = f"data/{image_name}_mask.png"

# 加载原始图像
image = np.array(Image.open(img_path).convert("RGB"))

# 加载 JSON 文件
with open(json_path, "r") as f:
    result = json.load(f)

# 提取 mask 信息（最大概率一般会是第一个物体，编号0）
# 也可以修改target_index去提取编号为 1 的 annotation
target_index = 0  # ← 你想提取第几个 mask，就改这里
rle = result["annotations"][target_index]["segmentation"]
mask = mask_util.decode(rle)  # shape: [H, W], dtype: uint8 (0 or 1)

# 将 mask 应用于图像 (有多种方案)
# 1. 构造 RGBA 图像，背景透明
# foreground = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)  # RGBA
# foreground[..., :3] = image  # RGB
# foreground[..., 3] = (mask * 255).astype(np.uint8)  # A通道为mask值（前景为255）

# 2. 创建一个新的白底图
white_background = np.ones_like(image) * 255  # 全白背景
foreground = np.where(mask[..., None], image, white_background)  # 保留前景，背景为白

# 保存为 PNG（保留透明背景）
Image.fromarray(foreground).save(output_path)
