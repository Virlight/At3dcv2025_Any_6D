import json
import numpy as np
from PIL import Image
import pycocotools.mask as mask_util

# 路径配置
image_base = "house.png"
image_name = image_base.split(".")[0]
img_path = f"data/{image_base}"
json_path = "Grounded-SAM-2/outputs/grounded_sam2_hf_demo/grounded_sam2_hf_model_demo_results.json"
output_path = f"data/{image_name}_mask_centered.png"

# 加载原始图像
image = np.array(Image.open(img_path).convert("RGB"))

# 加载 JSON 文件
with open(json_path, "r") as f:
    result = json.load(f)

# 提取 mask 信息
target_index = 0  # 你想提取第几个 mask，就改这里
rle = result["annotations"][target_index]["segmentation"]
mask = mask_util.decode(rle)  # [H, W], dtype: uint8 (0 or 1)

# 找出前景 bounding box
ys, xs = np.where(mask > 0)
ymin, ymax = ys.min(), ys.max()
xmin, xmax = xs.min(), xs.max()

# 裁剪出前景（带mask区域）
obj_img = image[ymin:ymax+1, xmin:xmax+1]
obj_mask = mask[ymin:ymax+1, xmin:xmax+1]

# 创建白色背景
H, W = image.shape[:2]
centered_img = np.ones_like(image) * 255

# 计算新位置（居中）
obj_h, obj_w = obj_img.shape[:2]
center_y = H // 2
center_x = W // 2
y1 = center_y - obj_h // 2
x1 = center_x - obj_w // 2

# 粘贴前景到中央，只在mask区域覆盖
mask_indices = np.where(obj_mask > 0)
centered_img[y1:y1+obj_h, x1:x1+obj_w][mask_indices] = obj_img[mask_indices]

# 保存
Image.fromarray(centered_img).save(output_path)
