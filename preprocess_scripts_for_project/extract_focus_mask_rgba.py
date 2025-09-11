import argparse
import json
import numpy as np
from PIL import Image
import pycocotools.mask as mask_util
import re
import os

def sanitize_filename(s):
    # 简单把非字母数字下划线替换成下划线，避免文件名非法字符
    return re.sub(r'[^a-zA-Z0-9_-]', '_', s)

parser = argparse.ArgumentParser()
parser.add_argument('--text-prompt', type=str, required=True, help="Text prompt used as base for filenames")
parser.add_argument('--image-path', type=str, required=True, help="Input image file path")
parser.add_argument('--depth', action='store_true', help="Whether to process depth images")
args = parser.parse_args()

depth = args.depth

text_prompt = args.text_prompt
text_prompt = sanitize_filename(text_prompt)

# 从 image_path 解析文件名和文件夹
img_path = args.image_path
image_base = os.path.basename(img_path)   # 如 "000000_scene2.png"
image_name = os.path.splitext(image_base)[0]  # 去掉扩展名 "000000_scene2"

# 路径配置
json_path = "Grounded-SAM-2/outputs/grounded_sam2_hf_demo/grounded_sam2_hf_model_demo_results.json"

if depth:
    depth_path = f"data/{image_name}_depth.png"
    output_depth_path = f"data/{image_name}_depth_focus.png"

# 加载原始图像
image = np.array(Image.open(img_path).convert("RGB"))
image_height, image_width = image.shape[:2]

# 如果需要深度
if depth:
    depth_img = Image.open(depth_path)
    depth_array = np.array(depth_img)

# 加载 JSON 文件
with open(json_path, "r") as f:
    result = json.load(f)

# 提取 mask 信息
target_index = 0
rle = result["annotations"][target_index]["segmentation"]
mask = mask_util.decode(rle)

# 提取 bounding box
xmin, ymin, xmax, ymax = result["annotations"][target_index]["bbox"]
xmin = int(round(xmin))
ymin = int(round(ymin))
xmax = int(round(xmax))
ymax = int(round(ymax))

# 对bbox加padding 0.1
w = xmax - xmin + 1
h = ymax - ymin + 1

# 循环生成不同padding
for idx, pad_ratio in enumerate(np.linspace(0, 1.0, 11)):
    # 复制原始bbox
    xmin_i, ymin_i, xmax_i, ymax_i = xmin, ymin, xmax, ymax

    # 计算padding
    w = xmax_i - xmin_i + 1
    h = ymax_i - ymin_i + 1
    pad_w = int(round(w * pad_ratio))
    pad_h = int(round(h * pad_ratio))

    # 扩展bbox，防止越界
    xmin_i = max(0, xmin_i - pad_w)
    ymin_i = max(0, ymin_i - pad_h)
    xmax_i = min(image_width - 1, xmax_i + pad_w)
    ymax_i = min(image_height - 1, ymax_i + pad_h)

    # 裁剪区域
    obj_img = image[ymin_i:ymax_i+1, xmin_i:xmax_i+1]
    obj_mask = mask[ymin_i:ymax_i+1, xmin_i:xmax_i+1]

    # 透明背景处理，确保掩码外RGB白色，Alpha为0
    obj_img_masked = obj_img.copy()
    mask_indices = np.where(obj_mask == 0)
    obj_img_masked[mask_indices] = [255, 255, 255]  # 背景白色

    h_crop, w_crop = obj_img.shape[:2]
    rgba_img = np.zeros((h_crop, w_crop, 4), dtype=np.uint8)

    # RGB通道赋值为obj_img_masked
    rgba_img[..., :3] = obj_img_masked

    # Alpha通道：掩码区域255，其它0
    rgba_img[..., 3] = obj_mask.astype(np.uint8) * 255

    # 分离RGB和Alpha通道
    rgb = rgba_img[..., :3]
    alpha = rgba_img[..., 3]


    if depth:
        mask_indices = np.where(obj_mask > 0)
        depth_crop = depth_array[ymin_i:ymax_i+1, xmin_i:xmax_i+1]
        depth_bg = np.zeros_like(depth_crop)
        depth_bg[mask_indices] = depth_crop[mask_indices]

    # 保证正方形
    target_size = 128
    side_length = max(h_crop, w_crop, target_size)

    pad_top = (side_length - h_crop) // 2
    pad_bottom = side_length - h_crop - pad_top
    pad_left = (side_length - w_crop) // 2
    pad_right = side_length - w_crop - pad_left

    # Padding
    rgb_padded = np.pad(
        rgb,
        ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
        mode="constant",
        constant_values=255  # RGB padding用白色填充
    )

    alpha_padded = np.pad(
        alpha,
        ((pad_top, pad_bottom), (pad_left, pad_right)),
        mode="constant",
        constant_values=0  # Alpha padding用透明填充
    )

    rgba_padded = np.concatenate([rgb_padded, alpha_padded[..., None]], axis=2)

    # 输出文件名
    output_path = f"data_eval/{image_name}_{text_prompt}_focus{idx}.png"

    Image.fromarray(rgba_padded).save(output_path)
    print(f"Saved: {output_path}")

    if depth:
        depth_padded = np.pad(
            depth_bg,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode="constant",
            constant_values=0
        )
        output_depth_path = f"data_eval/{image_name}_{text_prompt}_depth_focus{idx}.png"
        Image.fromarray(depth_padded).save(output_depth_path)