import json
import numpy as np
from PIL import Image
import pycocotools.mask as mask_util

depth = False  # 是否处理深度图像

# 路径配置
image_base = "000000_scene2.png"
image_name = image_base.split(".")[0]
img_path = f"data/{image_base}"
json_path = "Grounded-SAM-2/outputs/grounded_sam2_hf_demo/grounded_sam2_hf_model_demo_results.json"
output_path = f"data/{image_name}_focus.png"

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
for idx, pad_ratio in enumerate(np.linspace(0.1, 1.0, 10), start=1):
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

    # 白底
    h_crop, w_crop = obj_img.shape[:2]
    white_bg = np.ones_like(obj_img) * 255

    mask_indices = np.where(obj_mask > 0)
    white_bg[mask_indices] = obj_img[mask_indices]

    if depth:
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
        white_bg,
        ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
        mode="constant",
        constant_values=255
    )

    # 输出文件名
    output_path = f"data/{image_name}_focus{idx}.png"

    Image.fromarray(rgb_padded).save(output_path)
    print(f"Saved: {output_path}")

    if depth:
        depth_padded = np.pad(
            depth_bg,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode="constant",
            constant_values=0
        )
        output_depth_path = f"data/{image_name}_depth_focus{idx}.png"
        Image.fromarray(depth_padded).save(output_depth_path)