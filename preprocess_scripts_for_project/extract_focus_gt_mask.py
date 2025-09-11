import json
import numpy as np
from PIL import Image
import pycocotools.mask as mask_util
import os
import cv2

# # 路径配置
# image_base = "000000_scene2.png"
# image_name = image_base.split(".")[0]
# img_path = f"data/{image_base}"

# SCENE_DIR="/media/haoliang/windows/linux_share/housecat6d/scene02"
# MASK_DIR= f"{SCENE_DIR}/instance"
# mask_files = [
#     os.path.join(MASK_DIR, "000000_cup-stanford.png"),
#     os.path.join(MASK_DIR, "000000_cutlery-fork_1.png"),
#     os.path.join(MASK_DIR, "000000_remote-black.png"),
#     os.path.join(MASK_DIR, "000000_bottle-dettol_washing_machine.png")
# ]

depth = False  # 是否处理深度图像

BASE_DIR = "/media/haoliang/windows/linux_share/housecat6d"

# 目标列表（编号和名称）
targets = [
    "1 can-kidney_beans",
    "2 remote-black",
    "4 bottle-evian_frozen",
    "4 cup-white_whisker",
    "4 cutlery-spoon_1",
    "4 box-barilla",
    "5 glass-small",
    "5 remote-silver",
    "6 cutlery-knife_1",
    "7 teapot-blue_floral",
    "8 bottle-85_alcool",
    "8 shoe-white_viva_sandal_right",
    "9 bottle-deodorant_spray",
    "10 teapot-green_grass",
    "10 box-iglo"
]

for target in targets:
    scene_num, obj_name = target.split(" ", 1)
    scene_str = f"scene{int(scene_num):02d}"  # 格式化成两位数，如scene01，scene02

    SCENE_DIR = os.path.join(BASE_DIR, scene_str)
    MASK_DIR = os.path.join(SCENE_DIR, "instance")

    image_base = f"000000_{scene_str}.png"  # 例如 000000_scene02.png
    image_name = image_base.split(".")[0]
    img_path = os.path.join("data", image_base)  # 你的原始图片路径，按需修改

    mask_path = os.path.join(MASK_DIR, f"000000_{obj_name}.png")

    if not os.path.exists(mask_path):
        print(f"Warning: Mask file {mask_path} does not exist, skipping.")
        continue

    # 读取mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = (mask != 255)

    coords = np.argwhere(mask)
    print(f"Processing {mask_path} with {coords.shape} mask pixels.")
    if coords.shape[0] == 0:
        print(f"No valid mask pixels found in {mask_path}, skipping.")
        continue  # 跳过当前循环
    ymin, xmin = coords.min(axis=0)
    ymax, xmax = coords.max(axis=0)

    if depth:
        depth_path = f"data/{image_name}_depth.png"
        output_depth_path = f"data/{image_name}_depth_focus.png"

    # 加载原始图像（这里按需改路径）
    if not os.path.exists(img_path):
        print(f"Warning: Image file {img_path} does not exist, skipping.")
        continue
    image = np.array(Image.open(img_path).convert("RGB"))
    image_height, image_width = image.shape[:2]

    if depth:
        depth_img = Image.open(depth_path)
        depth_array = np.array(depth_img)

    w = xmax - xmin + 1
    h = ymax - ymin + 1

    for idx, pad_ratio in enumerate(np.linspace(0, 1.0, 11)):
        xmin_i, ymin_i, xmax_i, ymax_i = xmin, ymin, xmax, ymax

        w = xmax_i - xmin_i + 1
        h = ymax_i - ymin_i + 1
        pad_w = int(round(w * pad_ratio))
        pad_h = int(round(h * pad_ratio))

        xmin_i = max(0, xmin_i - pad_w)
        ymin_i = max(0, ymin_i - pad_h)
        xmax_i = min(image_width - 1, xmax_i + pad_w)
        ymax_i = min(image_height - 1, ymax_i + pad_h)

        obj_img = image[ymin_i:ymax_i + 1, xmin_i:xmax_i + 1]
        obj_mask = mask[ymin_i:ymax_i + 1, xmin_i:xmax_i + 1]

        # 把掩码外区域RGB设为白色
        obj_img_masked = obj_img.copy()
        mask_indices = np.where(obj_mask == 0)
        obj_img_masked[mask_indices] = [255, 255, 255]

        h_crop, w_crop = obj_img.shape[:2]
        rgba_img = np.zeros((h_crop, w_crop, 4), dtype=np.uint8)

        # RGB通道赋值为obj_img_masked
        rgba_img[..., :3] = obj_img_masked

        # Alpha通道：掩码区域255，其它0
        rgba_img[..., 3] = obj_mask.astype(np.uint8) * 255

        if depth:
            mask_indices = np.where(obj_mask > 0)
            depth_crop = depth_array[ymin_i:ymax_i + 1, xmin_i:xmax_i + 1]
            depth_bg = np.zeros_like(depth_crop)
            depth_bg[mask_indices] = depth_crop[mask_indices]

        target_size = 128
        side_length = max(h_crop, w_crop, target_size)

        pad_top = (side_length - h_crop) // 2
        pad_bottom = side_length - h_crop - pad_top
        pad_left = (side_length - w_crop) // 2
        pad_right = side_length - w_crop - pad_left

        # 分离RGB和Alpha通道
        rgb = rgba_img[..., :3]
        alpha = rgba_img[..., 3]

        # 分别pad，RGB用白色，Alpha用透明
        rgb_padded = np.pad(
            rgb,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode="constant",
            constant_values=255
        )

        alpha_padded = np.pad(
            alpha,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode="constant",
            constant_values=0
        )

        rgba_padded = np.concatenate([rgb_padded, alpha_padded[..., None]], axis=2)

        output_dir = "data_eval_gt"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{image_name}_{obj_name}_focus{idx}.png")

        Image.fromarray(rgba_padded).save(output_path)
        print(f"Saved: {output_path}")

        if depth:
            depth_padded = np.pad(
                depth_bg,
                ((pad_top, pad_bottom), (pad_left, pad_right)),
                mode="constant",
                constant_values=0
            )
            output_depth_path = os.path.join(output_dir, f"{image_name}_depth_focus{idx}.png")
            Image.fromarray(depth_padded).save(output_depth_path)
