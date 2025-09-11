#!/bin/bash

# 目标数组，格式：scene_number target_name
# declare -a targets=(
#   "1 can-kidney_beans"
#   "2 remote-black"
#   "4 bottle-evian_frozen"
#   "4 cup-white_whisker"
#   "4 cutlery-spoon_1"
#   "4 box-barilla"
#   "5 glass-small"
#   "5 remote-silver"
#   "6 cutlery-knife_1"
#   "7 teapot-blue_floral"
#   "8 bottle-85_alcool"
#   "8 shoe-white_viva_sandal_right"
#   "9 bottle-deodorant_spray"
#   "10 teapot-green_grass"
#   "10 box-iglo"
# )

declare -a targets=(
  "1 can"
  "2 remote"
  "4 bottle"
  "4 cup"
  "4 spoon"
  "4 blue-box"
  "5 glass"
  "5 remote"
  "6 knife"
  "7 teapot"
  "8 bottle"
  "8 shoe"
  "9 bottle"
  "10 teapot"
  "10 blue-box"
)

BASE_SCENE_DIR="/media/haoliang/windows/linux_share/housecat6d"
GSAM_DIR="Grounded-SAM-2"

for target in "${targets[@]}"; do
  read -r scene_num target_name <<< "$target"

  # 生成两位数场景名，例如 scene01, scene02...
  scene_dir="${BASE_SCENE_DIR}/scene$(printf "%02d" "$scene_num")"
  # img_path="${scene_dir}/rgb/000000.png"

  img_path="../data/000000_scene$(printf "%02d" "$scene_num").png"

  # 文本提示就是目标名
  text_prompt="$target_name"

  echo "Processing Scene: $scene_dir"
  echo "Using Image: $img_path"
  echo "Text prompt: $text_prompt"

  # 保存当前目录
  cwd=$(pwd)

  # 进入 Grounded-SAM-2 目录
  cd "$GSAM_DIR" || { echo "Failed to enter $GSAM_DIR"; exit 1; }

  # 运行 SAM 脚本
  python grounded_sam2_hf_model_demo.py --text-prompt "$text_prompt." --img-path "$img_path"

  # 返回原目录
  cd "$cwd" || { echo "Failed to return to $cwd"; exit 1; }

  img_path="data/000000_scene$(printf "%02d" "$scene_num").png"

  # 运行提取脚本
  python src/extract_focus_mask_rgba.py --text-prompt "$text_prompt" --image-path "$img_path"

  echo "Finished processing $text_prompt in scene $scene_num"
  echo "------------------------------"
done

echo "All targets processed."