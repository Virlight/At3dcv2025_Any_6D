#!/bin/bash

SCENE_DIR="dataset_house6d/scene2"
COLOR_IMG_DIR="$SCENE_DIR/rgb_relightning"
MESH_DIR="$SCENE_DIR/generated_mesh/can"
GT_POSE="can-fanta.txt"
GT_MESH="can-fanta.obj"
MASK_PATH="000000_can-fanta.png"

# Get the first color image only
color_img=$(ls "$COLOR_IMG_DIR"/* | head -n 1)
if [ -z "$color_img" ]; then
    echo "No color images found in $COLOR_IMG_DIR"
    exit 1
fi

for mesh_path in "$MESH_DIR"/*; do
    mesh_name=$(basename "$mesh_path")
    mesh_name_base="${mesh_name%.*}"
    color_img_name=$(basename "$color_img")
    color_img_name_base="${color_img_name%.*}"
    OBJ="${color_img_name_base}_${mesh_name_base}"
    echo "Processing color image: $color_img_name with mesh: $mesh_name"
    echo python run.py \
        --scene_dir "$SCENE_DIR" \
        --color_img "$color_img_name" \
        --mesh_path "$mesh_name" \
        --obj "$OBJ" \
        --gt_pose "$GT_POSE"\
        --gt_mesh "$GT_MESH"\
        --mask_path "$MASK_PATH"
    python run.py \
        --scene_dir "$SCENE_DIR" \
        --color_img "$color_img_name" \
        --mesh_path "$mesh_path" \
        --obj "$OBJ" \
        --gt_pose "$GT_POSE"\
        --gt_mesh "$GT_MESH"\
        --mask_path "$MASK_PATH"
done
