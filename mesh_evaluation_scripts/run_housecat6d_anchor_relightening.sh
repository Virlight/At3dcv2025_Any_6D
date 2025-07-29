#!/bin/bash

# run on the 000000 in every scene every object
BRIGHTENED_DIR="/mnt/projects/at3dcv/6D_pose/dataset/benchmark_target/input_image_brightened"

# Iterate through scene folders in input_image_brightened
for scene_folder in "$BRIGHTENED_DIR"/scene*/; do
    scene_name=$(basename "$scene_folder")
    SCENE_DIR="/mnt/projects/at3dcv/6D_pose/dataset/housecat6d/$scene_name"
    
    MASK_DIR="$SCENE_DIR/instance"
    LABEL_DIR="$SCENE_DIR/labels"

    echo "Processing scene: $scene_name"
    
    # Iterate through object folders in the current scene
    for obj_folder in "$scene_folder"*/; do
        obj_name=$(basename "$obj_folder")
        
        echo "  Processing object: $obj_name"
        
        # Iterate through all brightened images in the object folder
        for color_img_path in "$scene_folder"*.png; do
            color_img=$(basename "$color_img_path")
            
            echo "    Processing image: $color_img"
            
            # Find the corresponding mask file for this object
            mask_file="$MASK_DIR/000000_${obj_name}.png"
            
            if [ ! -f "$mask_file" ]; then
                echo "    Warning: Mask file not found: $mask_file"
                continue
            fi
            
            # reset the GT_MESH every time
            GT_MESH="/mnt/projects/at3dcv/6D_pose/dataset/housecat6d/obj_models_small_size_final"

            OBJ="$obj_name"

            # Extract the part before '-' in OBJ
            OBJ_PREFIX="${OBJ%%-*}"

            # Find the corresponding folder in GT_MESH
            MESH_FOLDER="$GT_MESH/$OBJ_PREFIX"

            # Find the exact .obj file for the mesh
            GT_MESH="$MESH_FOLDER/${OBJ}.obj"

            # Find the corresponding label file for the current OBJ
            LABEL_FILE="$LABEL_DIR/000000_label.pkl"
            
            echo "      Using label file: $LABEL_FILE"
            echo "      Using mesh file: $GT_MESH"
            echo "      Using mask file: $mask_file"
            
            echo python run_housecat6d_anchor.py \
                --scene_dir "$SCENE_DIR" \
                --color_img "$color_img_path" \
                --label_file "$LABEL_FILE" \
                --obj "$OBJ" \
                --gt_mesh "$GT_MESH" \
                --mask_path "$mask_file" \
                --img_to_3d 
            python run_housecat6d_anchor.py \
                --scene_dir "$SCENE_DIR" \
                --color_img "$color_img_path" \
                --label_file "$LABEL_FILE" \
                --obj "$OBJ" \
                --gt_mesh "$GT_MESH" \
                --mask_path "$mask_file" \
                --img_to_3d 
        done
    done
    # # just one scene
    # break
done