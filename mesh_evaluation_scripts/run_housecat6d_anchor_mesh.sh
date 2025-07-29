#!/bin/bash

# Iterate through mesh_huanyuan folder and find corresponding mask files
BENCHMARK_MESH_ROOT="/mnt/projects/at3dcv/6D_pose/dataset/benchmark_target/mesh_huanyuan/groundtruth_rectangle_bbox_focus3"

for i in $(seq -w 8 9); do
    SCENE_NAME="scene$(printf "%02d" $i)"
    SCENE_DIR="/mnt/projects/at3dcv/6D_pose/dataset/housecat6d/$SCENE_NAME"
    COLOR_IMG_DIR="$SCENE_DIR/rgb"
    MASK_DIR="$SCENE_DIR/instance"
    LABEL_DIR="$SCENE_DIR/labels"
    
    MESH_SCENE_DIR="$BENCHMARK_MESH_ROOT/$SCENE_NAME"
    
    # Check if mesh scene directory exists
    if [ ! -d "$MESH_SCENE_DIR" ]; then
        echo "Warning: Mesh scene directory not found: $MESH_SCENE_DIR, skipping scene $SCENE_NAME"
        continue
    fi

    color_img="$COLOR_IMG_DIR/000000.png"

    # Iterate through each object folder in the mesh scene directory
    for obj_folder in "$MESH_SCENE_DIR"/*; do
        if [ ! -d "$obj_folder" ]; then
            continue
        fi
        
        obj_name=$(basename "$obj_folder")
        
        # Use the folder name as the object name for benchmark mesh
        OBJ="$obj_name"
        
        # Find mesh file in the object folder (should be 000000_objname.obj)
        BENCHMARK_MESH="$obj_folder/000000_${obj_name}.obj"
        
        if [ ! -f "$BENCHMARK_MESH" ]; then
            echo "Warning: Expected benchmark mesh not found: $BENCHMARK_MESH, skipping object $obj_name"
            continue
        fi
        
        echo "Debug: Using benchmark mesh: $BENCHMARK_MESH"
        echo "Debug: Object name from folder: $OBJ"
        
        # Find corresponding mask file
        mask_file="$MASK_DIR/000000_${OBJ}.png"
        
        # If exact match not found, try with object folder name
        if [ ! -f "$mask_file" ]; then
            mask_file="$MASK_DIR/000000_${obj_name}.png"
        fi
        
        # If still not found, skip this object
        if [ ! -f "$mask_file" ]; then
            echo "Warning: No corresponding mask file found for object $OBJ (tried $obj_name), skipping..."
            continue
        fi

        # Find GT mesh (same logic as run_housecat6d_anchor.sh)
        GT_MESH="/mnt/projects/at3dcv/6D_pose/dataset/housecat6d/obj_models_small_size_final"
        
        # Extract the part before '-' in OBJ
        OBJ_PREFIX="${OBJ%%-*}"
        
        # Find the corresponding folder in GT_MESH
        MESH_FOLDER="$GT_MESH/$OBJ_PREFIX"
        
        # Find the exact .obj file for the mesh
        GT_MESH="$MESH_FOLDER/${OBJ}.obj"

        echo "Processing scene: $SCENE_NAME"
        echo "Processing color image: $(basename "$color_img") with OBJ: $OBJ"
        echo "Found benchmark mesh: $BENCHMARK_MESH"
        echo "Found corresponding mask: $mask_file"
        
        # Find the corresponding label file for the current OBJ
        LABEL_FILE="$LABEL_DIR/000000_label.pkl"
        echo "Using label file: $LABEL_FILE"
        echo "Using GT mesh file: $GT_MESH"

        echo python run_housecat6d_anchor_uniform_resize.py \
            --scene_dir "$SCENE_DIR" \
            --color_img "$(basename "$color_img")" \
            --label_file "$LABEL_FILE" \
            --obj "$OBJ" \
            --gt_mesh "$GT_MESH" \
            --mask_path "$mask_file" \
            --mesh_path "$BENCHMARK_MESH"
        python run_housecat6d_anchor_uniform_resize.py \
            --scene_dir "$SCENE_DIR" \
            --color_img "$(basename "$color_img")" \
            --label_file "$LABEL_FILE" \
            --obj "$OBJ" \
            --gt_mesh "$GT_MESH" \
            --mask_path "$mask_file" \
            --mesh_path "$BENCHMARK_MESH"
        
        # # just one object
        # break
    done
done