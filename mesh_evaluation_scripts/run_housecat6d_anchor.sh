#!/bin/bash

# run on the 000000 in every scene every object
for i in $(seq -w 1); do
    SCENE_DIR="/mnt/projects/at3dcv/6D_pose/dataset/housecat6d/scene$(printf "%02d" $i)"
    COLOR_IMG_DIR="$SCENE_DIR/rgb"
    
    MASK_DIR="$SCENE_DIR/instance"
    LABEL_DIR="$SCENE_DIR/labels"

    color_img="$COLOR_IMG_DIR/000000.png"

    # Find the file in MASK_DIR that starts with 000000 and remove '000000_' prefix for OBJ
    for mask_file in $(find "$MASK_DIR" -type f -name '000000_*'); do
        # reset the GT_MESH every time
        GT_MESH="/mnt/projects/at3dcv/6D_pose/dataset/housecat6d/obj_models_small_size_final"

        mask_filename=$(basename "$mask_file")
        OBJ="${mask_filename#000000_}"
        OBJ="${OBJ%.*}"

        # Extract the part before '-' in OBJ
        OBJ_PREFIX="${OBJ%%-*}"

        # Find the corresponding folder in GT_MESH
        MESH_FOLDER="$GT_MESH/$OBJ_PREFIX"

        # Find the exact .obj file for the mesh
        GT_MESH="$MESH_FOLDER/${OBJ}.obj"

        echo "Processing scene: scene${i}"
        echo "Processing color image: $(basename "$color_img") with OBJ: $OBJ"
        # Find the corresponding label file for the current OBJ
        LABEL_FILE="$LABEL_DIR/000000_label.pkl"
        echo "Using label file: $LABEL_FILE"
        echo "Using mesh file: $GT_MESH"
        echo python run_housecat6d_anchor.py \
            --scene_dir "$SCENE_DIR" \
            --color_img "$(basename "$color_img")" \
            --label_file "$LABEL_FILE" \
            --obj "$OBJ" \
            --gt_mesh "$GT_MESH" \
            --mask_path "$mask_file" \
            --img_to_3d 
        python run_housecat6d_anchor.py \
            --scene_dir "$SCENE_DIR" \
            --color_img "$(basename "$color_img")" \
            --label_file "$LABEL_FILE" \
            --obj "$OBJ" \
            --gt_mesh "$GT_MESH" \
            --mask_path "$mask_file" \
            --img_to_3d 
    break
    done
done