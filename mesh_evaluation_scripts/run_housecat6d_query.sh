#!/bin/bash

# Run query evaluation for every scene in housecat6d
echo "Starting batch HouseCat6D query evaluation..."

for i in $(seq -w 8 9); do
    SCENE_DIR="/mnt/projects/at3dcv/6D_pose/dataset/housecat6d/scene$(printf "%02d" $i)"
    # the anchor file consists of multiple objects, each with multiple reference frames
    ANCHOR_DIR="/mnt/projects/at3dcv/6D_pose/Any6D/anchor_housecat6d/baseline/scene$(printf "%02d" $i)"
   
    if [ -d "$SCENE_DIR" ] && [ -d "$ANCHOR_DIR" ]; then
        for OBJ_DIR in "$ANCHOR_DIR"/*; do
            if [ -d "$OBJ_DIR" ]; then
                OBJ_NAME=$(basename "$OBJ_DIR")
                for REF_FRAME_DIR in "$OBJ_DIR"/*/; do
                    if [ -d "$REF_FRAME_DIR" ]; then
                        REF_FRAME_NAME=$(basename "$REF_FRAME_DIR")
                        echo "Processing query for $SCENE_DIR, object $OBJ_NAME, reference frame $REF_FRAME_DIR"
                        python run_housecat6d_query_adjust_add.py \
                            --name "housecat6d_scene$(printf "%02d" $i)_${OBJ_NAME}_${REF_FRAME_NAME}" \
                            --query_path "$SCENE_DIR" \
                            --anchor_path "$REF_FRAME_DIR" \
                            --running_stride 10
                    fi
                done
            fi
        done
    else
        echo "Skipping $SCENE_DIR or $ANCHOR_DIR (directory not found)"
    fi
done

echo "Batch HouseCat6D query evaluation complete."
