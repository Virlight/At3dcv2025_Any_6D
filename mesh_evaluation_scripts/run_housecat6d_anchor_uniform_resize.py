import os
import glob
import numpy as np
import cv2
import trimesh
import pandas as pd
import pickle
from tqdm import tqdm

from estimater_uniform import Any6D
from foundationpose.Utils_for_mesh_compare import get_bounding_box, vis_mask_contours, draw_xyz_axis,visualize_frame_results, calculate_chamfer_distance_gt_mesh, align_mesh_to_coordinate
import nvdiffrast.torch as dr
import argparse
from pytorch_lightning import seed_everything

from sam2_instantmesh import *
import logging



# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("run_housecat6d_anchor.log"),
        logging.StreamHandler()
    ]
)

def load_intrinsics(path):
    return np.loadtxt(path)

def load_label(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="HouseCat6D Anchor Demo")
    parser.add_argument("--img_to_3d", action="store_true", help="Running with InstantMesh+SAM2")
    parser.add_argument("--scene_dir", type=str, required=True, help="Path to scene directory (e.g. dataset/housecat6d/scene01)")
    parser.add_argument("--mesh_path", type=str, default=None, help="Path to mesh file (or use label)")
    parser.add_argument("--obj", type=str, required=True, help="Object name (e.g. box-kfc)")
    parser.add_argument("--frame", type=str, default="000000", help="Frame name (e.g. 000000)")
    parser.add_argument("--color_img", type=str, default=None, help="Color image filename (default: {frame}.png)")
    parser.add_argument("--depth_img", type=str, default=None, help="Depth image filename (default: {frame}.png)")
    parser.add_argument("--label_file", type=str, default=None, help="Label file name (default: labels/{frame}_label.pkl)")
    parser.add_argument("--mask_path", type=str, default=None, help="Mask file name (default: instance/{frame}_{obj}.png)")
    parser.add_argument("--intrinsic_path", type=str, default="intrinsics.txt", help="Camera intrinsic file")
    parser.add_argument("--gt_mesh", type=str, default=None, help="Ground truth mesh path (optional)")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory root")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    seed_everything(0)
    glctx = dr.RasterizeCudaContext()
    frame = args.frame
    obj = args.obj
    scene_dir = args.scene_dir
    mesh_path = args.mesh_path
    output_dir = os.path.join("results", os.path.basename(scene_dir), obj, frame)
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(filename=os.path.join(output_dir, 'process.log'), level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    logging.info("Started HouseCat6D anchor script.")

    # Color and depth
    color_img = args.color_img or f"{frame}.png"
    depth_img = args.depth_img or f"{frame}.png"
    color = cv2.cvtColor(cv2.imread(os.path.join(scene_dir, "rgb", color_img)), cv2.COLOR_BGR2RGB)
    depth = cv2.imread(os.path.join(scene_dir, "depth", depth_img), cv2.IMREAD_ANYDEPTH).astype(np.float32) / 1000.0

    # Label
    label_file = args.label_file or os.path.join(scene_dir, "labels", f"{frame}_label.pkl")
    if os.path.exists(label_file):
        with open(label_file, 'rb') as f:
            label = pickle.load(f)
    else:
        label = None

    # invert the mask for any6d
    mask_path = args.mask_path or os.path.join(scene_dir, "instance", f"{frame}_{obj}.png")
    if os.path.exists(mask_path):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask != 255)
        if mask.shape != color.shape[:2]:
            mask = cv2.resize(mask.astype(np.uint8), (color.shape[1], color.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
        logging.info(f"Loaded mask from {mask_path} and resized to color image size.")
    else:
        mask = None
        logging.warning(f"Mask file not found at {mask_path}, skipping mask loading.")

    # Mesh
    if args.img_to_3d:
        logging.info("Running InstantMesh+SAM2 pipeline.")
        cv2.imwrite(os.path.join(output_dir, f'{obj}_mask.png'), mask.astype(np.uint8) * 255)
        cmin, rmin, cmax, rmax = get_bounding_box(mask).astype(np.int32)
        input_box = np.array([cmin, rmin, cmax, rmax])[None, :]
        mask_refine = running_sam_box(color, input_box)
        cv2.imwrite(os.path.join(output_dir, f'{obj}_mask_refine.png'), mask_refine.astype(np.uint8) * 255)

        input_image = preprocess_image(color, mask_refine, output_dir, obj)
        images = diffusion_image_generation(output_dir, output_dir, obj, input_image=input_image)
        instant_mesh_process(images, output_dir, obj)
        
        mesh = trimesh.load(os.path.join(output_dir, f'mesh_{obj}.obj'))
        logging.info("Generated and aligned mesh using InstantMesh+SAM2.")
    else:
        mesh = trimesh.load(mesh_path)
        logging.info(f"Loaded mesh from {mesh_path}.")

    mesh = align_mesh_to_coordinate(mesh)
    mesh.export(os.path.join(output_dir, f'center_mesh_{obj}.obj'))

    est = Any6D(symmetry_tfs=None, mesh=mesh, debug_dir=output_dir, debug=2 if args.debug else 0)
    logging.info("Initialized Any6D estimator.")

    # Intrinsic
    intrinsic_path = os.path.join(scene_dir, args.intrinsic_path)
    if intrinsic_path.endswith('.yml') or intrinsic_path.endswith('.yaml'):
        with open(intrinsic_path, 'r') as file:
            data = yaml.load(file, Loader=yaml.FullLoader)
        intrinsic = np.array([[data["depth"]["fx"], 0.0, data["depth"]["ppx"]],
                              [0.0, data["depth"]["fy"], data["depth"]["ppy"]],
                              [0.0, 0.0, 1.0]])
    elif intrinsic_path.endswith('.txt'):
        intrinsic = np.loadtxt(intrinsic_path)
    else:
        raise ValueError(f"Unsupported intrinsic file format: {intrinsic_path}")
    np.savetxt(os.path.join(output_dir, f'K.txt'), intrinsic)
    logging.info("Loaded and saved camera intrinsic matrix.")

    # Pose estimation
    pred_pose = est.register_any6d(K=intrinsic, rgb=color, depth=depth, ob_mask=mask, iteration=5, name=obj, uniform_scale=True)
    logging.info("Estimated pose using Any6D.")

    # GT pose
    if label and "model_list" in label and obj in label["model_list"]:
        index = label["model_list"].index(obj)
        rotation = label['rotations'][index]
        translation = label['translations'][index]
        tmp = np.hstack((rotation, translation.reshape(3, 1)))
        gt_pose = np.eye(4)
        gt_pose[:3, :] = tmp
        logging.info(f"Loaded GT pose for {obj}.")
    else:
        gt_pose = None
        logging.warning(f"GT pose for {obj} not found in label.")

    # Visualization
    if gt_pose is not None:
        # Visualize GT pose
        gt_img = vis_mask_contours(color, mask, color=(255, 1, 154))
        gt_pose_img = draw_xyz_axis(gt_img, ob_in_cam=gt_pose, scale=0.1, K=intrinsic, thickness=3, transparency=0, is_input_rgb=True)

        # Visualize Predicted pose
        pred_img = vis_mask_contours(color, mask, color=(1, 255, 154))
        pred_pose_img = draw_xyz_axis(pred_img, ob_in_cam=pred_pose, scale=0.1, K=intrinsic, thickness=3, transparency=0, is_input_rgb=True)
        
        # Concatenate GT and Predicted pose images side by side and save
        concat_img = np.concatenate((gt_pose_img, pred_pose_img), axis=1)
        cv2.imwrite(os.path.join(output_dir, f'{obj}_gt_pred_concat.png'), concat_img)

    # GT mesh
    gt_mesh_path = args.gt_mesh or mesh_path
    gt_mesh = trimesh.load(gt_mesh_path)
    gt_mesh.export(os.path.join(output_dir, f'gt_mesh_{obj}.obj'))
    print(f"Loaded GT mesh from {gt_mesh_path}.")

    visualize_frame_results(color=color, gt_mesh=gt_mesh, est=est, K=intrinsic, gt_pose=gt_pose, pred_pose=pred_pose,
                        metric=None, obj_f=obj, frame_idx=0, save_path=output_dir, glctx=glctx,
                        name=f'demo_data', mesh_index=0, init=False, save_on_folder=True)


    chamfer_dis = calculate_chamfer_distance_gt_mesh(gt_pose, gt_mesh, pred_pose, est.mesh, output_dir)
    logging.info(f"Calculated Chamfer Distance: {chamfer_dis}")

    np.savetxt(os.path.join(output_dir, f'{obj}_initial_pose.txt'), pred_pose)
    if gt_pose is not None:
        np.savetxt(os.path.join(output_dir, f'{obj}_gt_pose.txt'), gt_pose)
    est.mesh.export(os.path.join(output_dir, f'final_mesh_{obj}.obj'))
    np.savetxt(os.path.join(output_dir, f'{obj}_cd.txt'), [chamfer_dis])

    # Append Chamfer Distance to existing Excel file with scene and object info
    chamfer_df = pd.DataFrame([{
        "scene": os.path.basename(scene_dir),
        "object": obj,
        "chamfer_distance": chamfer_dis
    }])
    chamfer_csv_path = "results/chamfer_distances.csv"
    if os.path.exists(chamfer_csv_path):
        chamfer_df.to_csv(chamfer_csv_path, mode='a', header=False, index=False)
    else:
        chamfer_df.to_csv(chamfer_csv_path, index=False)
    # Optionally: plot predicted pose
    # ...existing code for plotting if needed...
    
