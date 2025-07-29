
import os
import trimesh
import yaml
import numpy as np
import cv2
import torch

from PIL import Image
from estimater import Any6D

from foundationpose.Utils import get_bounding_box, visualize_frame_results, calculate_chamfer_distance_gt_mesh, align_mesh_to_coordinate
import nvdiffrast.torch as dr
import argparse
# bus error when loading
from pytorch_lightning import seed_everything

from sam2_instantmesh import *
import logging
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

glctx = dr.RasterizeCudaContext()

if __name__=='__main__':

    # Setup logging
    log_path = 'results/demo_mustard/process.log'
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logging.basicConfig(filename=log_path, level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    logging.info("Started demo script.")
    seed_everything(0)
    logging.info("Set random seed.")

    parser = argparse.ArgumentParser(description="Set experiment name and paths")
    parser.add_argument("--ycb_model_path", type=str, default="YCB_Video_Models", help="Path to the YCB Video Models")
    parser.add_argument("--img_to_3d", action="store_true",help="Running with InstantMesh+SAM2")
    parser.add_argument("--plot_views", action="store_true", help="plot for difference")
    args = parser.parse_args()

    ycb_model_path = args.ycb_model_path
    img_to_3d = args.img_to_3d

    results = []
    demo_path = 'demo_data'
    mesh_path = os.path.join(demo_path, f'mustard.obj')
    
    

    depth_scale = 1000.0
    color_files = sorted([f for f in os.listdir(demo_path) if f.startswith('color') and (f.endswith('.png') or f.endswith('.jpg'))])
    print(color_files)
    
    for color_file in color_files:
        # Set obj based on the base name of the color_file (e.g., 'color_mustard.png' -> 'mustard')
        base = os.path.splitext(os.path.basename(color_file))[0]  # e.g., 'color_mustard'
        if base.startswith('color_'):
            obj = base[len('color_'):]  # e.g., 'mustard'
        else:
            obj = base
        print(obj)
        save_path = f'results/{obj}'
    # Load the demo input
        logging.info("Loading demo input.")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            logging.info(f"Created directory: {save_path}")

        if color_file.endswith('.png'):
            color = cv2.cvtColor(cv2.imread(os.path.join(demo_path, color_file)), cv2.COLOR_BGR2RGB)
            depth = cv2.imread(os.path.join(demo_path, 'depth.png'), cv2.IMREAD_ANYDEPTH).astype(np.float32) / depth_scale
            Image.fromarray(color).save(os.path.join(save_path, color_file))
            logging.info("Loaded color and depth images for PNG.")
        elif color_file.endswith('.jpg'): # change to
            color = cv2.cvtColor(cv2.imread(os.path.join(demo_path, color_file)), cv2.COLOR_BGR2RGB)
            # Adjust depth filename if needed, here assuming 'depth.jpg' exists for jpg color
            depth_file = color_file.replace('color', 'depth').replace('.jpg', '.png')
            if not os.path.exists(os.path.join(demo_path, depth_file)):
                depth_file = 'depth.png'  # fallback
            depth = cv2.imread(os.path.join(demo_path, depth_file), cv2.IMREAD_ANYDEPTH).astype(np.float32) / depth_scale
            Image.fromarray(color).save(os.path.join(save_path, color_file))
            logging.info("Loaded color and depth images for JPG.")

        # load the mask for object
        label = np.load(os.path.join(demo_path, 'labels.npz'))
        obj_num = 5
        mask = np.where(label['seg'] == obj_num, 255, 0).astype(np.bool_)
        logging.info("Loaded label and mask.")

        if img_to_3d:
            logging.info("Running InstantMesh+SAM2 pipeline.")
            cmin, rmin, cmax, rmax = get_bounding_box(mask).astype(np.int32)
            input_box = np.array([cmin, rmin, cmax, rmax])[None, :]
            mask_refine = running_sam_box(color, input_box)

            # the input_image is after segmented
            input_image = preprocess_image(color, mask_refine, save_path, obj)
            # Save the original color image and the input_image
            input_image.save(os.path.join(save_path, f'{obj}_input_image.png'))
            images = diffusion_image_generation(save_path, save_path, obj, input_image=input_image)
            instant_mesh_process(images, save_path, obj)

            mesh = trimesh.load(os.path.join(save_path, f'mesh_{obj}.obj'))
            mesh = align_mesh_to_coordinate(mesh)
            mesh.export(os.path.join(save_path, f'center_mesh_{obj}.obj'))

            mesh = trimesh.load(os.path.join(save_path, f'center_mesh_{obj}.obj'))
            logging.info("Generated and aligned mesh using InstantMesh+SAM2.")
        else:
            mesh = trimesh.load(mesh_path)
            logging.info(f"Loaded mesh from {mesh_path}.")

        est = Any6D(symmetry_tfs=None, mesh=mesh, debug_dir=save_path, debug=2)
        logging.info("Initialized Any6D estimator.")

        # camera info
        intrinsic_path = f"{demo_path}/836212060125_640x480.yml"
        with open(intrinsic_path, 'r') as file:
            data = yaml.load(file, Loader=yaml.FullLoader)

        intrinsic = np.array([[data["depth"]["fx"], 0.0, data["depth"]["ppx"]], [0.0, data["depth"]["fy"], data["depth"]["ppy"]], [0.0, 0.0, 1.0], ], )
        np.savetxt(os.path.join(save_path, f'K.txt'), intrinsic)
        logging.info("Loaded and saved camera intrinsic matrix.")
        pred_pose = est.register_any6d(K=intrinsic, rgb=color, depth=depth, ob_mask=mask, iteration=5, name=f'demo')
        logging.info("Estimated pose using Any6D.")

        pose_list = label['pose_y']
        index_list = np.unique(label['seg'])
        index = (np.where(index_list == obj_num)[0] - 1).tolist()[0]
        tmp = pose_list[index]
        gt_pose = np.eye(4)
        gt_pose[:3, :] = tmp
    # load gt_mesh
        gt_mesh = trimesh.load(f'{ycb_model_path}/models/006_mustard_bottle/textured_simple.obj')
        logging.info("Loaded ground truth mesh.")
        # Transform est.mesh with pred_pose and save it
        transformed_mesh = est.mesh.copy()
        transformed_mesh.apply_transform(pred_pose)
        transformed_mesh.export(os.path.join(save_path, f'transformed_{obj}.obj'))

        chamfer_dis = calculate_chamfer_distance_gt_mesh(gt_pose, gt_mesh, pred_pose, est.mesh)
        logging.info(f"Calculated Chamfer Distance: {chamfer_dis}")

        np.savetxt(os.path.join(save_path, f'{obj}_initial_pose.txt'), pred_pose)
        np.savetxt(os.path.join(save_path, f'{obj}_gt_pose.txt'), gt_pose)
        # save the estimated mesh
        est.mesh.export(os.path.join(save_path, f'final_mesh_{obj}.obj'))

        np.savetxt(os.path.join(save_path, f'{obj}_cd.txt'), [chamfer_dis])
        logging.info("Saved results to disk.")


        if args.plot_views:
            # Transform predicted mesh vertices to GT coordinate frame and visualize
            import matplotlib.pyplot as plt

            def transform_pts(pts, pose):
                # pts: (N, 3), pose: (4, 4)
                pts_h = np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1)
                pts_trans = (pose @ pts_h.T).T[:, :3]
                return pts_trans

            def draw_coordinate_frame(ax, pose, label, length=0.05):
                origin = pose[:3, 3]
                x_axis = pose[:3, 0] * length
                y_axis = pose[:3, 1] * length
                z_axis = pose[:3, 2] * length
                ax.quiver(*origin, *x_axis, color='r', linewidth=2)
                ax.quiver(*origin, *y_axis, color='g', linewidth=2)
                ax.quiver(*origin, *z_axis, color='b', linewidth=2)
                ax.text(*(origin + 1.5 * length), label, fontsize=10)

            def plot_mesh(ax, vertices, faces, color='cyan', alpha=0.3):
                mesh_collection = Poly3DCollection(vertices[faces], alpha=alpha, facecolor=color, edgecolor='k', linewidths=0.05)
                ax.add_collection3d(mesh_collection)

            # Transform est.mesh vertices to predicted pose, then to GT frame
            pred_pts = est.mesh.vertices
            pred_pts = transform_pts(pred_pts, pred_pose)
            pred_pts = transform_pts(pred_pts, np.linalg.inv(gt_pose))
            faces = est.mesh.faces

            # GT mesh in canonical frame
            gt_pts = gt_mesh.vertices
            gt_faces = gt_mesh.faces

            # Define three viewpoints: (elev, azim)
            viewpoints = [
                (30, 45),   # default isometric
                (90, 0),    # top-down
                (0, 0),     # front
            ]
            for i, (elev, azim) in enumerate(viewpoints):
                fig = plt.figure(figsize=(8, 8))
                ax = fig.add_subplot(111, projection='3d')
                # Plot predicted mesh (transformed to GT frame) in blue
                plot_mesh(ax, pred_pts, faces, color='blue', alpha=0.3)
                # Plot GT mesh in orange
                plot_mesh(ax, gt_pts, gt_faces, color='orange', alpha=0.3)
                # Draw GT coordinate frame (identity in GT frame)
                draw_coordinate_frame(ax, np.eye(4), 'GT', length=0.05)
                # Draw predicted pose coordinate frame (transformed to GT frame)
                pred_in_gt = np.linalg.inv(gt_pose) @ pred_pose
                draw_coordinate_frame(ax, pred_in_gt, 'Pred', length=0.05) 
                ax.set_title(f'Predicted (in GT frame) vs GT Pose (View {i+1})')
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                ax.set_box_aspect([1,1,1])
                # Autoscale to fit all meshes
                all_verts = np.vstack([pred_pts, gt_pts])
                min_xyz = all_verts.min(axis=0)
                max_xyz = all_verts.max(axis=0)
                ax.set_xlim(min_xyz[0], max_xyz[0])
                ax.set_ylim(min_xyz[1], max_xyz[1])
                ax.set_zlim(min_xyz[2], max_xyz[2])
                ax.view_init(elev=elev, azim=azim)
                plt.tight_layout()
                plt.savefig(os.path.join(save_path, f'{obj}_pred_in_gt_frame_view{i+1}.png'))
                plt.close()
                logging.info(f"Saved mesh-in-GT-frame visualization for view {i+1}.")

        results.append({
            'Object': obj,
            'Object_Number': obj_num,
            'Chamfer_Distance': float(chamfer_dis)
            })
            # Save results to diff_light_result.txt
        result_file = 'diff_light_result.txt'
        # 'a' mode will create the file if it doesn't exist
        with open(result_file, 'a') as f:
            f.write(f"{obj}\t{obj_num}\t{chamfer_dis}\n")
        logging.info(f"Appended results to {result_file}.")

    logging.info("Demo script finished.")
