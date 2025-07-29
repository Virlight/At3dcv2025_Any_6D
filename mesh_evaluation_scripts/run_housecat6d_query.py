import copy

from foundationpose.datareader import Ho3dReader,HouseCat6DReader
from estimater import *
from bop_toolkit_lib.pose_error_custom import mssd, mspd, vsd

from metrics import *
import json
# from bop_toolkit_lib.renderer_vispy import RendererVispy
from pytorch_lightning import seed_everything
from datetime import datetime
import logging

# run anchor folder one in a time

# to-do
# find ob_id for model_info rendering
# 1. add mssd and mspd
# 2. add vsd
# 3. add mssd and mspd error
# 4. add vsd error

# fininshed
# 1. ob_id is to know thich object to render

if __name__ == '__main__':

    seed_everything(0)

    parser = argparse.ArgumentParser(description="Set experiment name and paths")

    parser.add_argument("--name", type=str, default="any6d", help="Experiment name")
    # anchor(wanna see the effect of different ref)
    parser.add_argument("--anchor_path", type=str, default="dexycb_reference_view_ours", help="Path to the YCB-V model info JSON")
    # query
    parser.add_argument("--query_path", type=str, default="/mnt/home/projects/6D_pose/dataset/ho3d", help="Path to the HO3D dataset root")
    # for mssd and mspd error
    parser.add_argument("--ycbv_modesl_info_path", type=str, default="./models_info.json", help="Path to the YCB-V model info JSON")
    parser.add_argument("--running_stride", type=int, default=10, help="Running stride")

    args = parser.parse_args()

    name = args.name
    query_path = args.query_path
    ycbv_modesl_info_path = args.ycbv_modesl_info_path
    running_stride = args.running_stride
    anchor_path = args.anchor_path
    
    date_str = f'{datetime.now():%Y-%m-%d_%H-%M-%S}'
    save_root = f"./results/housecat6d_query_results/{name}/{date_str}"
    save_results_est_path = f'{save_root}'

    os.makedirs(save_results_est_path, exist_ok=True)

    # obj_folder is for query
    # Parse anchor folder to get object names and reference view numbers
    # Parse anchor_path to get object name and reference view number (last two parts of the path)
    anchor_parts = os.path.normpath(anchor_path).split(os.sep)
    if len(anchor_parts) >= 2 and anchor_parts[-1].isdigit() and len(anchor_parts[-1]) == 6:
        obj_name = f"{anchor_parts[-2]}_{anchor_parts[-1]}"
    else:
        raise ValueError(f"anchor_path '{anchor_path}' does not end with 'objname/6digit'")
    

    object_metrics = {obj_name: {
        'ADD': [], 
        'ADD-S': [], 
        # 'AR': [], 
        # 'MSSD': [], 
        # 'MSPD': [],
        'VSD': [], 
        'R error': [], 
        'T error': [], 
        'cls_id': [], 
        'instance_id': []
        } }
    all_frame_data = {
        'Frame_ID': [],
        'Class': [],
        'ADD-S': [],
        'ADD': [],
        # 'AR': [],
        # 'MSSD': [],
        # 'MSPD': [],
        'VSD': [],
        'R_error': [],
        'T_error': [],
        }

    excel_files = []
    glctx = dr.RasterizeCudaContext()
    mesh_tmp = copy.deepcopy(trimesh.primitives.Box(extents=np.ones((3)), transform=np.eye(4)))
    mesh = trimesh.Trimesh(vertices=mesh_tmp.vertices.copy(), faces= mesh_tmp.faces.copy())
    est = Any6D(mesh=mesh, scorer=ScorePredictor(), refiner=PoseRefinePredictor(), debug_dir=save_results_est_path, debug=0, glctx=glctx)

    # renderer = RendererVispy(640, 480, mode='depth')
    obj_count = 0

    data = []


    # Since there is only one object, use obj_name directly (obj_name is a string like "cat_000001")
    obj_f = obj_name
    for _ in [obj_f]:  # keep the loop structure for compatibility

        # (check)
        # the query videos is the same for objects in same scene
        video_dir = os.path.join(f"{query_path}")
        reader = HouseCat6DReader(video_dir, query_path)
        reader.color_files = reader.color_files[::running_stride]

        # # ob_id is for finding the model info
        # ob_id = reader.get_obj_id()

        # # get bop information
        # with open(ycbv_modesl_info_path, 'r') as f:
        #     model_info = json.load(f)
        # trans_disc = [{"R": np.eye(3), "t": np.array([[0, 0, 0]]).T}]  # Identity.
        # if "symmetries_discrete" in model_info[f"{ob_id}"]:
        #     for sym in model_info[f"{ob_id}"]["symmetries_discrete"]:
        #         sym_4x4 = np.reshape(sym, (4, 4))
        #         R = sym_4x4[:3, :3]
        #         t = sym_4x4[:3, 3].reshape((3, 1))
        #         trans_disc.append({"R": R, "t": t})

        K_anchor = np.loadtxt(reader.get_reference_K(anchor_path))

        gt_mesh = reader.get_gt_mesh(anchor_path)
        gt_diameter = reader.get_gt_mesh_diamter(anchor_path)
        
        mesh = trimesh.load(reader.get_reference_view_1_mesh(anchor_path))

        gt_mesh_dict = {
            'pts': np.asarray(gt_mesh.vertices) * 1e3,
            'normals': np.asarray(gt_mesh.face_normals),
            'faces': np.asarray(gt_mesh.faces),
            }
        # renderer.my_add_object(gt_mesh_dict, ob_id)
        pred_pose_path = reader.get_reference_view_1_pose(anchor_path)
        pred_pose_a = np.loadtxt(pred_pose_path)
        gt_pose_a = np.loadtxt(pred_pose_path.replace('initial','gt'))


        est.reset_object(mesh=mesh, symmetry_tfs=None)

        # for i in tqdm(range(0, 1 , 1), desc=f"{obj_f} - Frames"):
        for i in tqdm(range(0, len(reader.color_files), 1), desc=f"{obj_f} - Frames"):
            logging.info(f"Processing frame {i} for object {obj_f}")
            gt_pose_q = reader.get_gt_pose(i,obj_f)

            if gt_pose_q is None:
                continue

            color_file = reader.color_files[i]
            color = cv2.cvtColor(cv2.imread(color_file), cv2.COLOR_BGR2RGB)
            H, W = color.shape[:2]
            depth = reader.get_depth(i)
            mask = reader.get_mask(i, obj_f).astype(np.bool_)
            
            # Save the mask as an image for debugging/visualization
            mask_save_path = os.path.join(save_results_est_path, f"mask_{obj_f}.png")
            cv2.imwrite(mask_save_path, (mask.astype(np.uint8) * 255))

# find the pose m->q
            pred_pose_q = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=mask, iteration=5, name=obj_f)
# obtain pose A->Q by pose m->A and pose m->Q
            pose_aq = pred_pose_q @ np.linalg.inv(pred_pose_a)  # obtained pose A->Q
            pred_q = pose_aq @ gt_pose_a

            err_R, err_T = compute_RT_distances(pred_q, gt_pose_q)

            pose_recall_th = [(5, 5), (5, 10), (10, 10)]

            for r_th, t_th in pose_recall_th:
                succ_r, succ_t = err_R <= r_th, err_T <= t_th
                succ_pose = np.logical_and(succ_r, succ_t).astype(float)

            add = compute_add(gt_mesh.vertices, pred_q, gt_pose_q)
            adds = compute_adds(gt_mesh.vertices, pred_q, gt_pose_q)

            add_thres = float(add <= gt_diameter * 0.1)
            adds_thres = float(adds <= gt_diameter * 0.1)

            pred_q, gt_pose_q = pred_q.astype(np.float16), gt_pose_q.astype(np.float16)

            pred_r, pred_t = pred_q[:3, :3], np.expand_dims(pred_q[:3, 3], axis=1) * 1e3
            gt_r, gt_t = gt_pose_q[:3, :3], np.expand_dims(gt_pose_q[:3, 3], axis=1) * 1e3

            # mssd_err = mssd(pose_est=pred_q, pose_gt=gt_pose_q, pts=gt_mesh.vertices, syms=trans_disc) * 1e3
            # mspd_err = mspd(pose_est=pred_q, pose_gt=gt_pose_q, pts=gt_mesh.vertices, K=reader.K, syms=trans_disc)

            # mssd_rec = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
            # mspd_rec = np.array([5, 10, 15, 20, 25, 30, 35, 40, 45, 50])

            vsd_delta = 15.0
            vsd_taus = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
            vsd_rec = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])

            # vsd_errs = vsd(pred_r, pred_t, gt_r, gt_t, (depth *1e3), reader.K.reshape(3, 3), vsd_delta, vsd_taus, True, (gt_diameter*1e3), renderer, ob_id)
            # vsd_errs = np.asarray(vsd_errs)
            # all_vsd_recs = np.stack([vsd_errs < rec_i for rec_i in vsd_rec], axis=1)
            # mean_vsd = all_vsd_recs.mean()

            # mssd_cur_rec = mssd_rec * (gt_diameter * 1e3)
            # mean_mssd = (mssd_err < mssd_cur_rec).mean()
            # mean_mspd = (mspd_err < mspd_rec).mean()

            # mean_ar = (mean_mssd + mean_mspd + mean_vsd) / 3.

            object_metrics[obj_f]['ADD'].append(add_thres)
            object_metrics[obj_f]['ADD-S'].append(adds_thres)
            # object_metrics[obj_f]['AR'].append(mean_ar)
            object_metrics[obj_f]['VSD'].append(0)  # mean_vsd placeholder
            # object_metrics[obj_f]['MSSD'].append(mean_mssd)
            # object_metrics[obj_f]['MSPD'].append(mean_mspd)
            object_metrics[obj_f]['R error'].append(err_R.tolist()[0])
            object_metrics[obj_f]['T error'].append(err_T.tolist()[0])
            object_metrics[obj_f]['cls_id'].append(obj_f)
            object_metrics[obj_f]['instance_id'].append(obj_count)

            print(gt_pose_q)
            print(pred_pose_q)


            try:
                visualize_frame_results_gt(color=color, gt_mesh=gt_mesh, K=reader.K, gt_pose=gt_pose_q, pred_pose=pred_pose_q, metric=object_metrics[obj_f], obj_f=f"{obj_f}", frame_idx=i, save_path=save_results_est_path, glctx=glctx, name=f"{len(reader.color_files)}_{name}",nocs_metric=True, est_mesh=est.mesh)
            except:
                pass
            obj_count+=1

        df_obj = pd.DataFrame({
            'Frame_ID': object_metrics[obj_f]['instance_id'],
            'Class': object_metrics[obj_f]['cls_id'],
            'ADD-S': object_metrics[obj_f]['ADD-S'],
            'ADD': object_metrics[obj_f]['ADD'],
            # 'AR': object_metrics[obj_f]['AR'],
            # 'MSSD': object_metrics[obj_f]['MSSD'],
            # 'MSPD': object_metrics[obj_f]['MSPD'],
            'VSD': object_metrics[obj_f]['VSD'],
            'R_error': object_metrics[obj_f]['R error'],
            'T_error': object_metrics[obj_f]['T error'],
            })

        means_all = {
            'ADD-S': np.mean(object_metrics[obj_f]['ADD-S']) * 100,
            'ADD': np.mean(object_metrics[obj_f]['ADD']) * 100,
            # 'AR': np.mean(object_metrics[obj_f]['AR']) * 100,
            # 'MSSD': np.mean(object_metrics[obj_f]['MSSD']) * 100,
            # 'MSPD': np.mean(object_metrics[obj_f]['MSPD']) * 100,
            'VSD': np.mean(object_metrics[obj_f]['VSD']) * 100,
            'R_error': np.mean(object_metrics[obj_f]['R error']),
            'T_error': np.mean(object_metrics[obj_f]['T error'])
            }

        mean_row_df = pd.DataFrame({
            'Frame_ID': ['MEAN'],
            'Class': [obj_f],
            'ADD-S': [f"{means_all['ADD-S']:.1f}"],
            'ADD': [f"{means_all['ADD']:.1f}"],
            # 'AR': [f"{means_all['AR']:.1f}"],
            # 'MSSD': [f"{means_all['MSSD']:.1f}"],
            # 'MSPD': [f"{means_all['MSPD']:.1f}"],
            'VSD': [f"{means_all['VSD']:.1f}"],
            'R_error': [f"{means_all['R_error']:.1f}"],
            'T_error': [f"{means_all['T_error']:.1f}"]
            })

        df_obj = pd.concat([df_obj, mean_row_df], ignore_index=True)

        row_data = {
            'Class_ID': obj_f,
            'ADD-S': f"{means_all['ADD-S']:.1f}",
            'ADD': f"{means_all['ADD']:.1f}",
            # 'AR': f"{means_all['AR']:.1f}",
            # 'MSSD': f"{means_all['MSSD']:.1f}",
            # 'MSPD': f"{means_all['MSPD']:.1f}",
            'VSD': f"{means_all['VSD']:.1f}",
            }

        data.append(row_data)

        df_obj.to_excel(f'{save_results_est_path}/{obj_f}_metrics_results.xlsx', index=False)
        all_frame_data['Frame_ID'].extend(object_metrics[obj_f]['instance_id'])
        all_frame_data['Class'].extend(object_metrics[obj_f]['cls_id'])
        all_frame_data['ADD-S'].extend(object_metrics[obj_f]['ADD-S'])
        all_frame_data['ADD'].extend(object_metrics[obj_f]['ADD'])
        # all_frame_data['AR'].extend(object_metrics[obj_f]['AR'])
        # all_frame_data['MSSD'].extend(object_metrics[obj_f]['MSSD'])
        # all_frame_data['MSPD'].extend(object_metrics[obj_f]['MSPD'])
        all_frame_data['VSD'].extend(object_metrics[obj_f]['VSD'])
        all_frame_data['R_error'].extend(object_metrics[obj_f]['R error'])
        all_frame_data['T_error'].extend(object_metrics[obj_f]['T error'])

    overall_means = {
        'ADD-S': np.mean(object_metrics[obj_f]['ADD-S']) * 100,
        'ADD': np.mean(object_metrics[obj_f]['ADD']) * 100,
        # 'AR': np.mean(object_metrics[obj_f]['AR']) * 100,
        # 'MSSD': np.mean(object_metrics[obj_f]['MSSD']) * 100,
        # 'MSPD': np.mean(object_metrics[obj_f]['MSPD']) * 100,
        'VSD': np.mean(object_metrics[obj_f]['VSD']) * 100,
        }

    mean_row = {
        'Class_ID': 'MEAN',
        'ADD-S': f"{overall_means['ADD-S']:.1f}",
        'ADD': f"{overall_means['ADD']:.1f}",
        # 'AR': f"{overall_means['AR']:.1f}",
        # 'MSSD': f"{overall_means['MSSD']:.1f}",
        # 'MSPD': f"{overall_means['MSPD']:.1f}",
        'VSD': f"{overall_means['VSD']:.1f}",
        }
    data.append(mean_row)

    df = pd.DataFrame(data)
    df.to_excel(f'{save_results_est_path}/0_mean_all_metrics_classes_results.xlsx', index=False)

    df_all_frames = pd.DataFrame(all_frame_data)

    means_all = {
        'Frame_ID': 'MEAN',
        'Class': 'ALL',
        'ADD-S': f"{df_all_frames['ADD-S'].mean() * 100:.1f}",
        'ADD': f"{df_all_frames['ADD'].mean() * 100:.1f}",
        # 'AR': f"{df_all_frames['AR'].mean() * 100:.1f}",
        # 'MSSD': f"{df_all_frames['MSSD'].mean() * 100:.1f}",
        # 'MSPD': f"{df_all_frames['MSPD'].mean() * 100:.1f}",
        'VSD': f"{df_all_frames['VSD'].mean() * 100:.1f}",
        'R_error': f"{df_all_frames['R_error'].mean():.1f}",
        'T_error': f"{df_all_frames['T_error'].mean():.1f}",
        }

    df_all_frames = pd.concat([df_all_frames, pd.DataFrame([means_all])], ignore_index=True)

    output_path = f'{save_results_est_path}/0_all_frames_metrics_results.xlsx'
    df_all_frames.to_excel(output_path, index=False)
    print(f"\nAll frames metrics saved to {output_path}")

    print("\nSaved data preview:")
    print(df_all_frames)
