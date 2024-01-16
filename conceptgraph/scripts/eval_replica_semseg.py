import gzip
import os
import glob
from pathlib import Path
import argparse
import pickle

import numpy as np
import open3d as o3d
import pandas as pd

import torch

import open_clip

from chamferdist.chamfer import knn_points
from gradslam.structures.pointclouds import Pointclouds

from conceptgraph.dataset.replica_constants import (
    REPLICA_EXISTING_CLASSES, REPLICA_CLASSES,
    REPLICA_SCENE_IDS, REPLICA_SCENE_IDS_,
)
from conceptgraph.slam.slam_classes import MapObjectList
from conceptgraph.utils.vis import get_random_colors
from conceptgraph.utils.eval import compute_confmatrix, compute_pred_gt_associations, compute_metrics


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--replica_root", type=Path, default=Path("~/rdata/Replica/").expanduser()
    )
    parser.add_argument(
        "--replica_semantic_root",
        type=Path,
        default=Path("~/rdata/Replica-semantic/").expanduser()
    )
    parser.add_argument(
        "--pred_exp_name", 
        type=str,
        default="ram_withbg_allclasses_overlap_maskconf0.25_simsum1.2_dbscan.1_masksub",
        help="The name of cfslam experiment. Will be used to load the result. "
    )
    parser.add_argument(
        "--n_exclude", type=int, default=1, choices=[1, 4, 6],
        help='''Number of classes to exclude:
        1: exclude "other"
        4: exclude "other", "floor", "wall", "ceiling"
        6: exclude "other", "floor", "wall", "ceiling", "door", "window"
        ''',
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0"
    )
    return parser

def eval_replica(
    scene_id: str,
    scene_id_: str,
    class_names: list[str],
    class_feats: torch.Tensor,
    args: argparse.Namespace,
    class_all2existing: torch.Tensor,
    ignore_index=[],
    gt_class_only: bool = True, # only compute the conf matrix for the GT classes
):
    class2color = get_random_colors(len(class_names))

    '''Load the GT point cloud'''
    gt_pc_path = os.path.join(
        args.replica_semantic_root, scene_id_, "Sequence_1", "saved-maps-gt"
    )
    gt_pose_path = os.path.join(
        args.replica_semantic_root, scene_id_, "Sequence_1", "traj_w_c.txt"
    )
    
    gt_map = Pointclouds.load_pointcloud_from_h5(gt_pc_path)
    gt_poses = np.loadtxt(gt_pose_path)
    gt_poses = torch.from_numpy(gt_poses.reshape(-1, 4, 4)).float()

    gt_xyz = gt_map.points_padded[0]
    gt_color = gt_map.colors_padded[0]
    gt_embedding = gt_map.embeddings_padded[0]  # (N, num_class)
    gt_class = gt_embedding.argmax(dim=1)  # (N,)
    gt_class = class_all2existing[gt_class]  # (N,)
    assert gt_class.min() >= 0
    assert gt_class.max() < len(REPLICA_EXISTING_CLASSES)

    # transform pred_xyz and gt_xyz according to the first pose in gt_poses
    gt_xyz = gt_xyz @ gt_poses[0, :3, :3].t() + gt_poses[0, :3, 3]
    
    # Get the set of classes that are used for evaluation
    all_class_index = np.arange(len(class_names))
    ignore_index = np.asarray(ignore_index)
    if gt_class_only:
        # Only consider the classes that exist in the current scene
        existing_index = gt_class.unique().cpu().numpy()
        non_existing_index = np.setdiff1d(all_class_index, existing_index)
        ignore_index = np.append(ignore_index, non_existing_index)
        print(
            "Using only the classes that exists in GT of this scene: ",
            len(existing_index),
        )

    keep_index = np.setdiff1d(all_class_index, ignore_index)

    print(
        f"{len(keep_index)} classes remains. They are: ",
        [(i, class_names[i]) for i in keep_index],
    )
    
    '''Load the predicted point cloud'''
    result_paths = glob.glob(
        os.path.join(
            args.replica_root, scene_id, "pcd_saves", 
            f"full_pcd_{args.pred_exp_name}*.pkl.gz"
        )
    )
    if len(result_paths) == 0:
        raise ValueError(f"No result found for {scene_id} with {args.pred_exp_name}")
        
    # Get the newest result over result_paths
    result_paths = sorted(result_paths, key=os.path.getmtime)
    result_path = result_paths[-1]
    print(f"Loading mapping result from {result_path}")
    
    with gzip.open(result_path, "rb") as f:
            results = pickle.load(f)
        
    objects = MapObjectList()
    objects.load_serializable(results['objects'])

    # Compute the CLIP similarity for the mapped objects and assign class to them
    object_feats = objects.get_stacked_values_torch("clip_ft").to(args.device)
    object_feats = object_feats / object_feats.norm(dim=-1, keepdim=True) # (num_objects, D)
    object_class_sim = object_feats @ class_feats.T # (num_objects, num_classes)
    
    # suppress the logits to -inf that are not in torch.from_numpy(keep_class_index)
    object_class_sim[:, ignore_index] = -1e10
    object_class = object_class_sim.argmax(dim=-1) # (num_objects,)
    
    if args.n_exclude == 1:
        if results['bg_objects'] is None:
            print("Warning: no background objects found. This is expected if only SAM is used, but not the detector. ")
        else:
            # Also add the background objects
            bg_objects = MapObjectList()
            bg_objects.load_serializable(results['bg_objects'])
            
            # Assign class to the background objects (hard assignment)
            for obj in bg_objects:
                cn = obj['class_name'][0]
                c = class_names.index(cn.lower())
                object_class = torch.cat([object_class, object_class.new_full([1], c)])
                
            objects += bg_objects
    
    pred_xyz = []
    pred_color = []
    pred_class = []
    for i in range(len(objects)):
        obj_pcd = objects[i]['pcd']
        pred_xyz.append(np.asarray(obj_pcd.points))
        pred_color.append(np.asarray(obj_pcd.colors))
        pred_class.append(np.ones(len(obj_pcd.points)) * object_class[i].item())
        
    pred_xyz = torch.from_numpy(np.concatenate(pred_xyz, axis=0))
    pred_color = torch.from_numpy(np.concatenate(pred_color, axis=0))
    pred_class = torch.from_numpy(np.concatenate(pred_class, axis=0)).long()
    
    '''Load the SLAM reconstruction results, to ensure fair comparison'''
    slam_path = os.path.join(
        args.replica_root, scene_id, "rgb_cloud"
    )
    slam_pointclouds = Pointclouds.load_pointcloud_from_h5(slam_path)
    slam_xyz = slam_pointclouds.points_padded[0]
    
    # To ensure fair comparison, build the prediction point cloud based on the slam results
    # Search for NN of slam_xyz in pred_xyz
    slam_nn_in_pred = knn_points(
        slam_xyz.unsqueeze(0).cuda().contiguous().float(),
        pred_xyz.unsqueeze(0).cuda().contiguous().float(),
        lengths1=None,
        lengths2=None,
        return_nn=True,
        return_sorted=True,
        K=1,
    )
    idx_slam_to_pred = slam_nn_in_pred.idx.squeeze(0).squeeze(-1)
    
    # # predicted point cloud in open3d
    # print("Before resampling")
    # pred_pcd = o3d.geometry.PointCloud()
    # pred_pcd.points = o3d.utility.Vector3dVector(pred_xyz.numpy())
    # pred_pcd.colors = o3d.utility.Vector3dVector(class2color[pred_class.numpy()])
    # o3d.visualization.draw_geometries([pred_pcd])
    
    # Resample the pred_xyz and pred_class based on slam_nn_in_pred
    pred_xyz = slam_xyz
    pred_class = pred_class[idx_slam_to_pred.cpu()]
    pred_color = pred_color[idx_slam_to_pred.cpu()]
    
    # # predicted point cloud in open3d
    # print("After resampling")
    # pred_pcd = o3d.geometry.PointCloud()
    # pred_pcd.points = o3d.utility.Vector3dVector(pred_xyz.numpy())
    # pred_pcd.colors = o3d.utility.Vector3dVector(class2color[pred_class.numpy()])
    # o3d.visualization.draw_geometries([pred_pcd])
    
    # Compute the associations between the predicted and ground truth point clouds
    idx_pred_to_gt, idx_gt_to_pred = compute_pred_gt_associations(
        pred_xyz.unsqueeze(0).cuda().contiguous().float(),
        gt_xyz.unsqueeze(0).cuda().contiguous().float(),
    )
    
    # Only keep the points on the 3D reconstructions that are mapped to
    # GT point that is in keep_index
    label_gt = gt_class[idx_pred_to_gt.cpu()]
    pred_keep_idx = torch.isin(label_gt, torch.from_numpy(keep_index))
    pred_class = pred_class[pred_keep_idx]
    idx_pred_to_gt = idx_pred_to_gt[pred_keep_idx]
    idx_gt_to_pred = None  # not to be used
    
    # Compute the confusion matrix
    confmatrix = compute_confmatrix(
        pred_class.cuda(),
        gt_class.cuda(),
        idx_pred_to_gt,
        idx_gt_to_pred,
        class_names,
    )
    
    assert confmatrix.sum(0)[ignore_index].sum() == 0
    assert confmatrix.sum(1)[ignore_index].sum() == 0
    
    '''Visualization for debugging'''
    # class2color = get_random_colors(len(class_names))

    # # GT point cloud in open3d
    # gt_pcd = gt_map.open3d(0)
    # gt_pcd.transform(gt_poses[0].numpy())
    # gt_pcd.colors = o3d.utility.Vector3dVector(class2color[gt_class])
    
    # # predicted point cloud in open3d
    # pred_pcd = o3d.geometry.PointCloud()
    # pred_pcd.points = o3d.utility.Vector3dVector(pred_xyz.numpy())
    # pred_pcd.colors = o3d.utility.Vector3dVector(class2color[pred_class.numpy()])

    # o3d.visualization.draw_geometries([pred_pcd])
    # o3d.visualization.draw_geometries([gt_pcd])
    
    return confmatrix, keep_index
    

def main(args: argparse.Namespace):

    # map REPLICA_CLASSES to REPLICA_EXISTING_CLASSES
    class_all2existing = torch.ones(len(REPLICA_CLASSES)).long() * -1
    for i, c in enumerate(REPLICA_EXISTING_CLASSES):
        class_all2existing[c] = i
    class_names = [REPLICA_CLASSES[i] for i in REPLICA_EXISTING_CLASSES]
    
    if args.n_exclude == 1:
        exclude_class = [class_names.index(c) for c in [
            "other"
        ]]
    elif args.n_exclude == 4:
        exclude_class = [class_names.index(c) for c in [
            "other", "floor", "wall", "ceiling"
        ]]
    elif args.n_exclude == 6:
        exclude_class = [class_names.index(c) for c in [
            "other", "floor", "wall", "ceiling", "door", "window"
        ]]
    else:
        raise ValueError("Invalid n_exclude: %d" % args.n_exclude)
    
    print("Excluding classes: ", [(i, class_names[i]) for i in exclude_class])

    # Compute the CLIP embedding for each class
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms("ViT-H-14", "laion2b_s32b_b79k")
    clip_model = clip_model.to(args.device)
    clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")
    prompts = [f"an image of {c}" for c in class_names]
    text = clip_tokenizer(prompts)
    text = text.to(args.device)
    class_feats = clip_model.encode_text(text)
    class_feats /= class_feats.norm(dim=-1, keepdim=True) # (num_classes, D)

    conf_matrices = {}
    conf_matrix_all = 0
    for scene_id, scene_id_ in zip(REPLICA_SCENE_IDS, REPLICA_SCENE_IDS_):
        print("Evaluating on:", scene_id, scene_id_)
        conf_matrix, keep_index = eval_replica(
            scene_id = scene_id,
            scene_id_ = scene_id_,
            class_names = class_names,
            class_feats = class_feats,
            args = args,
            class_all2existing = class_all2existing,
            ignore_index = exclude_class,
        )
        
        conf_matrix = conf_matrix.detach().cpu()
        conf_matrix_all += conf_matrix

        conf_matrices[scene_id] = {
            "conf_matrix": conf_matrix,
            "keep_index": keep_index,
        }
        
    # Remove the rows and columns that are not in keep_class_index
    conf_matrices["all"] = {
        "conf_matrix": conf_matrix_all,
        "keep_index": conf_matrix_all.sum(axis=1).nonzero().reshape(-1),
    }
    
    results = []
    for scene_id, res in conf_matrices.items():
        conf_matrix = res["conf_matrix"]
        keep_index = res["keep_index"]
        conf_matrix = conf_matrix[keep_index, :][:, keep_index]
        keep_class_names = [class_names[i] for i in keep_index]

        mdict = compute_metrics(conf_matrix, keep_class_names)
        results.append(
            {
                "scene_id": scene_id,
                "miou": mdict["miou"] * 100.0,
                "mrecall": np.mean(mdict["recall"]) * 100.0,
                "mprecision": np.mean(mdict["precision"]) * 100.0,
                "mf1score": np.mean(mdict["f1score"]) * 100.0,
                "fmiou": mdict["fmiou"] * 100.0,
            }
        )
        
    df_result = pd.DataFrame(results)
    
    save_path = "./results/%s/replica_ex%d_results.csv" % (
        args.pred_exp_name, args.n_exclude
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df_result.to_csv(save_path, index=False)

    # Also save the conf_matrices
    save_path = "./results/%s/replica_ex%d_conf_matrices.pkl" % (
        args.pred_exp_name, args.n_exclude
    )
    pickle.dump(conf_matrices, open(save_path, "wb"))


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)