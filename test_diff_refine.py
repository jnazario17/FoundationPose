import numpy as np
import trimesh
import cv2
import torch
import nvdiffrast.torch as dr
from estimater import FoundationPose
from Utils import add_err, compute_mesh_diameter, make_mesh_tensors, nvdiffrast_render
from augmentations import apply_occlusion, apply_blur, apply_combined
from diff_renderer import DifferentiablePoseOptimizer

data_dir = 'demo_data/kinect_driller_seq'
K = np.loadtxt(f'{data_dir}/cam_K.txt')
mesh = trimesh.load(f'{data_dir}/mesh/textured_mesh.obj')
diam = compute_mesh_diameter(mesh.vertices)
mesh_tensors = make_mesh_tensors(mesh)
glctx = dr.RasterizeCudaContext()

est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, glctx=glctx)
diff_opt = DifferentiablePoseOptimizer(glctx=glctx, lr_rot=0.008, lr_trans=0.003, num_iters=30)

print(f"Mesh loaded, diameter: {diam:.4f} m, 0.1d threshold: {diam*10:.2f} cm")

for frame_idx in [1, 45, 75, 90, 105]:
    gt_pose = np.loadtxt(f'{data_dir}/ob_in_cam/{frame_idx:07d}.txt')
    rgb = cv2.imread(f'{data_dir}/rgb/{frame_idx:07d}.png')[..., ::-1].copy()
    depth = (cv2.imread(f'{data_dir}/depth/{frame_idx:07d}.png', -1) / 1000.0).astype(np.float32)
    H, W = rgb.shape[:2]

    _, depth_r, _ = nvdiffrast_render(K=K, H=H, W=W, ob_in_cams=torch.tensor(gt_pose, device='cuda', dtype=torch.float)[None], mesh_tensors=mesh_tensors, glctx=glctx)
    gt_mask = (depth_r[0] > 0).cpu().numpy().astype(np.uint8)

    seed = frame_idx * 100 + 7
    rgb_occ, depth_occ, occ_mask = apply_occlusion(rgb, depth, gt_mask, occlusion_ratio=0.40, seed=seed)
    pred_occ = est.register(K=K, rgb=rgb_occ, depth=depth_occ, ob_mask=gt_mask, iteration=5)
    add_occ = add_err(pred_occ, gt_pose, mesh.vertices) * 100

    pred_diff = diff_opt.optimize_pose(
        init_pose=pred_occ,
        mesh_tensors=mesh_tensors,
        K=K,
        H=H,
        W=W,
        obs_rgb=rgb_occ,
        obs_depth=depth_occ,
        obs_mask=gt_mask,
        occ_mask=occ_mask,
        w_depth=1.0,
        w_sil=0.5,
        w_rgb=0.0
    )
    add_diff = add_err(pred_diff, gt_pose, mesh.vertices) * 100
    print(f"Frame {frame_idx:3d}: Occluded ADD = {add_occ:.2f} cm -> WITH Diff = {add_diff:.2f} cm (diff: {add_occ - add_diff:+.2f} cm)")
