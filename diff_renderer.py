import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import nvdiffrast.torch as dr
from Utils import to_homo_torch, glcam_in_cvcam, projection_matrix_from_intrinsics


def rodrigues_to_rotation_matrix(rot_vec):
    """Convert an axis-angle rotation vector (3,) or (B, 3) to rotation matrix using Rodrigues formula with smooth gradient at zero."""
    if rot_vec.ndim == 1:
        rot_vec = rot_vec.unsqueeze(0)
    theta_sq = torch.sum(rot_vec ** 2, dim=-1, keepdim=True)  # (B, 1)
    theta = torch.sqrt(theta_sq + 1e-12)
    
    A = torch.sin(theta) / theta
    B = (1.0 - torch.cos(theta)) / (theta_sq + 1e-12)
    
    wx, wy, wz = rot_vec[:, 0], rot_vec[:, 1], rot_vec[:, 2]
    zeros = torch.zeros_like(wx)
    K = torch.stack([
        zeros, -wz, wy,
        wz, zeros, -wx,
        -wy, wx, zeros
    ], dim=-1).reshape(-1, 3, 3)
    
    I = torch.eye(3, dtype=rot_vec.dtype, device=rot_vec.device).unsqueeze(0).expand(rot_vec.shape[0], 3, 3)
    R = I + A.unsqueeze(-1) * K + B.unsqueeze(-1) * torch.bmm(K, K)
    return R.squeeze(0) if R.shape[0] == 1 else R


class DifferentiablePoseOptimizer:
    """
    Refines 6D pose via test-time Differentiable Rendering.
    Uses nvdiffrast to differentiate depth, silhouette mask, and photometric objectives
    directly with respect to 6-DoF pose parameters (Lie algebra so(3) rotation + translation).
    """
    def __init__(self, glctx=None, lr_rot=0.008, lr_trans=0.003, num_iters=25, device='cuda:0'):
        self.device = torch.device(device)
        self.glctx = glctx if glctx is not None else dr.RasterizeCudaContext(device=self.device)
        self.lr_rot = lr_rot
        self.lr_trans = lr_trans
        self.num_iters = num_iters

    def optimize_pose(
        self,
        init_pose,
        mesh_tensors,
        K,
        H,
        W,
        obs_rgb=None,
        obs_depth=None,
        obs_mask=None,
        occ_mask=None,
        w_depth=1.0,
        w_sil=0.5,
        w_rgb=0.1
    ):
        """
        init_pose: (4, 4) numpy array or torch tensor (camera frame)
        mesh_tensors: dictionary containing 'pos' (N, 3), 'faces' (M, 3)
        K: (3, 3) camera intrinsics
        H, W: image height and width
        obs_rgb: (H, W, 3) float [0, 1] (optional)
        obs_depth: (H, W) float in meters (optional)
        obs_mask: (H, W) binary mask [0, 1]
        occ_mask: (H, W) binary mask of occluders to ignore (optional)
        """
        if isinstance(init_pose, np.ndarray):
            init_pose_t = torch.tensor(init_pose, dtype=torch.float32, device=self.device)
        else:
            init_pose_t = init_pose.clone().to(dtype=torch.float32, device=self.device)

        R_init = init_pose_t[:3, :3].clone()
        t_init = init_pose_t[:3, 3].clone()

        # Learnable perturbations: so(3) rotation vector and translation delta
        rot_vec = nn.Parameter(torch.zeros(3, dtype=torch.float32, device=self.device, requires_grad=True))
        trans_delta = nn.Parameter(torch.zeros(3, dtype=torch.float32, device=self.device, requires_grad=True))

        optimizer = torch.optim.Adam([
            {'params': [rot_vec], 'lr': self.lr_rot},
            {'params': [trans_delta], 'lr': self.lr_trans}
        ])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.num_iters, eta_min=1e-5)

        # Prepare mesh tensors
        pos = mesh_tensors['pos'].to(self.device, dtype=torch.float32)
        faces = mesh_tensors['faces'].to(self.device, dtype=torch.int32).contiguous()
        has_tex = 'tex' in mesh_tensors and mesh_tensors['tex'] is not None

        # Camera projection matrix
        proj_mat_np = projection_matrix_from_intrinsics(K, height=H, width=W, znear=0.01, zfar=10.0)
        proj_mat = torch.tensor(proj_mat_np, dtype=torch.float32, device=self.device).reshape(4, 4)
        glcam_mat = torch.tensor(glcam_in_cvcam, dtype=torch.float32, device=self.device)

        # Prepare observation tensors
        if obs_depth is not None:
            depth_gt = torch.as_tensor(obs_depth, dtype=torch.float32, device=self.device)
            valid_depth_gt = (depth_gt > 0.05) & (depth_gt < 5.0)
        else:
            depth_gt, valid_depth_gt = None, None

        if obs_mask is not None:
            mask_gt = torch.as_tensor(obs_mask, dtype=torch.float32, device=self.device)
        else:
            mask_gt = None

        if occ_mask is not None:
            unoccluded = torch.as_tensor(occ_mask == 0, dtype=torch.bool, device=self.device)
        else:
            unoccluded = torch.ones((H, W), dtype=torch.bool, device=self.device)

        if obs_rgb is not None and w_rgb > 0:
            rgb_gt = torch.as_tensor(obs_rgb, dtype=torch.float32, device=self.device)
            if rgb_gt.max() > 1.0:
                rgb_gt = rgb_gt / 255.0
        else:
            rgb_gt = None

        best_loss = float('inf')
        best_R = R_init.clone()
        best_t = t_init.clone()

        for step in range(self.num_iters):
            optimizer.zero_grad()

            # Compute current rotation and translation
            delta_R = rodrigues_to_rotation_matrix(rot_vec)
            R_curr = delta_R @ R_init
            t_curr = t_init + trans_delta

            # Transform mesh vertices to camera coordinates
            pts_cam = (pos @ R_curr.T) + t_curr.unsqueeze(0)  # (N, 3)

            # Transform to OpenGL camera for projection clipping
            pts_glcam = pts_cam @ glcam_mat[:3, :3].T + glcam_mat[:3, 3].unsqueeze(0)
            pts_homo = to_homo_torch(pts_glcam)  # (N, 4)
            pos_clip = (proj_mat @ pts_homo.T).T.unsqueeze(0).contiguous()  # (1, N, 4)

            # Rasterize
            rast_out, _ = dr.rasterize(self.glctx, pos_clip, faces, resolution=np.array([H, W]))

            # Interpolate 3D camera coordinates to get depth
            xyz_map, _ = dr.interpolate(pts_cam.unsqueeze(0).contiguous(), rast_out, faces)
            pred_depth = torch.flip(xyz_map[0, ..., 2], dims=[0])  # Flip Y to match OpenCV image coords

            # Differentiable Silhouette
            rendered_mask_raw = torch.clamp(rast_out[0, ..., -1], 0.0, 1.0)
            # Antialias for smooth silhouette edge gradients
            rendered_mask = dr.antialias(
                rendered_mask_raw.unsqueeze(0).unsqueeze(-1).expand(1, H, W, 1),
                rast_out,
                pos_clip,
                faces
            )[0, ..., 0]
            pred_mask = torch.flip(rendered_mask, dims=[0])

            loss = torch.tensor(0.0, device=self.device)

            # 1. Depth loss (Robust Huber/Smooth L1 loss on unoccluded foreground)
            if depth_gt is not None:
                rendered_valid = (pred_depth > 0.05) & (pred_depth < 5.0)
                eval_mask = rendered_valid & valid_depth_gt & unoccluded
                if obs_mask is not None:
                    eval_mask = eval_mask & (mask_gt > 0.5)

                if eval_mask.sum() > 20:
                    depth_diff = pred_depth[eval_mask] - depth_gt[eval_mask]
                    loss_depth = F.smooth_l1_loss(depth_diff, torch.zeros_like(depth_diff), beta=0.01)
                    loss = loss + w_depth * loss_depth

            # 2. Silhouette Mask Loss (BCE / MSE)
            if mask_gt is not None and w_sil > 0:
                sil_eval_mask = unoccluded
                loss_sil = F.mse_loss(pred_mask[sil_eval_mask], mask_gt[sil_eval_mask])
                loss = loss + w_sil * loss_sil

            # 3. Photometric / RGB loss (if texture available)
            if has_tex and rgb_gt is not None and w_rgb > 0:
                tex = mesh_tensors['tex'] if mesh_tensors['tex'].ndim == 4 else mesh_tensors['tex'].unsqueeze(0)
                texc, _ = dr.interpolate(mesh_tensors['uv'].unsqueeze(0).contiguous(), rast_out, mesh_tensors['uv_idx'].contiguous())
                pred_color = dr.texture(tex.contiguous(), texc, filter_mode='linear')[0]
                pred_color = torch.flip(pred_color, dims=[0])
                rgb_eval_mask = (pred_mask > 0.5) & (mask_gt > 0.5) & unoccluded
                if rgb_eval_mask.sum() > 20:
                    loss_rgb = F.l1_loss(pred_color[rgb_eval_mask], rgb_gt[rgb_eval_mask])
                    loss = loss + w_rgb * loss_rgb

            # 4. MAP Prior / Soft Anchor Regularization to prevent drift on corrupted noise
            loss_reg = 10.0 * torch.sum(trans_delta ** 2) + 2.0 * torch.sum(rot_vec ** 2)
            loss = loss + loss_reg

            loss_val = loss.item()
            if loss_val < best_loss and not torch.isnan(loss):
                best_loss = loss_val
                best_R = R_curr.detach().clone()
                best_t = t_curr.detach().clone()

            loss.backward()
            optimizer.step()
            scheduler.step()

        # Build final SE(3) pose matrix
        refined_pose = torch.eye(4, dtype=torch.float32, device=self.device)
        refined_pose[:3, :3] = best_R
        refined_pose[:3, 3] = best_t

        return refined_pose.cpu().numpy()
