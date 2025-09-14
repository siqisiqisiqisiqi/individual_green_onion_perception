import os
import time

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import cv2
from sklearn.neighbors import NearestNeighbors

extrinsic_param = "./validate_data/camera_param/extri_param.npz"

with np.load(extrinsic_param) as X:
    mtx, dist, Mat, tvecs = [X[i] for i in ('mtx', 'dist', 'Mat', 'tvecs')]

fx = mtx[0, 0]
fy = mtx[1, 1]
cx = mtx[0, 2]
cy = mtx[1, 2]


def _green_score(colors, mode="g_minus_max"):
    """
    Robust 'greenness' score per point from RGB in [0,1].
    - g_minus_max:  G - max(R,B)  (≈0 for white/yellow, >0 for green)
    - g_minus_rb:   G - 0.5*(R+B)
    - ndgi:         (G - R) / (G + R + 1e-9)
    """
    R, G, B = colors[:, 0], colors[:, 1], colors[:, 2]
    if mode == "g_minus_max":
        return G - np.maximum(R, B)
    elif mode == "g_minus_rb":
        return G - 0.5 * (R + B)
    elif mode == "ndgi":
        return (G - R) / (G + R + 1e-9)
    else:
        raise ValueError(f"unknown green metric: {mode}")


def robust_minmax(x, lo=5, hi=95, eps=1e-9):
    x = np.asarray(x, dtype=float)
    finite = np.isfinite(x)
    if not np.any(finite):
        return np.zeros_like(x)
    p_lo, p_hi = np.percentile(x[finite], [lo, hi])
    if abs(p_hi - p_lo) < eps:
        return np.zeros_like(x)
    return np.clip((x - p_lo) / (p_hi - p_lo), 0.0, 1.0)


def voxel_downsample_np(points, voxel=0.001):
    """
    Downsample by keeping ONE point per voxel (cube of size `voxel`, in meters).
    Returns a subset of `points` (no averaging).
    Also returns the indices chosen, so you can apply them to colors/normals too.
    """
    if len(points) == 0:
        return points, np.array([], dtype=int)

    # 1) quantize to voxel grid (works for negative coords too)
    keys = np.floor(points / voxel).astype(np.int64)  # (N,3) integers

    # 2) row-wise unique using a structured view
    keys_view = keys.view([('', keys.dtype)] * keys.shape[1]).ravel()
    # first point per occupied voxel
    _, idx = np.unique(keys_view, return_index=True)

    # 3) pick representatives
    return points[idx], idx


def res_2_mask(res, do_close=True):
    masks = res.masks.data.cpu().numpy()
    opt_masks = []
    for m in masks:
        m = (m > 0).astype(np.uint8)

        # skip tiny/degenerate masks
        if m.sum() < 100:
            continue

        num, labels, stats, _ = cv2.connectedComponentsWithStats(
            m, connectivity=8)

        # only pick the largest part of the segmentation
        if num >= 3:
            m_filtered = np.zeros_like(m)
            areas = stats[1:, cv2.CC_STAT_AREA]
            if np.max(areas) < 100:
                continue
            main_id = 1 + np.argmax(areas)
            m_filtered[labels == main_id] = 1
        else:
            m_filtered = np.copy(m)
        if do_close:
            m_filtered = cv2.morphologyEx(
                m_filtered, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)

        # # visualize the green onion mask
        # plt.imshow(m_filtered, cmap="gray", vmin=0, vmax=1)
        # plt.axis("off")
        # plt.show()
        opt_masks.append(m_filtered)

    return np.array(opt_masks)


def precompute_rays_world(H, W):
    """
    Precompute world-space ray directions r_w(u,v) for each pixel:
      P_w = C_w + s * r_w,  where C_w is camera center in world.
    T_wc: 4x4 transform (camera->world): P_w = R_wc @ P_c + t_wc
    """
    R_wc = Mat
    t_wc = tvecs.squeeze()  # camera origin in world

    u = np.arange(W, dtype=np.float32)
    v = np.arange(H, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)  # (H,W)

    # Direction in camera coords (pinhole, z=1)
    x_c = (uu - cx) / fx
    y_c = (vv - cy) / fy
    d_c = np.stack([x_c, y_c, np.ones_like(x_c)], axis=-1)  # (H,W,3)

    # Rotate to world frame (no need to normalize)
    # r_w = R_wc @ d_c
    r_w = d_c @ R_wc.T  # (H,W,3)

    return r_w.astype(np.float32), t_wc.astype(np.float32)


def project_mask_to_points(bgr, height_mm, masks, max_pts=50000):
    H, W, _ = bgr.shape
    rgb01 = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rays_w, C_w = precompute_rays_world(H, W)

    height_mm = height_mm * (-1)
    z_min_mm, z_max_mm = -100, 100
    max_pts = int(1e6)
    eps = 1e-9
    geoms, all_points, all_points_color = [], [], []

    for mask in masks:
        ys, xs = np.nonzero(mask)
        Zmm = height_mm[ys, xs].astype(np.float32)
        valid = (Zmm >= z_min_mm) & (Zmm <= z_max_mm) & np.isfinite(Zmm)
        if not np.any(valid):
            all_points.append(np.empty((0, 3), np.float32))
            continue

        xs, ys, Zmm = xs[valid], ys[valid], Zmm[valid]
        r = rays_w[ys, xs]
        col = rgb01[ys, xs]              # colors from the image

        rwz = r[:, 2]
        good = np.abs(rwz) > 1e-8
        if not np.any(good):
            all_points.append(np.empty((0, 3), np.float32))
            continue

        r, Zmm, col = r[good], Zmm[good], col[good]
        Zw = Zmm / 1000.0
        s = (Zw - C_w[2]) / (r[:, 2] + eps)
        Pw = C_w[None, :] + s[:, None] * r

        if len(Pw) > max_pts:
            idx = np.random.default_rng(0).choice(
                len(Pw), size=max_pts, replace=False)
            Pw, col = Pw[idx], col[idx]

        Pw, downsample_idx = voxel_downsample_np(Pw)
        col = col[downsample_idx]

        ## visualization test
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(Pw.astype(np.float32))
        # pcd.colors = o3d.utility.Vector3dVector(col.astype(np.float32))
        # axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        # o3d.visualization.draw_geometries([pcd, axes])

        # c, (v1, v2, v3), evals, l = pca_3d(Pw)

        # len1 = 0.05
        # len2 = 0.02
        # p0 = c.astype(float)
        # p1 = (c + len1 * v1).astype(float)   # endpoint for v1
        # p2 = (c + len2 * v2).astype(float)   # endpoint for v2
        # axes_ls = o3d.geometry.LineSet()
        # axes_ls.points = o3d.utility.Vector3dVector(np.vstack([p0, p1, p2]).astype(np.float32))
        # axes_ls.lines  = o3d.utility.Vector2iVector(np.array([[0, 1], [0, 2]], dtype=np.int32))
        # axes_ls.colors = o3d.utility.Vector3dVector(np.array([
        #     [1.0, 0.0, 0.0],  # v1 in red
        #     [0.0, 1.0, 0.0],  # v2 in green
        # ], dtype=np.float32))

        # # Small sphere to mark the PCA center
        # center_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.002)
        # center_sphere.paint_uniform_color([1.0, 1.0, 0.0])  # yellow
        # center_sphere.translate(p0)  # move to center

        # geoms.append(pcd)
        # geoms.append(axes_ls)
        # geoms.append(center_sphere)

        # geoms.append(pcd)

        all_points.append(Pw.astype(np.float32))
        all_points_color.append(col.astype(np.float32))

    # axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    # o3d.visualization.draw_geometries([*geoms, axes])
    return all_points, all_points_color


def pca_3d(points):
    C = points.mean(axis=0)
    X = points - C
    cov = (X.T @ X) / max(len(points), 1)
    w, V = np.linalg.eigh(cov)     # ascending eigvals
    v1, v2, v3 = V[:, 2], V[:, 1], V[:, 0]   # principal -> smallest
    # normalize
    v1 /= np.linalg.norm(v1) + 1e-9
    v2 /= np.linalg.norm(v2) + 1e-9
    v3 /= np.linalg.norm(v3) + 1e-9

    t = (points - C) @ v1
    tmin, tmax = float(np.percentile(t, 5)), float(np.percentile(t, 95))
    L = tmax - tmin
    return C, (v1, v2, v3), w[::-1], L


def pca(points, colors, green_mode="g_minus_max", end_quantile=0.1):
    C = points.mean(axis=0)

    points2d = points[:, :2]
    C2d = points2d.mean(axis=0)
    X = points2d - C2d
    cov = (X.T @ X) / max(len(points2d), 1)
    w, V = np.linalg.eigh(cov)     # ascending eigvals
    v1, v2 = V[:, 1], V[:, 0]   # principal -> smallest
    # normalize
    v1 /= np.linalg.norm(v1) + 1e-9
    v2 /= np.linalg.norm(v2) + 1e-9

    s = (points2d - C2d) @ v2
    smin, smax = float(np.percentile(s, 5)), float(np.percentile(s, 95))
    width = smax - smin

    t = (points2d - C2d) @ v1
    tmin, tmax = float(np.percentile(t, 5)), float(np.percentile(t, 95))
    L = tmax - tmin

    # direction
    gscore = _green_score(np.asarray(colors), mode=green_mode)
    q_lo = np.quantile(t, end_quantile)
    q_hi = np.quantile(t, 1.0 - end_quantile)
    neg_mask = t <= q_lo       # "negative end" of the axis
    pos_mask = t >= q_hi       # "positive end" of the axis
    # Use median for robustness
    pos_green = float(np.median(gscore[pos_mask])) if np.any(
        pos_mask) else np.nan
    neg_green = float(np.median(gscore[neg_mask])) if np.any(
        neg_mask) else np.nan
    flipped = False
    # If the positive end is *less* green than the negative end, flip v1 (point to leaves)
    if np.isfinite(pos_green) and np.isfinite(neg_green) and (pos_green < neg_green):
        v1 = -v1
        v2 = -v2
        flipped = True
    return C, (v1, v2), w[::-1], L, width


# todo: occlude metric should also try to include the width occlusion
def compute_occlude(instances, features, expected_l=0.12, expected_w=0.014):
    ids = [inst["id"] for inst in instances]
    lengths = [inst["length"] for inst in instances]
    widths = [inst["width"] for inst in instances]

    for oid, L, W in zip(ids, lengths, widths):
        occlusion = 0.5 * min(L / expected_l, 1.0) + \
            0.5 * min(W / expected_w, 1.0)
        features[oid]["occlusion"] = float(occlusion)

    return features


def sample_points(P, m):
    n = len(P)
    idx = np.random.choice(n, size=m, replace=True)
    return P[idx]


def compute_crowding_kdist(instances, points, features, sample_per_inst=50, k=50):
    """
    Compute 'crowding_kdist' for each instance:
    the mean distance (mm) to the k nearest points belonging to OTHER instances.
    """
    all_pts = []
    all_lbl = []
    for oid, p in enumerate(points):
        p = sample_points(p, int(len(p) * 0.1))
        all_pts.append(p)
        all_lbl.append(np.full(len(p), oid, dtype=np.int64))

    all_pts = np.vstack(all_pts)            # shape (M, 3)
    all_lbl = np.concatenate(all_lbl)

    # 2) Build a single kNN index over ALL points.
    #    Ask for extra neighbors (k + buffer) so we can drop self-labeled points.
    # small buffer, bounded by dataset size
    buffer = 5
    result = {}
    for inst in instances:

        oid = inst['id']
        q = points[oid]
        q = sample_points(q, sample_per_inst)

        R = all_pts[all_lbl != oid]
        nns = NearestNeighbors(
            n_neighbors=min(k + buffer, len(R)),
            algorithm="ball_tree"
        ).fit(R)

        # 3) Query neighbors for Q among ALL points once.
        dists, _ = nns.kneighbors(q, return_distance=True)

        per_q = np.mean(
            dists, axis=1) if dists.size > 0 else np.array([np.inf])
        features[oid]["kdist"] = float(
            np.mean(per_q)) if len(per_q) else float("inf")

    return features


def compute_wall_factor(instances, box_rect, features, p_mid=1.5):
    centers = [inst["center"] for inst in instances]
    for idx, c in enumerate(centers):
        x, y = float(c[0]), float(c[1])
        xL, xR, yT, yB = map(float, box_rect)
        W = max(xR - xL, 1e-9)
        H = max(yB - yT, 1e-9)
        # --- 1) Center preference (left–right) ---
        y_lr = np.clip((y - yT) / H, 0.0, 1.0)
        S_mid = 1.0 - (2.0 * abs(y_lr - 0.5))**p_mid  # center=1, edges→0
        features[idx]["wall_factor"] = S_mid
    return features


def rank_instances(features, w_occ=0.5, w_kdist=0.2, w_height=0.3):
    """
    features: list of dicts like
      {'id': int, 'height': float, 'occlusion': float, 'kdist': float}
    returns: list sorted best→worst, each with added 'score' and normalized fields
    """
    ids = np.array([f["id"] for f in features])
    occ = np.array([f["occlusion"]
                   for f in features], dtype=float)        # ↑ better
    kdist = np.array([f["kdist"] for f in features],
                     dtype=float)        # ↑ better
    height = np.array([f["height"] for f in features],
                      dtype=float)        # ↑ better

    occ_n = np.clip(occ, 0.0, 1.0)
    kdist_n = robust_minmax(kdist, lo=0, hi=100)
    height_n = robust_minmax(height, lo=5, hi=95)

    score = w_occ * occ_n + w_kdist * kdist_n + w_height * height_n

    out = []
    for f, s, on, kn, hn in zip(features, score, occ_n, kdist_n, height_n):
        g = dict(f)
        g.update({"score": float(s), "occ_n": float(on),
                 "kdist_n": float(kn), "height_n": float(hn)})
        out.append(g)

    out.sort(key=lambda d: d["score"], reverse=True)
    sort_id = [inst['id'] for inst in out]
    return out, sort_id
