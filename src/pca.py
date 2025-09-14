import numpy as np
import cv2


def pca_pose_from_mask(mask: np.ndarray):
    """
    mask: uint8/bool array [H, W] where 1 indicates the onion region.
    Returns:
      center_xy: (x, y)
      v_major: unit vector along onion length (principal axis)
      v_minor: unit vector across onion thickness (grasp direction)
      theta_rad: orientation angle of v_major in image frame
      evals: eigenvalues (var along axes)
    """
    ys, xs = np.nonzero(mask)
    pts = np.stack([xs, ys], axis=1).astype(np.float32)  # [N, 2] in (x, y)

    center = pts.mean(axis=0)
    cov = np.cov(pts, rowvar=False)  # 2x2
    evals, evecs = np.linalg.eigh(cov)  # symmetric -> eigh
    order = np.argsort(evals)[::-1]     # sort descending
    evals = evals[order]
    evecs = evecs[:, order]
    v_major = evecs[:, 0] / np.linalg.norm(evecs[:, 0])
    v_minor = evecs[:, 1] / np.linalg.norm(evecs[:, 1])
    theta = np.arctan2(v_major[1], v_major[0])  # radians

    return center, v_major, v_minor, theta, evals


def _whiteness_map_Lab(image_bgr, gamma=0.8):
    """Higher = whiter (bulb). Robust to lighting compared to RGB."""
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[..., 0]
    a = lab[..., 1] - 128.0
    b = lab[..., 2] - 128.0
    chroma = np.sqrt(a * a + b * b)
    return L - gamma * chroma


def root_to_leaves_sign(image_bgr, inst_mask, center_xy, v_major, end_pct=0.15):
    """
    Decide the sign of v_major so that +v_major points from ROOT -> LEAVES.
    We look at the top/bottom 'end_pct' of projected pixels and compare whiteness.
    If the '+' end is whiter (root), the arrow to leaves must be the '-' direction.
    Returns: sign ∈ {+1, -1}, and the mean whiteness at the two ends for debugging.
    """
    ys, xs = np.nonzero(inst_mask)
    pts = np.stack([xs, ys], axis=1).astype(np.float32)

    # project pixels onto the major axis
    t = (pts - center_xy) @ v_major
    lo = np.percentile(t, 100 * end_pct)
    hi = np.percentile(t, 100 * (1.0 - end_pct))
    sel_plus = t >= hi
    sel_minus = t <= lo

    W = _whiteness_map_Lab(image_bgr)
    # gather whiteness at those end pixels
    w_plus = W[pts[sel_plus, 1].astype(int), pts[sel_plus, 0].astype(int)]
    w_minus = W[pts[sel_minus, 1].astype(int), pts[sel_minus, 0].astype(int)]
    m_plus = float(w_plus.mean()) if w_plus.size else -1e9
    m_minus = float(w_minus.mean()) if w_minus.size else -1e9

    # If '+' end is whiter → '+' is ROOT → vector to LEAVES must be the '-' direction.
    sign = -1 if m_plus > m_minus else +1
    return sign, m_plus, m_minus