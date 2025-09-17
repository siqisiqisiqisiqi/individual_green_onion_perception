from glob import glob
import re

from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt

from src.utils import res_2_mask, project_mask_to_points, pca
from src.visualization import show_instance_v2


def compute_width_profile(points,
                          colors,
                          K=64,
                          trim=0.02,
                          slab_sigma=2.0,
                          min_pts=10,
                          smooth_window=5,
                          visual=True):
    """
    Compute width profile y(x) along the PCA long axis.

    Args:
        points: (N,2) or (N,>=2) array of 2D points (e.g., from mask contour or 2D projection).
        K: number of evenly spaced stations along the long axis.
        trim: percentile-trim for outlier rejection in width (0..0.5). E.g., 0.02 = 2% at each tail.
        slab_sigma: half-slab size in units of station spacing (controls how wide each station samples).
        min_pts: minimum points needed in a slab to compute width; else yields NaN (later interpolated).
        smooth_window: optional odd integer for moving-average smoothing of width (0 = no smoothing).
        return_extra: if True, also returns axis-centerline projections, masks, etc.

    Returns:
        x_norm: (K,) stations in [0,1] along long axis
        width:  (K,) width (diameter) at each station (same units as input)
        L:      total length along long axis (max - min projection)
        meta:   dict with C, v1, v2, t stations, slab_halfwidth, etc. (if return_extra=True)
    """
    P = points[:, :2]
    C, (v1, v2), _, max_l, max_width = pca(P, colors)

    # Project points into PCA frame
    X = P - C
    t = X @ v1  # coordinate along the long axis
    s = X @ v2  # coordinate along the short axis

    # Length and stations
    t_min, t_max = np.min(t), np.max(t)
    L = float(t_max - t_min)
    # station centers (world units along v1)
    t_k = np.linspace(t_min, t_max, K)
    x_norm = (t_k - t_min) / L
    # slab half-width as a multiple of station spacing
    dt = (t_max - t_min) / max(K - 1, 1)
    slab_halfwidth = slab_sigma * dt

    # Compute width per station using a slab of points around each t_k
    width = np.full(K, np.nan, dtype=np.float32)
    S_HI = np.full(K, np.nan, dtype=np.float32)
    S_LO = np.full(K, np.nan, dtype=np.float32)

    for i, tk in enumerate(t_k):
        mask = np.abs(t - tk) <= slab_halfwidth
        if np.count_nonzero(mask) < min_pts:
            continue
        s_slice = np.sort(s[mask])

        # Trim percentiles to suppress outliers/jaggies
        if trim > 0.0:
            lo = int(np.floor(trim * len(s_slice)))
            hi = int(np.ceil((1 - trim) * len(s_slice))) - 1
            lo = np.clip(lo, 0, len(s_slice) - 1)
            hi = np.clip(hi, 0, len(s_slice) - 1)
            s_lo, s_hi = s_slice[lo], s_slice[hi]
        else:
            s_lo, s_hi = s_slice[0], s_slice[-1]

        width[i] = float(s_hi - s_lo)  # diameter at this station
        S_HI[i] = s_hi
        S_LO[i] = s_lo

    # Fill small gaps (NaNs) by linear interpolation over valid neighbors
    if np.any(np.isnan(width)):
        valid = ~np.isnan(width)
        if valid.any():
            idx = np.arange(K)
            width = np.interp(idx, idx[valid], width[valid])
        else:
            width[:] = 0.0

    # Optional simple smoothing (moving average with odd window)
    if isinstance(smooth_window, int) and smooth_window >= 3 and smooth_window % 2 == 1:
        pad = smooth_window // 2

        padded = np.pad(width, (pad, pad), mode='edge')
        kernel = np.ones(smooth_window, dtype=np.float32) / smooth_window
        width = np.convolve(padded, kernel, mode='valid')

        padded = np.pad(S_HI, (pad, pad), mode='edge')
        kernel = np.ones(smooth_window, dtype=np.float32) / smooth_window
        S_HI = np.convolve(padded, kernel, mode='valid')

        padded = np.pad(S_LO, (pad, pad), mode='edge')
        kernel = np.ones(smooth_window, dtype=np.float32) / smooth_window
        S_LO = np.convolve(padded, kernel, mode='valid')

    if visual:
        dpi = 100
        fig_w = 1920 / dpi
        fig_h = 300 / dpi

        fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
        ax = fig.add_subplot(111)
        ax.plot(x_norm, width)           # default color; no style assumptions
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("green onion profile")
        ax.grid(True)
        fig.tight_layout()

        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        rgba = np.asarray(renderer.buffer_rgba())     # shape (h, w, 4)
        bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
        plt.close(fig)
        return (x_norm, width, max_l, max_width, bgr)
    else:
        return (x_norm, width, max_l, max_width)


def stack_and_show(img_bgr, plot_bgr, layout="horizontal", winname="image+plot"):
    """
    Stack the image and plot and display with cv2.imshow.
    layout: "vertical" (image over plot) or "horizontal" (side-by-side)
    """
    if layout == "vertical":
        # match widths
        W = img_bgr.shape[1]
        plot_bgr = cv2.resize(
            plot_bgr, (W, plot_bgr.shape[0]), interpolation=cv2.INTER_AREA)
        combo = np.vstack([img_bgr, plot_bgr])
    else:
        # match heights
        H = img_bgr.shape[0]
        plot_bgr = cv2.resize(
            plot_bgr, (plot_bgr.shape[1], H), interpolation=cv2.INTER_AREA)
        combo = np.hstack([img_bgr, plot_bgr])

    cv2.imshow(winname, combo)
    k = cv2.waitKey(0)
    return k


weights = "./weights/yolo11_green_onion_Aug_31.pt"
rgb_images = sorted(glob("./validate_data/rgb/Image_008.png"))

for img_path in rgb_images:
    index = int(re.search(r"(\d+)", img_path).group(1))
    depth_path = f"./validate_data/depth/Image_{index:03d}.npy"
    images = img_path

    model = YOLO(weights)
    bgr = cv2.imread(images)
    height = np.load(depth_path)

    res = model(images, verbose=False, retina_masks=True)[0]
    masks = res_2_mask(res)
    # colors in rgb01
    points, colors = project_mask_to_points(bgr, height, masks)

    WIN = "instance+profile"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    for idx, (p, col) in enumerate(zip(points, colors)):
        x_norm, width, max_l, max_width, bgr_profile = compute_width_profile(
            p, col)

        # print(f"x_norm is {max_l}.")
        # print(f"width is {max_width}.")

        vis = show_instance_v2(masks, bgr, idx)
        k = stack_and_show(vis, bgr_profile, layout="vertical", winname=WIN)
        if k == ord('q'):
            cv2.destroyAllWindows()               # quit
            break
    break
