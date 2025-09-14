import numpy as np
import cv2
import random
import matplotlib.pyplot as plt

from src.pca import pca_pose_from_mask, root_to_leaves_sign


def color_for(idx):
    random.seed(int(idx) + 42)
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))


def full_mask_view(result, *, bg="orig", alpha=0.45, outline=True, pca=False):
    """
    Build a full-image view with ONLY instance masks rendered.
    bg: "orig" to use the original image as background, "blank" for black.
    alpha: mask transparency.
    outline: draw polygon edges or not.
    """
    # base image
    base = result.orig_img.copy() if bg == "orig" else np.zeros_like(result.orig_img)
    if result.masks is None or result.masks.xy is None or len(result.masks.xy) == 0:
        return base

    overlay = base.copy()
    for i, poly in enumerate(result.masks.xy):
        # handle single or multi-part polygons
        if isinstance(poly, list):
            segs = [np.asarray(p, dtype=np.int32) for p in poly]
        else:
            segs = [np.asarray(poly, dtype=np.int32)]
        col = color_for(i)
        cv2.fillPoly(overlay, segs, col)
        if outline:
            cv2.polylines(overlay, segs, True, col, 2)

        if pca:
            m = result.masks.data[i].cpu().numpy()
            m = (m > 0).astype(np.uint8)
            if m.sum() < 50:   # skip tiny/degenerate masks
                continue
            center_xy, v_major, v_minor, _, evals = pca_pose_from_mask(m)
            sign, w_plus, w_minus = root_to_leaves_sign(
                base, m, center_xy, v_major)
            v_major = v_major * sign

            # visualization
            p_center = tuple(np.round(center_xy).astype(int))
            p_major = tuple(np.round(center_xy + 60 * v_major).astype(int))
            cv2.arrowedLine(overlay, p_center, p_major, (255, 0, 0),
                            2, tipLength=0.2)

            gp = tuple(np.round(center_xy).astype(int))
            cv2.circle(overlay, gp, 4, (0, 0, 255), -1)

    cv2.addWeighted(overlay, alpha, base, 1 - alpha, 0, base)
    return base


def show_instance(result, idx):
    """Render ONLY one instance (idx) on top of the original image."""
    img = result.orig_img.copy()

    # boxes / classes / confs
    boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else None
    clses = result.boxes.cls.cpu().numpy().astype(
        int) if result.boxes is not None else None
    confs = result.boxes.conf.cpu().numpy() if result.boxes is not None else None

    if result.masks is None:
        return img
    polys = result.masks.xy
    n = len(polys)
    if idx >= n:
        return img

    overlay = img.copy()
    segs = polys[idx]
    if isinstance(segs, list):
        segments = [np.asarray(s, dtype=np.int32) for s in segs]
    else:
        segments = [np.asarray(segs, dtype=np.int32)]

    cls_id = clses[idx] if clses is not None else 0
    color = color_for(idx)

    cv2.fillPoly(overlay, segments, color)
    cv2.addWeighted(overlay, 0.45, img, 0.55, 0, img)
    cv2.polylines(img, segments, isClosed=True, color=color, thickness=2)

    # text anchor
    if boxes is not None:
        x1, y1, x2, y2 = boxes[idx].astype(int)
        tx, ty = x1, max(0, y1 - 8)
    else:
        c = np.mean(np.vstack(segments), axis=0).astype(int)
        tx, ty = int(c[0]), max(0, int(c[1]) - 8)

    name_map = getattr(result, "names", None)
    name = name_map.get(int(cls_id), str(int(cls_id))) if isinstance(
        name_map, dict) else str(int(cls_id))
    conf_txt = f"{confs[idx]:.2f}" if confs is not None else "-"
    text = f"{name} {conf_txt}"

    cv2.rectangle(img, (tx, max(0, ty - 18)),
                  (tx + 8 * len(text), ty + 4), color, -1)
    cv2.putText(img, text, (tx + 3, ty - 2), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(img, f"[{idx+1}/{n}] Space/Right=next  Left=prev  f=full  n=next image  q=quit",
                (10, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA)
    return img


def visiual_seg(res):
    WIN = "instance-viewer"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    ninst = len(res.masks.xy)
    idx = 0

    while 0 <= idx < ninst:
        view = show_instance(res, idx)
        cv2.imshow(WIN, view)
        k = cv2.waitKey(0) & 0xFF
        if k in (32, ord('d'), 0x53):        # Space / 'd' / → next
            idx += 1
        elif k in (ord('a'), 0x51):          # 'a' / ← prev
            idx -= 1
        elif k == ord('f'):                  # show full immediately
            break
        elif k == ord('n'):                  # next image
            idx = ninst  # exit to summary then advance
            break
        elif k == ord('q'):                  # quit
            break
        else:
            idx += 1


def show_pca(result, idx, view):
    img = result.orig_img.copy()

    if result.masks is None:
        return view
    polys = result.masks.xy
    n = len(polys)
    if idx >= n:
        return view

    m = result.masks.data[idx].cpu().numpy()
    m = (m > 0).astype(np.uint8)
    if m.sum() < 50:   # skip tiny/degenerate masks
        return view
    center_xy, v_major, v_minor, _, evals = pca_pose_from_mask(m)
    sign, w_plus, w_minus = root_to_leaves_sign(img, m, center_xy, v_major)
    v_major = v_major * sign
    theta = np.arctan2(v_major[1], v_major[0])

    # visualization
    p_center = tuple(np.round(center_xy).astype(int))
    p_major = tuple(np.round(center_xy + 60 * v_major).astype(int))
    cv2.arrowedLine(view, p_center, p_major, (255, 0, 0),
                    2, tipLength=0.2)

    gp = tuple(np.round(center_xy).astype(int))
    cv2.circle(view, gp, 4, (0, 0, 255), -1)
    return view


def visual_order_pca(masks, bgr, sort_idx, out=None):
    WIN = "ordered-instance-viewer"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    ninst = len(masks)

    i = 0
    while 0 <= i < ninst:
        sorted_indx = sort_idx[i]
        view = show_instance_v2(masks, bgr, sorted_indx)
        view = show_pca_v2(masks, bgr, sorted_indx, view)
        cv2.imshow(WIN, view)

        if out:
            print(out[sorted_indx])

        k = cv2.waitKey(0) & 0xFF
        if k in (32, ord('d'), 0x53):        # Space / 'd' / → next
            i += 1
        elif k in (ord('a'), 0x51):          # 'a' / ← prev
            i -= 1
        elif k == ord('f'):                  # show full immediately
            break
        elif k == ord('n'):                  # next image
            i = ninst  # exit to summary then advance
            break
        elif k == ord('q'):                  # quit
            break
        else:
            i += 1


def show_instance_v2(masks, bgr, idx):
    """Render ONLY one instance (idx) on top of the original image."""
    vis = bgr.copy()
    m = masks[idx]
    rng = np.random.default_rng(42)
    colors = (rng.uniform(0, 255, size=(len(masks), 3)
                          ).astype(np.uint8))[:, ::-1]
    cnts, _ = cv2.findContours(
        m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, cnts, -1, colors[idx].tolist(), 2)
    return vis


def show_pca_v2(masks, bgr, sorted_idx, view):
    m = masks[sorted_idx]
    img = bgr.copy()
    center_xy, v_major, v_minor, _, evals = pca_pose_from_mask(m)
    sign, w_plus, w_minus = root_to_leaves_sign(img, m, center_xy, v_major)
    v_major = v_major * sign
    theta = np.arctan2(v_major[1], v_major[0])

    # visualization
    p_center = tuple(np.round(center_xy).astype(int))
    p_major = tuple(np.round(center_xy + 60 * v_major).astype(int))
    cv2.arrowedLine(view, p_center, p_major, (255, 0, 0),
                    2, tipLength=0.2)

    gp = tuple(np.round(center_xy).astype(int))
    cv2.circle(view, gp, 4, (0, 0, 255), -1)
    return view


def visualize_height_map(height_map: np.ndarray,
                         title: str = None,
                         percentile_clip=(2, 98),
                         # treat these as invalid if your map uses 0 for “no data”
                         invalid_values=(0,),
                         units: str = "m",
                         save_path: str = None,
                         show: bool = True):
    """
    Visualize a height/depth map with robust scaling and a colorbar.
    """
    h = height_map.astype(np.float32).copy()

    # Mask out invalid readings (e.g., zeros) if specified
    if invalid_values is not None:
        for iv in invalid_values:
            h[h == iv] = np.nan

    # Robust contrast: clip by percentiles ignoring NaNs
    finite_vals = h[np.isfinite(h)]
    if finite_vals.size == 0:
        print("Warning: height_map has no finite values to display.")
        return
    vmin, vmax = np.percentile(finite_vals, percentile_clip)

    plt.figure(figsize=(6, 5))
    im = plt.imshow(h, vmin=vmin, vmax=vmax)
    plt.title(title or "Height map")
    plt.axis('off')
    cbar = plt.colorbar(im)
    cbar.set_label(f"Height [{units}]")

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()