import re

import cv2
import numpy as np
from glob import glob
from ultralytics import YOLO


from src.utils import res_2_mask, project_mask_to_points, pca, compute_wall_factor
from src.utils import compute_occlude, compute_crowding_kdist, rank_instances
from src.visualization import visiual_seg, visual_order_pca, visualize_height_map


def inference(image_folder, weights):
    rgb_images = sorted(glob(f"{image_folder}/rgb/*.*"))
    model = YOLO(weights)

    for img_path in rgb_images:
        index = int(re.search(r"(\d+)", img_path).group(1))
        depth_path = f"{image_folder}/depth/Image_{index:03d}.npy"

        bgr = cv2.imread(img_path)
        height = np.load(depth_path)

        res = model(img_path, verbose=False, retina_masks=True)[0]
        # visiual_seg(res)

        # mask optimization: connected components labeling and minimum pixel threshold
        masks = res_2_mask(res)

        # from depth to pointcloud
        points, colors = project_mask_to_points(bgr, height, masks)

        instances = []
        features = []
        # pca calculation
        for i, (p, col) in enumerate(zip(points, colors)):
            c, (v1, v2), evals, l, wid = pca(p, col)
            inst = {"id": i, "center": c, "v1": v1, "length": l, "width": wid}
            height = {"id": i, "height": -1 * c[-1]}
            instances.append(inst)
            features.append(height)
        features = compute_occlude(instances, features)
        features = compute_crowding_kdist(instances, points, features)
        # features = compute_wall_factor(instances, boxrect, features)

        # priority calculation
        sort_features, sort_idx = rank_instances(features)
        visual_order_pca(masks, bgr, sort_idx, instances)

        # visualize_height_map(height, title=f"Height – Image {index}", units="mm")
        break


if __name__ == "__main__":
    image_folder = "./validate_data"
    weights = "./weights/yolo11_green_onion_Aug_31.pt"
    inference(image_folder, weights)
