from ultralytics import YOLO
import time
from glob import glob

# weight = "./weights/yolo11_green_onion_Sep_03_without_occlude.pt"
# model = YOLO(weight)

# model.export(format="onnx")
onnx_model = YOLO(
    "./weights/yolo11_green_onion_Sep_03_without_occlude.onnx", task='segment')
image_list = glob("./validate_data/rgb/*")
for img in image_list:
    t = time.time()
    result = onnx_model(img)
    print(time.time() - t)
