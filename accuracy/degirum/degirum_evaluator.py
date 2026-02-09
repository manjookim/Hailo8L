import degirum as dg
import degirum_tools
from degirum_tools.evaluator.detection_eval import ObjectDetectionModelEvaluator

# Load the detection model
model = dg.load_model(
    model_name="yolov8n-det--640x640_quant_hailort_multidevice_1",
    inference_host_address="@local",
    zoo_url="/app/tappas/rpi2/npu/accuracy_new/degirum/yolov8n-det--640x640_quant_hailort_multidevice_1",
    token=''
)

# Optional class ID remapping: model → COCO
classmap = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
            27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51,
            52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77,
            78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]

# Create evaluator
evaluator = ObjectDetectionModelEvaluator(model, classmap=classmap)

# Evaluation inputs
image_dir = "/app/tappas/rpi2/npu/accuracy/coco/images/val2017"
coco_json = "/app/tappas/rpi2/npu/accuracy/coco/annotations/instances_val2017.json"


# Evaluate and return mAP results
results = evaluator.evaluate(image_dir, coco_json, max_images=0)

# Print COCO-style mAP results
print("COCO mAP stats:", results[0])
