import degirum as dg
import json
import os

# 1. 모델 로드 (이전 단계에서 생성한 JSON 파일)
model = dg.load_model(
    model_name='yolov8s',
    inference_host_address='@local',
    zoo_url='/app/tappas/rpi2/npu/accuracy_new/degirum/yolov8s'
)
coco_images_dir = '/app/tappas/rpi2/npu/accuracy/coco/images/val2017'
image_paths = [os.path.join(coco_images_dir, f) for f in os.listdir(coco_images_dir)]

all_predictions = []

for image_path in image_paths:
    image_id = int(os.path.basename(image_path).split('.')[0])
    inference_result = model(image_path)
    all_predictions.extend(inference_result.results)
# 5. 모든 예측 결과를 JSON 파일로 저장
with open('predictions.json', 'w') as f:
    json.dump(all_predictions, f)
