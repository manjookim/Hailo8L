import os
import json
import numpy as np
import cv2
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import hailo_platform as hp

def preprocess_letterbox(img_path, target_shape=(640, 640)):
    target_h, target_w = target_shape
    img = cv2.imread(img_path)
    if img is None:
        print(f"Warning: Unable to read image {img_path}")
        return None
    original_h, original_w = img.shape[:2]
    scale = min(target_w / original_w, target_h / original_h)
    new_w, new_h = int(original_w * scale), int(original_h * scale)
    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padded_img = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
    pad_top = (target_h - new_h) // 2
    pad_left = (target_w - new_w) // 2
    padded_img[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized_img
    final_img = cv2.cvtColor(padded_img, cv2.COLOR_BGR2RGB)
    return final_img

hef_path = "/app/tappas/rpi2/npu/accuracy_new/degirum/yolov8s/yolov8s.hef"
hef = hp.HEF(hef_path)
params = hp.VDevice.create_params()
vdevice = hp.VDevice(params)
configure_params = hp.ConfigureParams.create_from_hef(hef, interface=hp.HailoStreamInterface.PCIe)
network_group = vdevice.configure(hef, configure_params)[0]

input_vinfo = hef.get_input_vstream_infos()[0]
output_vinfo = hef.get_output_vstream_infos()[0]

coco_gt = COCO("/app/tappas/rpi2/npu/accuracy/coco/annotations/instances_val2017.json")
img_dir = "/app/tappas/rpi2/npu/accuracy/coco/images/val2017"
results = []

model_h, model_w, _ = input_vinfo.shape
cats = coco_gt.loadCats(coco_gt.getCatIds())
yolo_to_coco_id_map = {i: cat['id'] for i, cat in enumerate(cats)}

input_vs_params = hp.InputVStreamParams.make_from_network_group(network_group, quantized=True, format_type=hp.FormatType.UINT8)
output_vs_params = hp.OutputVStreamParams.make_from_network_group(network_group, quantized=False, format_type=hp.FormatType.FLOAT32)

with network_group.activate():
    with hp.InferVStreams(network_group, input_vs_params, output_vs_params) as infer_pipeline:
        all_img_ids = coco_gt.getImgIds()
        num_images = len(all_img_ids)
        for i, img_id in enumerate(all_img_ids):
            img_info = coco_gt.loadImgs(img_id)[0]
            img_path = os.path.join(img_dir, img_info['file_name'])
            
            img = preprocess_letterbox(img_path, target_shape=(model_h, model_w))
            if img is None: continue
            input_data = {input_vinfo.name: np.expand_dims(img, axis=0)}
            
            output_data = infer_pipeline.infer(input_data)
            
            detections_by_class = output_data[output_vinfo.name][0]

            total_detections = sum(len(dets) for dets in detections_by_class)

            if total_detections > 0:
                print(f"\nImage {img_id}: Found {total_detections} total detections!")

            original_h, original_w = img_info['height'], img_info['width']
            scale_ratio = min(model_w / original_w, model_h / original_h)
            pad_w = (model_w - original_w * scale_ratio) / 2
            pad_h = (model_h - original_h * scale_ratio) / 2
            
            for class_id, class_detections in enumerate(detections_by_class):
                if len(class_detections) == 0:
                    continue
                
                for detection in class_detections:
                    if len(detection) != 5: continue
                    y_min_norm, x_min_norm, y_max_norm, x_max_norm, score = detection
                    
                    x_min = x_min_norm * model_w
                    y_min = y_min_norm * model_h
                    x_max = x_max_norm * model_w
                    y_max = y_max_norm * model_h

                    x1_scaled = (x_min - pad_w) / scale_ratio
                    y1_scaled = (y_min - pad_h) / scale_ratio
                    x2_scaled = (x_max - pad_w) / scale_ratio
                    y2_scaled = (y_max - pad_h) / scale_ratio

                    w_box = x2_scaled - x1_scaled
                    h_box = y2_scaled - y1_scaled

                    results.append({
                        "image_id": int(img_id),
                        "category_id": int(yolo_to_coco_id_map.get(int(class_id), -1)),
                        "bbox": [float(x1_scaled), float(y1_scaled), float(w_box), float(h_box)],
                        "score": float(score)
                    })

            print(f"Processed image {i+1}/{num_images}: {img_info['file_name']}", end='\r')

print("\n\nInference finished. Calculating mAP...")
res_file = "coco_results.json"
with open(res_file, "w") as f:
    json.dump(results, f)

coco_dt = coco_gt.loadRes(res_file)
coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()
