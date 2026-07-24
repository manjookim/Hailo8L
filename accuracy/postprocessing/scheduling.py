# -*- coding: utf-8 -*-
import multiprocessing as mp
import hailo_platform as hp
import numpy as np
import cv2 as cv
import glob, os, time, csv, re, datetime, sys
from pycocotools import mask as mask_util
import time


DET_PATH  = "/app/tappas/rpi2/npu/yolov8s/yolov8s.hef"
SEG_PATH  = "/app/tappas/rpi2/npu/yolov8s/yolov8s_seg.hef"
POSE_PATH = "/app/tappas/rpi2/npu/yolov8s/yolov8s_pose.hef"

IMAGE_DIR = "/app/tappas/rpi2/npu/accuracy/coco/images/val2017"
NUM_IMAGES = 100

coco_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
COCO_ID_MAP = {i: coco_ids[i] for i in range(len(coco_ids))}

#SCHEDULER_CONFIG = {
#    "DET":  {"priority": 3, "threshold": 5},
#    "SEG":  {"priority": 5, "threshold": 3},
#    "POSE": {"priority": 1, "threshold": 1},
#    "TIMEOUT": 30
#}

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)

def preprocess_letterbox(img_path, target_shape=(640, 640)):
    target_h, target_w = target_shape
    img = cv.imread(img_path)
    if img is None: return None, None
    
    original_h, original_w = img.shape[:2]
    scale = min(target_w / original_w, target_h / original_h)
    new_w, new_h = int(original_w * scale), int(original_h * scale)
    resized_img = cv.resize(img, (new_w, new_h), interpolation=cv.INTER_LINEAR)
    
    padded_img = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
    pad_top = (target_h - new_h) // 2
    pad_left = (target_w - new_w) // 2
    padded_img[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized_img
    
    # BGR -> RGB 변환 및 UINT8 유지
    final_img = cv.cvtColor(padded_img, cv.COLOR_BGR2RGB)
    meta = {
        'orig_h': original_h, 'orig_w': original_w, 
        'pad_h': pad_top, 'pad_w': pad_left, 
        'scale': scale
    }
    return final_img, meta

class YOLOv8DetDecoder:
    def __init__(self, model_shape=(640, 640), id_map=None):
        self.model_shape = model_shape
        self.id_map = id_map or {}

    def decode(self, detections, meta):
        results = []
        model_h, model_w = self.model_shape

        for class_id, class_dets in enumerate(detections):
            for det in class_dets:
                if len(det) < 5: continue
                
                ymin_n, xmin_n, ymax_n, xmax_n, score = det
                if score < 0.25: continue 

                x1 = (xmin_n * model_w - meta['pad_w']) / meta['scale']
                y1 = (ymin_n * model_h - meta['pad_h']) / meta['scale']
                x2 = (xmax_n * model_w - meta['pad_w']) / meta['scale']
                y2 = (ymax_n * model_h - meta['pad_h']) / meta['scale']

                x_min = max(0, min(x1, x2))
                y_min = max(0, min(y1, y2))
                w = max(0, abs(x2 - x1))
                h = max(0, abs(y2 - y1))
                
                results.append({
                    "image_id": int(img_id),
                    "category_id": int(id_map.get(class_id, class_id)),
                    "bbox": [float(x_min), float(y_min), float(w), float(h)],
                    "score": float(score)
                })
        return results


class YOLOv8SegDecoder:
    CONF_THRES = 0.01
    IOU_THRES  = 0.65
    MASK_THRES = 0.5
    MAX_DET    = 300
    REG_MAX    = 16
    NUM_CLS    = 80
    NUM_COEFF  = 32
    
    PROTO_KEY = "yolov8s_seg/conv48"
    HEADS = [
        ("yolov8s_seg/conv44", "yolov8s_seg/conv45", "yolov8s_seg/conv46", 8),
        ("yolov8s_seg/conv60", "yolov8s_seg/conv61", "yolov8s_seg/conv62", 16),
        ("yolov8s_seg/conv73", "yolov8s_seg/conv74", "yolov8s_seg/conv75", 32),
    ]

    def dfl_decode(self, box_raw: np.ndarray) -> np.ndarray:
        x = np.exp(box_raw - box_raw.max(-1, keepdims=True))
        x /= x.sum(-1, keepdims=True)
        proj = np.arange(self.REG_MAX, dtype=np.float32)
        return (x * proj).sum(-1)


    def decode_head(self, box_feat: np.ndarray, cls_feat: np.ndarray,
                    coef_feat: np.ndarray, stride: int):
        """
        box_feat : (1, H, W, 4*self.REG_MAX)  raw logit
        cls_feat : (1, H, W, self.NUM_CLS)    already sigmoid
        coef_feat: (1, H, W, self.NUM_COEFF)
        반환: boxes_xyxy (N,4), scores (N,), class_ids (N,), coeffs (N,32)
        """
        _, H, W, _ = box_feat.shape
        N = H * W

        box_raw = box_feat.reshape(N, 4, self.REG_MAX)
        cls     = cls_feat.reshape(N, self.NUM_CLS)    
        coeff   = coef_feat.reshape(N, self.NUM_COEFF)

        ltrb = self.dfl_decode(box_raw)

        ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        cx = (xs.ravel() + 0.5).astype(np.float32)
        cy = (ys.ravel() + 0.5).astype(np.float32)

        x1 = (cx - ltrb[:, 0]) * stride
        y1 = (cy - ltrb[:, 1]) * stride
        x2 = (cx + ltrb[:, 2]) * stride
        y2 = (cy + ltrb[:, 3]) * stride
        boxes = np.stack([x1, y1, x2, y2], axis=1)

        class_ids = cls.argmax(axis=1)
        scores    = cls[np.arange(N), class_ids]  

        return boxes, scores, class_ids, coeff

    def nms(self, boxes: np.ndarray, scores: np.ndarray) -> list:
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1).clip(0) * (y2 - y1).clip(0)
        order = scores.argsort()[::-1]
        keep  = []
        while order.size:
            i = order[0]
            keep.append(i)
            xx1   = np.maximum(x1[i], x1[order[1:]])
            yy1   = np.maximum(y1[i], y1[order[1:]])
            xx2   = np.minimum(x2[i], x2[order[1:]])
            yy2   = np.minimum(y2[i], y2[order[1:]])
            inter = (xx2 - xx1).clip(0) * (yy2 - yy1).clip(0)
            iou   = inter / (areas[i] + areas[order[1:]] - inter + 1e-7)
            order = order[1:][iou <= self.IOU_THRES]
        return keep

    def process_masks(self, proto: np.ndarray, coeffs: np.ndarray,
                    boxes_xyxy: np.ndarray, meta: dict,
                    orig_hw: tuple) -> list:
        Ph, Pw   = proto.shape[1], proto.shape[2]
        proto_2d = proto[0].reshape(Ph * Pw, self.NUM_COEFF)

        masks_raw = (proto_2d @ coeffs.T).T
        masks_raw = masks_raw.reshape(len(coeffs), Ph, Pw)
        masks_sig = 1.0 / (1.0 + np.exp(-masks_raw))

        orig_h, orig_w = orig_hw
        pad_h, pad_w   = meta['pad_h'], meta['pad_w']
        scale          = meta['scale']
        sx, sy         = Pw / 640.0, Ph / 640.0

        rles = []
        for i in range(len(coeffs)):
            mask = masks_sig[i]

            x1p = int(np.clip(boxes_xyxy[i, 0] * sx, 0, Pw))
            y1p = int(np.clip(boxes_xyxy[i, 1] * sy, 0, Ph))
            x2p = int(np.clip(boxes_xyxy[i, 2] * sx, 0, Pw))
            y2p = int(np.clip(boxes_xyxy[i, 3] * sy, 0, Ph))
            crop = np.zeros_like(mask)
            crop[y1p:y2p, x1p:x2p] = mask[y1p:y2p, x1p:x2p]

            mask_640   = cv.resize(crop, (640, 640), interpolation=cv.INTER_LINEAR)
            h_unpad    = int(orig_h * scale)
            w_unpad    = int(orig_w * scale)
            mask_unpad = mask_640[pad_h: pad_h + h_unpad,
                                pad_w: pad_w + w_unpad]
            mask_orig  = cv.resize(mask_unpad, (orig_w, orig_h),
                                    interpolation=cv.INTER_LINEAR)

            binary = (mask_orig > self.MASK_THRES).astype(np.uint8)
            rle = mask_util.encode(np.asfortranarray(binary))
            rle['counts'] = rle['counts'].decode('utf-8')
            rles.append(rle)

        return rles

    def unpad_boxes(self, boxes_xyxy: np.ndarray, meta: dict, orig_hw: tuple) -> np.ndarray:
        orig_h, orig_w = orig_hw
        pad_h, pad_w   = meta['pad_h'], meta['pad_w']
        scale          = meta['scale']

        b = boxes_xyxy.copy().astype(np.float32)
        b[:, [0, 2]] -= pad_w
        b[:, [1, 3]] -= pad_h
        b /= scale
        b[:, [0, 2]] = b[:, [0, 2]].clip(0, orig_w)
        b[:, [1, 3]] = b[:, [1, 3]].clip(0, orig_h)
        return b

    def decode(self, raw_results, meta) -> list:
        img_id  = int(meta['img_id'])
        raw     = raw_results
        orig_hw = (meta['orig_h'], meta['orig_w'])

        proto = raw[self.PROTO_KEY]

        all_boxes, all_scores, all_cls, all_coeffs = [], [], [], []

        for box_key, cls_key, coef_key, stride in self.HEADS:
            b, s, c, k = self.decode_head(raw[box_key], raw[cls_key], raw[coef_key], stride)
            mask = s > self.CONF_THRES
            all_boxes.append(b[mask])
            all_scores.append(s[mask])
            all_cls.append(c[mask])
            all_coeffs.append(k[mask])

        boxes   = np.concatenate(all_boxes,   axis=0)
        scores  = np.concatenate(all_scores,  axis=0)
        cls_ids = np.concatenate(all_cls,     axis=0)
        coeffs  = np.concatenate(all_coeffs, axis=0)

        if len(boxes) == 0:
            return []

        keep_all = []
        for cid in np.unique(cls_ids):
            idx  = np.where(cls_ids == cid)[0]
            keep = self.nms(boxes[idx], scores[idx])
            keep_all.extend(idx[keep])
        keep_all = keep_all[:self.MAX_DET]

        if not keep_all:
            return []

        boxes   = boxes[keep_all]
        scores  = scores[keep_all]
        cls_ids = cls_ids[keep_all]
        coeffs  = coeffs[keep_all]

        rles       = self.process_masks(proto, coeffs, boxes, meta, orig_hw)
        boxes_orig = self.unpad_boxes(boxes, meta, orig_hw)

        results = []
        for i in range(len(keep_all)):
            x1, y1, x2, y2 = boxes_orig[i]
            w = float(x2 - x1)
            h = float(y2 - y1)
            if w <= 0 or h <= 0:
                continue
            results.append({
                "image_id"    : img_id,
                "category_id" : coco_ids[int(cls_ids[i])],
                "bbox"        : [float(x1), float(y1), w, h],
                "score"       : float(scores[i]),
                "segmentation": rles[i],
            })

        return results

class YOLOv8PoseDecoder:
    def __init__(self, conf_thres=0.3, iou_thres=0.45):
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.strides = [8, 16, 32]
        self.reg_max = 16
        
        # 하일로 노드 매핑
        self.node_map = {
            8:  {'reg': 'yolov8s_pose/conv43', 'cls': 'yolov8s_pose/conv44', 'kpt': 'yolov8s_pose/conv45'},
            16: {'reg': 'yolov8s_pose/conv57', 'cls': 'yolov8s_pose/conv58', 'kpt': 'yolov8s_pose/conv59'},
            32: {'reg': 'yolov8s_pose/conv70', 'cls': 'yolov8s_pose/conv71', 'kpt': 'yolov8s_pose/conv72'}
        }

    def decode(self, raw_data, meta):
        all_boxes, all_scores, all_kpts = [], [], []

        for stride in self.strides:
            nodes = self.node_map[stride]
            reg = raw_data[nodes['reg']][0]
            cls = raw_data[nodes['cls']][0]
            kpt = raw_data[nodes['kpt']][0]
            
            H, W = reg.shape[:2]
            scores_2d = cls.reshape(H, W) # 하일로 출력은 이미 확률값이므로 sigmoid 생략
            keep = scores_2d > self.conf_thres
            
            if not np.any(keep): continue
            
            idx = np.where(keep)
            grid_y, grid_x = idx[0].astype(np.float32), idx[1].astype(np.float32)
            
            # 1. Bbox 디코딩
            reg_keep = reg[idx[0], idx[1]].reshape(-1, 4, self.reg_max)
            dist = np.dot(softmax(reg_keep, axis=-1), np.arange(self.reg_max))
            
            cx, cy = (grid_x + 0.5) * stride, (grid_y + 0.5) * stride
            x1, y1 = cx - dist[:, 0] * stride, cy - dist[:, 1] * stride
            x2, y2 = cx + dist[:, 2] * stride, cy + dist[:, 3] * stride
            
            # 2. Keypoints 디코딩 (수정된 공식 적용)
            kpt_keep = kpt[idx[0], idx[1]].reshape(-1, 17, 3)
            kpt_keep[..., 0] = (kpt_keep[..., 0] * 2.0 + grid_x[:, None]) * stride
            kpt_keep[..., 1] = (kpt_keep[..., 1] * 2.0 + grid_y[:, None]) * stride
            
            all_boxes.append(np.stack([x1, y1, x2, y2], axis=1))
            all_scores.append(scores_2d[keep])
            all_kpts.append(kpt_keep)

        if not all_boxes: return []
        boxes, scores, kpts = np.concatenate(all_boxes), np.concatenate(all_scores), np.concatenate(all_kpts)
        indices = self.cpu_nms(boxes, scores)
        
        results = []
        for i in indices:
            b, pk = boxes[i], kpts[i]
            x1, y1 = (b[0] - meta['pad_w']) / meta['scale'], (b[1] - meta['pad_h']) / meta['scale']
            w, h = (b[2] - b[0]) / meta['scale'], (b[3] - b[1]) / meta['scale']
            
            formatted_kpts = []
            for kp in pk:
                kx, ky, kv = kp
                rx = (kx - meta['pad_w']) / meta['scale']
                ry = (ky - meta['pad_h']) / meta['scale']
                # 가시성 처리: 로짓값을 COCO 표준(1, 2)으로 변환
                v_flag = 2 if sigmoid(kv) > 0.5 else 1
                formatted_kpts.extend([float(rx), float(ry), int(v_flag)])
            
            results.append({
                "image_id": int(meta['img_id']),
                "category_id": 1,
                "bbox": [float(x1), float(y1), float(w), float(h)],
                "keypoints" : formatted_kpts,
                "score": float(scores[i])
            })
            
        return results

    def cpu_nms(self, boxes, scores):
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]; keep.append(i)
            xx1, yy1 = np.maximum(x1[i], x1[order[1:]]), np.maximum(y1[i], y1[order[1:]])
            xx2, yy2 = np.minimum(x2[i], x2[order[1:]]), np.minimum(y2[i], y2[order[1:]])
            w, h = np.maximum(0.0, xx2 - xx1), np.maximum(0.0, yy2 - yy1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            order = order[np.where(ovr <= self.iou_thres)[0] + 1]
        return keep


def run_inference(combo_name, paths, trial):
    params = hp.VDevice.create_params()
    params.scheduling_algorithm = hp.HailoSchedulingAlgorithm.ROUND_ROBIN
    
    with hp.VDevice(params) as vdevice:
        networks = []
        for p in paths:
            hef = hp.HEF(p)
            name = "DET" if "seg" not in p and "pose" not in p else ("SEG" if "seg" in p else "POSE")
            ng = vdevice.configure(hef)[0]
            #ng.set_scheduler_priority(SCHEDULER_CONFIG[name]["priority"])
            #ng.set_scheduler_threshold(SCHEDULER_CONFIG[name]["threshold"])
            #ng.set_scheduler_timeout(SCHEDULER_CONFIG["TIMEOUT"])
            in_p = hp.InputVStreamParams.make_from_network_group(ng, quantized=True, format_type=hp.FormatType.UINT8)
            out_p = hp.OutputVStreamParams.make_from_network_group(ng, quantized=False, format_type=hp.FormatType.FLOAT32)
            networks.append({"ng": ng, "in": in_p, "out": out_p, "name": hef.get_input_vstream_infos()[0].name, "model_name": name, "postprocessor": POSTPROCESSOR_MAP[name]})

        from contextlib import ExitStack
        with ExitStack() as stack:
            pipelines = [stack.enter_context(hp.InferVStreams(n["ng"], n["in"], n["out"])) for n in networks]
            images = sorted(glob.glob(f"{IMAGE_DIR}/*.jpg"))[:NUM_IMAGES]
            for img_path in images:
                img_id = int(os.path.splitext(os.path.basename(img_path))[0])

                frame, meta = preprocess_letterbox(img_path)
                if frame is None: continue
                meta['img_id'] = img_id

                for i, (pipe, net) in enumerate(zip(pipelines, networks)):
                    #frame, meta = preprocess_letterbox(img_path)
                    #if frame is None: continue
                    #meta['img_id'] = img_id

                    input_data = {net["name"]: np.expand_dims(frame, axis=0)}
                    raw_results  = pipe.infer(input_data)
                    #print(f"[{net['name']}] complete ! {time.time()}")
                    results = net["postprocessor"].decode(raw_results, meta)
                

POSTPROCESSOR_MAP = {
    "DET":  YOLOv8DetDecoder(model_shape=(640, 640), id_map=COCO_ID_MAP),
    "SEG":  YOLOv8SegDecoder(),
    "POSE": YOLOv8PoseDecoder(),
}


if __name__ == "__main__":
    combos = [
        #("DET", [DET_PATH]), ("SEG", [SEG_PATH]), ("POSE", [POSE_PATH]),
        #("DET_SEG", [DET_PATH, SEG_PATH]), ("DET_POSE", [DET_PATH, POSE_PATH]),
        #("SEG_POSE", [SEG_PATH, POSE_PATH]), 
        ("DET_SEG_POSE", [DET_PATH, SEG_PATH, POSE_PATH])
    ]

    for name, paths in combos:
        for t in range(1, 2):
            print(f"Running {name} Trial {t}...")
            start_time = time.time()
            
            p_bench = mp.Process(target=run_inference, args=(name, paths, t))
            p_bench.start()
            p_bench.join()
            
            total_time = time.time()-start_time
            time.sleep(5)

            print(f"Total Latency : {total_time} sec")
            print(f"End to End Latency per image : {((total_time)/NUM_IMAGES)*1000} ms")

    print("ALL DONE.")
