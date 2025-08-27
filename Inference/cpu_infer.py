# -*- coding: utf-8 -*-

import os
import time
import glob
import numpy as np
from PIL import Image
import onnxruntime as ort # ONNX 런타임 라이브러리

# 필요한 라이브러리 설치:
# pip install onnxruntime numpy Pillow

# Set the number of inference runs for better reliability.
num_runs = 5

# Model and image paths for classification
# ------------------------------------------------------------------- #
# TODO: 사용하실 ONNX 모델 경로와 이미지 디렉토리 경로를 설정해주세요.
onnx_path = "./320/yolov8n-obb.onnx"
image_dir = "./320/expanded_dota_320"
num_images = 100
# ------------------------------------------------------------------- #

# List of image paths
image_paths = sorted(glob.glob(f"{image_dir}/*.png"))[:num_images]
if not image_paths:
    raise FileNotFoundError(f"No JPG images found in {image_dir}")

# Verify that the ONNX file path is correct
if not os.path.exists(onnx_path):
    raise FileNotFoundError(f"ONNX file not found at: {onnx_path}")

# ONNX 런타임 세션 생성 (CPU 사용)
session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

# 모델의 입력 정보 가져오기
input_name = session.get_inputs()[0].name
_, _, h, w = session.get_inputs()[0].shape  # (batch_size, channels, height, width)

print("Input info:")
print(f"Name: {input_name}")
print(f"Shape: (1, 3, {h}, {w})") # ONNX는 보통 NCHW 형식을 사용합니다.

# List to hold pre-processed image data
image_data_list = []
print(f"\nPreprocessing {len(image_paths)} images...")
for img_path in image_paths:
    img = Image.open(img_path).convert("RGB").resize((w, h))
    img_np = np.array(img, dtype=np.float32)
    # 이미지 전처리: HWC -> CHW 및 0-1 정규화
    img_np = img_np / 255.0
    img_np = img_np.transpose(2, 0, 1)  # (height, width, channels) -> (channels, height, width)
    img_np = np.expand_dims(img_np, axis=0) # 배치 차원 추가 (1, C, H, W)
    image_data_list.append(img_np)
print("Preprocessing complete.")

total_inference_time_sum = 0.0

print("\nCPU inference is about to start. Press Ctrl-C on this terminal to stop.")
print("Ready... Starting inference in 3 seconds.")
time.sleep(1)
print("2...")
time.sleep(1)
print("1...")
time.sleep(1)

# Print a clear start signal to the screen
print("\n" + "="*30)
print(">>> INFERENCE START! <<<")
print("="*30 + "\n")

# Loop for multiple runs
for run_count in range(num_runs):
    print(f"\n--- Starting inference run {run_count+1}/{num_runs} ---")
    
    # ▼▼▼ [수정됨] 개별 추론 시간을 저장할 리스트 초기화 ▼▼▼
    processing_times = []

    # Infer each image individually with batch size 1
    num_processed = 0
    for img_np in image_data_list:
        # ▼▼▼ [수정됨] 개별 이미지 추론 시간 측정 시작 ▼▼▼
        start_time = time.time()
        
        # ONNX 런타임으로 추론 실행
        _ = session.run(None, {input_name: img_np})
        num_processed += 1
        
        # ▼▼▼ [수정됨] 개별 이미지 추론 시간 측정 종료 및 저장 ▼▼▼
        end_time = time.time()
        duration = end_time - start_time
        processing_times.append(duration)
    
    # ▼▼▼ [수정됨] 통계 계산 로직 변경 ▼▼▼
    current_run_time = sum(processing_times)
    current_run_std = np.std(processing_times)
    total_inference_time_sum += current_run_time

    if num_processed > 0:
        avg_time_per_image = current_run_time / num_processed
        fps = 1 / avg_time_per_image
        print(f"Run {run_count+1}: Total images processed: {num_processed}")
        print(f"Run {run_count+1}: Total inference time: {current_run_time * 1000:.2f} ms")
        print(f"Run {run_count+1}: Average inference time per image: {avg_time_per_image * 1000:.2f} ms")
        # ▼▼▼ [추가됨] 표준편차 출력 ▼▼▼
        print(f"Run {run_count+1}: std : {current_run_std * 1000:.2f}ms")
        print(f"Run {run_count+1}: Average FPS: {fps:.2f}")
        # ▼▼▼ [추가됨] 구분선 ▼▼▼
        print("------------------------------------------")

# Calculate and print overall average time
if num_runs > 0:
    overall_avg_total_time = total_inference_time_sum / num_runs
    overall_avg_time_per_image = overall_avg_total_time / num_images
    overall_avg_fps = 1 / overall_avg_time_per_image

    print("\n" + "=" * 50)
    # ▼▼▼ [수정됨] 출력 제목 변경 ▼▼▼
    print(f"Overall Results for {num_runs} runs:")
    print(f"Average Total Inference Time for {num_images} images: {overall_avg_total_time * 1000:.2f} ms")
    print(f"Average Inference Time per Image: {overall_avg_time_per_image * 1000:.2f} ms")
    print(f"Overall Average FPS: {overall_avg_fps:.2f}")
