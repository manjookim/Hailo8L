# -*- coding: utf-8 -*-

import time
import glob
import numpy as np
from PIL import Image
import hailo_platform as hp

hef_path = "yolov8n-obb.hef"
image_dir = "expanded_dota_images"
num_images = 250

# 이미지 경로 리스트
image_paths = sorted(glob.glob(f"{image_dir}/*.png"))[:num_images]
if not image_paths:
    raise FileNotFoundError(f"{image_dir}에 PNG 이미지가 없습니다.")

hef = hp.HEF(hef_path)
params = hp.VDevice.create_params()
params.scheduling_algorithm = hp.HailoSchedulingAlgorithm.NONE

with hp.VDevice(params) as vdevice:
    configure_params = hp.ConfigureParams.create_from_hef(hef, interface=hp.HailoStreamInterface.PCIe)
    network_group = vdevice.configure(hef, configure_params)[0]

    input_vinfo = hef.get_input_vstream_infos()[0]
    h, w, c = input_vinfo.shape

    print("Input vstream info:")
    print(f"Name: {input_vinfo.name}")
    print(f"Shape: {input_vinfo.shape}")

    # 이미지 하나 크기 (bytes)
    single_frame_size = w * h * c  # 예: 320*320*3 = 307200 bytes
    # Hailo NPU가 기대하는 입력 크기 (bytes)
    expected_size = 98304000  # 에러 메시지 참고

    frame_count = expected_size // single_frame_size
    print(f"Expected frame count (batch size): {frame_count}")

    # 이미지 데이터 배치용 배열 생성 (batch_size, H, W, C)
    batch_images = []

    for img_path in image_paths:
        img = Image.open(img_path).convert("RGB").resize((w, h))
        img_np = np.array(img, dtype=np.uint8)
        batch_images.append(img_np)

    # 단일 이미지가 1장 이상인데 batch_count가 더 크면 복제
    if len(batch_images) < frame_count:
        batch_images = batch_images * (frame_count // len(batch_images) + 1)

    batch_images = np.array(batch_images[:frame_count])  # shape: (frame_count, H, W, C)

    input_vs_params = hp.InputVStreamParams.make_from_network_group(network_group, quantized=False)
    output_vs_params = hp.OutputVStreamParams.make_from_network_group(network_group, quantized=False)

    with network_group.activate(network_group.create_params()):
        with hp.InferVStreams(network_group, input_vs_params, output_vs_params) as infer_pipeline:
            start_time = time.time()

            input_data = {input_vinfo.name: batch_images}
            _ = infer_pipeline.infer(input_data)

            total_time = time.time() - start_time

    print(f"총 추론 시간 (batch size {frame_count}): {total_time * 1000:.1f} ms")
    print(f"이미지당 평균 추론 시간: {total_time / frame_count * 1000:.2f} ms")
