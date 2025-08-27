# -*- coding: utf-8 -*-

import os
import time
import glob
import numpy as np
from PIL import Image
import hailo_platform as hp

# Set the number of inference runs for better reliability.
num_runs = 100
processing_times = []

# Model and image paths for classification
hef_path = "./320/yolov8n-obb.hef"
image_dir = "./320/expanded_dota_320"
num_images = 100

# List of image paths
image_paths = sorted(glob.glob(f"{image_dir}/*.png"))[:num_images]
if not image_paths:
    raise FileNotFoundError(f"No JPG images found in {image_dir}")

# Verify that the HEF file path is correct
if not os.path.exists(hef_path):
    raise FileNotFoundError(f"HEF file not found at: {hef_path}")

hef = hp.HEF(hef_path)
params = hp.VDevice.create_params()
params.scheduling_algorithm = hp.HailoSchedulingAlgorithm.ROUND_ROBIN

with hp.VDevice(params) as vdevice:
    configure_params = hp.ConfigureParams.create_from_hef(hef, interface=hp.HailoStreamInterface.PCIe)
    network_group = vdevice.configure(hef, configure_params)[0]

    input_vinfo = hef.get_input_vstream_infos()[0]
    h, w, c = input_vinfo.shape

    print("Input vstream info:")
    print(f"Name: {input_vinfo.name}")
    print(f"Shape: {input_vinfo.shape}")

    # List to hold image data
    image_data_list = []
    for img_path in image_paths:
        img = Image.open(img_path).convert("RGB").resize((w, h))
        img_np = np.array(img, dtype=np.uint8)
        image_data_list.append(img_np)

    input_vs_params = hp.InputVStreamParams.make_from_network_group(network_group, quantized=False)
    output_vs_params = hp.OutputVStreamParams.make_from_network_group(network_group, quantized=False)

    total_inference_time_sum = 0.0

    with hp.InferVStreams(network_group, input_vs_params, output_vs_params) as infer_pipeline:
        print("\nNPU inference is about to start. Press Ctrl+C on this terminal to stop.")
        print("You can now check the NPU utilization in another terminal with `python3 hailo_exporter.py`")
        
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
            processing_times = []

            # Infer each image individually with batch size 1
            num_processed = 0
            for img_np in image_data_list:
                start_time = time.time()
                input_data = {input_vinfo.name: np.expand_dims(img_np, axis=0)}
                _ = infer_pipeline.infer(input_data)
                num_processed += 1
                end_time = time.time()
                duration = end_time - start_time
                processing_times.append(duration)
            
            
            current_run_time = sum(processing_times)
            current_run_std = np.std(processing_times)
            total_inference_time_sum += current_run_time

            if num_processed > 0:
                avg_time_per_image = current_run_time / num_processed
                fps = 1 / avg_time_per_image
                print(f"Run {run_count+1}: Total images processed: {num_processed}")
                print(f"Run {run_count+1}: Total inference time: {current_run_time * 1000:.2f} ms")
                print(f"Run {run_count+1}: Average inference time per image: {avg_time_per_image * 1000:.2f} ms")
                print(f"Run {run_count+1}: std : {current_run_std * 1000:.2f}ms")
                print(f"Run {run_count+1}: Average FPS: {fps:.2f}")

    # Calculate and print overall average time
    if num_runs > 0:
        overall_avg_total_time = total_inference_time_sum / num_runs
        overall_avg_time_per_image = overall_avg_total_time / num_images
        overall_avg_fps = 1 / overall_avg_time_per_image

        print("\n" + "=" * 50)
        print(f"Overall Results for {num_runs} runs:")
        print(f"Average Total Inference Time: {overall_avg_total_time * 1000:.2f} ms")
        print(f"Average Inference Time per Image: {overall_avg_time_per_image * 1000:.2f} ms")
        print(f"Overall Average FPS: {overall_avg_fps:.2f}")
