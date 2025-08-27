# -*- coding: utf-8 -*-

import os
import time
import glob
import numpy as np
from PIL import Image
import hailo_platform as hp
import argparse  # 1. argparse 라이브러리를 추가합니다.

# 2. 메인 로직을 함수로 감싸서 인자를 받을 수 있도록 합니다.
def run_benchmark(hef_path: str, image_dir: str, num_images: int, num_runs: int):
    """
    Hailo HEF 모델의 추론 성능을 벤치마킹하고 통계를 출력합니다.
    """
    print("--- Benchmark Configuration ---")
    print(f"HEF File: {hef_path}")
    print(f"Image Dir: {image_dir}")
    print(f"Images per run: {num_images}")
    print(f"Number of runs: {num_runs}")
    print("---------------------------------")
    
    # List of image paths
    # jpg 외에 png 등 다른 확장자도 처리하려면 glob 패턴을 수정하세요. (예: *.[jp][pn]g)
    image_paths = sorted(glob.glob(f"{image_dir}/*.*"))[:num_images]
    if not image_paths:
        raise FileNotFoundError(f"No images found in {image_dir}")

    # Verify that the HEF file path is correct
    if not os.path.exists(hef_path):
        raise FileNotFoundError(f"HEF file not found at: {hef_path}")

    # --- 이하 로직은 기존 스크립트와 거의 동일합니다 ---

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
            # ... (카운트다운 부분은 가독성을 위해 생략, 필요 시 추가하세요) ...

            print("\n" + "="*30)
            print(">>> INFERENCE START! <<<")
            print("="*30 + "\n")

            # Loop for multiple runs
            for run_count in range(num_runs):
                print(f"\n--- Starting inference run {run_count+1}/{num_runs} ---")
                processing_times = []

                # Infer each image individually with batch size 1
                for img_np in image_data_list:
                    start_time = time.time()
                    input_data = {input_vinfo.name: np.expand_dims(img_np, axis=0)}
                    _ = infer_pipeline.infer(input_data)
                    end_time = time.time()
                    duration = end_time - start_time
                    processing_times.append(duration)
                
                current_run_time = sum(processing_times)
                current_run_std = np.std(processing_times)
                total_inference_time_sum += current_run_time

                if processing_times:
                    num_processed = len(processing_times)
                    avg_time_per_image = current_run_time / num_processed
                    fps = 1 / avg_time_per_image
                    print(f"Run {run_count+1}: Total images processed: {num_processed}")
                    print(f"Run {run_count+1}: Total inference time: {current_run_time * 1000:.2f} ms")
                    print(f"Run {run_count+1}: Average inference time per image: {avg_time_per_image * 1000:.2f} ms")
                    print(f"Run {run_count+1}: std : {current_run_std * 1000:.2f}ms")
                    print(f"Run {run_count+1}: Average FPS: {fps:.2f}")

        # Calculate and print overall average time
        if num_runs > 0 and num_images > 0:
            overall_avg_total_time = total_inference_time_sum / num_runs
            overall_avg_time_per_image = overall_avg_total_time / num_images
            overall_avg_fps = 1 / overall_avg_time_per_image

            print("\n" + "=" * 50)
            print(f"Overall Results for {num_runs} runs:")
            print(f"Average Total Inference Time for {num_images} images: {overall_avg_total_time * 1000:.2f} ms")
            print(f"Average Inference Time per Image: {overall_avg_time_per_image * 1000:.2f} ms")
            print(f"Overall Average FPS: {overall_avg_fps:.2f}")

# 3. 스크립트가 직접 실행될 때 인자를 파싱하고 함수를 호출하는 부분을 추가합니다.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generic Hailo Inference Benchmark Tool")
    parser.add_argument("--hef", type=str, required=True, help="Path to the .hef model file.")
    parser.add_argument("--image-dir", type=str, required=True, help="Path to the directory containing images.")
    parser.add_argument("--num-images", type=int, default=100, help="Number of images to infer per run.")
    parser.add_argument("--num-runs", type=int, default=5, help="Number of times to repeat the benchmark.")
    args = parser.parse_args()

    # 파싱된 인자를 사용하여 벤치마크 함수를 실행합니다.
    run_benchmark(
        hef_path=args.hef,
        image_dir=args.image_dir,
        num_images=args.num_images,
        num_runs=args.num_runs
    )
