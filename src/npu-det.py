import os
import time
import gi

gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

Gst.init(None)

hef_path = "/app/tappas/npu/yolov8n-det.hef"
image_dir = "/app/tappas/npu/expanded_coco_images"

total_pred_time_ms = 0
num_images = 0

try:
    image_files = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
    if not image_files:
        raise FileNotFoundError(f"No image files found in {image_dir}")
except FileNotFoundError as e:
    print(f"Error: {e}")
    exit(1)

for image_path in image_files:
    pipeline_str = (
        f"filesrc location={image_path} ! jpegdec ! videoconvert ! "
        "videoscale ! video/x-raw,format=RGB,width=640,height=640 ! "
        f"hailonet hef-path={hef_path} ! fakesink"
    )
    pipeline = Gst.parse_launch(pipeline_str)

    bus = pipeline.get_bus()

    pipeline.set_state(Gst.State.PLAYING)

    pred_time_start = time.time()

    msg = bus.timed_pop_filtered(Gst.CLOCK_TIME_NONE, Gst.MessageType.ERROR | Gst.MessageType.EOS)

    pred_time_end = time.time()

    if msg:
        if msg.type == Gst.MessageType.EOS:
            elapsed_time_ms = (pred_time_end - pred_time_start) * 1000  # ms로 변환
            total_pred_time_ms += elapsed_time_ms
            num_images += 1
            print(f"Processed {image_path} in {elapsed_time_ms:.2f} ms")
        elif msg.type == Gst.MessageType.ERROR:
            err, debug_info = msg.parse_error()
            print(f"Error processing {image_path}: {err.message}")
            if debug_info:
                print(f"Debug info: {debug_info}")
            break
    else:
        print(f"Pipeline timed out processing {image_path}. Exiting.")
        break

    pipeline.set_state(Gst.State.NULL)

if num_images > 0:
    avg_pred_time_ms = total_pred_time_ms / num_images
    fps = 1000 / avg_pred_time_ms  # 1초(1000ms) 나누기 평균 ms
    print("-" * 30)
    print(f"Total images processed: {num_images}")
    print(f"Total prediction time: {total_pred_time_ms:.2f} ms")
    print(f"Average prediction time per image: {avg_pred_time_ms:.2f} ms")
    print(f"Average FPS: {fps:.2f}")
