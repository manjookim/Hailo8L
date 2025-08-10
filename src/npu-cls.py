import os
import time
import gi
import numpy as np

gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

Gst.init(None)

hef_path = "/app/tappas/npu/yolov8n-cls.hef"
image_dir = "/app/tappas/npu/expanded_imagenet_images"
num_images = 250

try:
    image_files = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir)
                          if f.endswith(('.jpg', '.jpeg', '.png'))])[:num_images]
    if not image_files:
        raise FileNotFoundError(f"No image files found in {image_dir}")
except FileNotFoundError as e:
    print(f"Error: {e}")
    exit(1)

total_pred_time = 0
num_processed = 0

for image_path in image_files:
    pipeline_str = (
        f"filesrc location={image_path} ! jpegdec ! videoconvert ! "
        "videoscale ! video/x-raw,format=RGB,width=640,height=640 ! "
        f"hailonet hef-path={hef_path} ! "
        "fakesink sync=true"
    )

    pipeline = Gst.parse_launch(pipeline_str)

    bus = pipeline.get_bus()
    pipeline.set_state(Gst.State.PLAYING)

    start_time = time.time()

    msg = bus.timed_pop_filtered(Gst.CLOCK_TIME_NONE, Gst.MessageType.ERROR | Gst.MessageType.EOS)

    end_time = time.time()

    pipeline.set_state(Gst.State.NULL)

    if msg and msg.type == Gst.MessageType.EOS:
        elapsed_time_ms = (end_time - start_time) * 1000
        total_pred_time += elapsed_time_ms
        num_processed += 1
        print(f"Processed {os.path.basename(image_path)} in {elapsed_time_ms:.2f} ms")
    elif msg and msg.type == Gst.MessageType.ERROR:
        err, debug_info = msg.parse_error()
        print(f"Error processing {image_path}: {err.message}")
        if debug_info:
            print(f"Debug info: {debug_info}")
        break

if num_processed > 0:
    avg_time_ms = total_pred_time / num_processed
    fps = 1000 / avg_time_ms
    print("\n" + "-" * 50)
    print(f"Total images processed: {num_processed}")
    print(f"Total prediction time: {total_pred_time:.2f} ms")
    print(f"Average prediction time per image: {avg_time_ms:.2f} ms")
    print(f"Average FPS: {fps:.2f}")
