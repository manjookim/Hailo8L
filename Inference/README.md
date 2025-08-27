## DL Inference Workload

도커파일 빌드    
```
docker build -t my-hailo-tappas:final .
```

HAILO8L 이라는 이름의 컨테이너 생성     
```
docker run -it --privileged --device=/dev/hailo0 --ipc=host -v /tmp:/tmp \
-v /home/rpi2/npu:/app/tappas/npu \
-e HAILO_MONITOR=1 \
--name HAILO8L \
my-hailo-tappas:final
```

CPU DL Inference    
```
python3 cpu_infer.py
```

NPU DL Inference
```
python3 npu_infer.py
```


benchmark 실행    
```
python benchmark.py --hef ./640/yolov8n-det.hef --image-dir ./640/expanded_coco_images
```
