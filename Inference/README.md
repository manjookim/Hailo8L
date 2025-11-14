## DL Inference Workload

컨테이너 진입
```
docker exec -it <container_name> /bin/bash
```
가상환경 진입
```
source <venv_name>/bin/activate
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
