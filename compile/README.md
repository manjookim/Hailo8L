

## Hef 파일 생성 
- 모델 컴파일은 hailo_ai_sw_suite (Hailo SDK)가 설치된 PC(Host) 에서 수행     
- Hailo SDK / Hailomz / Degirum 세 가지 방법 중 골라서 사용    

### 1. Hailo SDK 활용

onnx 파일 생성 
```
yolo export model=yolov8n-seg.pt format=onnx imgsz=320
```
Parsing 
```   
hailo parser onnx /home/mjss/Downloads/yolo_new/yolov8n-seg.onnx \       
#	--start-node-names images \           
#	--end-node-names output0 output1 \          
	--hw-arch hailo8l
```
Optimization
```
hailo optimize yolov8n-seg.har --calib-set-path /home/mjss/Downloads/yolo_new/calib_data.npy
```
Compile 
```
hailo compiler yolov8n-seg_optimized.har --hw-arch hailo8l            
```


### 2. Hailo model zoo 활용
onnx 파일 생성   
```
yolo export model=yolov8n-seg.pt format=onnx imgsz=320
```
Compile 
```
hailomz compile --ckpt /home/mjss/Downloads/yolo_new/yolov8n.onnx \ # 컴파일하고 싶은 모델의 onnx 경로
 --calib-path /home/mjss/Downloads/yolo_new/expanded_coco_images \  # 해당 모델의 학습데이터 경로
 --yaml /home/mjss/Downloads/yolo_new/yolov8n-det.yaml \            # 모델의 컴파일 정보가 담긴 yaml 파일 경로 (hailo model zoo 참고)
 --hw-arch hailo8l                                                  # hailo 하드웨어 버전 명시  
```

### 3. Degirum 활용 
