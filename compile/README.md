

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

### 3. Degirum 활용 
