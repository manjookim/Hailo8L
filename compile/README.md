

## Hef 파일 생성 
- 모델 컴파일은 hailo_ai_sw_suite (Hailo SDK)가 설치된 PC(Host) 에서 수행     
- Hailo SDK / Hailomz / Degirum 세 가지 방법 중 골라서 사용

<br>

### 1. Hailo SDK 활용

1-1. onnx 파일 생성 
```
yolo export model=yolov8n-seg.pt format=onnx
#yolo export model=yolov8n-seg.pt format=onnx imgsz=320
```
메모리 오류 날 시 , imgsz 줄여서 해결   
1-2. Parsing 
```   
hailo parser onnx /home/mjss/Downloads/yolo_new/yolov8n-seg.onnx \       
#	--start-node-names images \           
#	--end-node-names output0 output1 \          
	--hw-arch hailo8l
```
1-3. Optimization
```
hailo optimize yolov8n-seg.har --calib-set-path /home/mjss/Downloads/yolo_new/calib_data.npy
```
1-4. Compile 
```
hailo compiler yolov8n-seg_optimized.har --hw-arch hailo8l            
```

<br>

### 2. Hailo model zoo 활용
2-1. onnx 파일 생성   
```
yolo export model=yolov8n-seg.pt format=onnx 
```
2-2. 모델 컴파일 정보가 담긴 yaml, alls 파일 작성 (필요에 따라 다름)        
[https://github.com/hailo-ai/hailo_model_zoo/tree/master/hailo_model_zoo/cfg](https://github.com/hailo-ai/hailo_model_zoo/tree/master/hailo_model_zoo/cfg) 참고하여 필요한 파일들 수정 및 사용 

2-3. Compile 
```
hailomz compile --ckpt /home/mjss/Downloads/yolo_new/yolov8n.onnx \ # 컴파일하고 싶은 모델의 onnx 경로
 --calib-path /home/mjss/Downloads/yolo_new/expanded_coco_images \  # 해당 모델의 학습데이터 경로
 --yaml /home/mjss/Downloads/yolo_new/yolov8n-det.yaml \            # 모델의 컴파일 정보가 담긴 yaml 파일 경로 (hailo model zoo 참고)
 --hw-arch hailo8l                                                  # hailo 하드웨어 버전 명시  
```

<br>

### 3. Degirum 활용 
3-1. [Degirum](https://hub.degirum.com/compiler) 회원가입 및 compiler 선택     
3-2. Compile 
- 컴파일할 모델의 pt 파일
- name, version, imgsz
- runtime : hailort
- device : hailo8l
- advanced options : calib-data(학습 데이터 업로드)     
<img width="962" height="1300" alt="image" src="https://github.com/user-attachments/assets/c2a6c673-a47b-4d98-b941-a54cce7a63a9" />

