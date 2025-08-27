
------
## Hef 파일 생성 

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



