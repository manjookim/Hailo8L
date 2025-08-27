# Hailo8L
Raspberry pi OS + Hailo 8L + AI inference 

Raspberry pi OS : debian bookworm           
Raspberry Pi version : Raspberry pi 5              
Hailo NPU version : Hailo-8L            
PcIe & Hailort version : 4.22.0            
Python version : 3.11

------

1. NPU란
2. HAILO란
3. HAILO에 사용되는 프레임워크
4. 호스트/rpi 나눠서 한 이유


1. HEf file
2. Inference
3. monitoring


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

------
### References
https://github.com/hailo-ai/hailort       
https://github.com/hailo-ai/Hailo-Application-Code-Examples        
https://github.com/hailo-ai/tappas     
https://github.com/hailo-ai/hailo_model_zoo    
https://hailo.ai/developer-zone/documentation/
