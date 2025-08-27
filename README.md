# Hailo8L
Raspberry pi OS + Hailo 8L + AI inference 

Raspberry pi OS : debian bookworm           
Raspberry Pi version : Raspberry pi 5              
Hailo NPU version : Hailo-8L            
PcIe & Hailort version : 4.22.0            
Python version : 3.11

------
도커파일 빌드    
```
docker build -t my-hailo-tappas:final .
```

hailo8l 이라는 이름의 컨테이너 생성     
```
docker run -it --name hailo8l   --device=/dev/hailo0   -v /home/rpi2/npu:/app/tappas/npu   my-hailo-tappas:final /bin/bash
```

------
### References
https://github.com/hailo-ai/hailort       
https://github.com/hailo-ai/Hailo-Application-Code-Examples        
https://github.com/hailo-ai/tappas     
https://github.com/hailo-ai/hailo_model_zoo    
https://hailo.ai/developer-zone/documentation/
