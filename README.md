# Hailo8L
Raspberry pi OS + Hailo 8L + AI inference 

Raspberry pi OS : debian bookworm           
Raspberry Pi version : Raspberry pi 5              
Hailo NPU version : Hailo-8L     
Dataflow Compiler version : v3.32.0    
Hailo Model Zoo version : v2.15    
PcIe & Hailort version : 4.22.0            
Python version : 3.11

------

####  NPU (Nenral Processing Unit)    
인공지능(AI)과 딥러닝 연산을 가속화하기 위해 특수하게 설계된 하드웨어 프로세서    
+ NPU는 고성능·저전력으로 설계되어 스마트폰, PC, 자율주행차 등 다양한 엣지 디바이스에서 AI 기능을 효율적으로 처리
+ CPU나 GPU의 부담을 줄여 전력 효율을 높이고 발열을 감소
####  HAILO란
https://hailo.ai/     

####  HAILO에 사용되는 프레임워크        
+ Hailort      
+ Hailort python binding        
+ tappas       
+ hailo model zoo       
+ hailo ai sw suite        

####  호스트(PC)/rpi 둘 다 사용 한 이유   
raspberry pi 의 OS는 debian 버전으로, hailo 에서 제공하는 툴들이 호환되지 않음 ->      
컴파일은 호스트에서, 추론은 rpi의 커스텀 도커 컨테이너에서 실행  

## PC 환경설정
- Hailo SDK (DFC) 설치     
hailo8_ai_sw_suite_2025-10.run 다운로드
```



## Rpi 환경설정

드라이버의 버전은 모두 같아야 함   
  
1. pcie 드라이버 설치
```
sudo dpkg -i hailort-pcie-driver_4.22.0_all.deb 
sudo reboot
```

2. [Dockerfile](https://github.com/manjookim/Hailo8L/blob/main/Dockerfile) 작성 및 저장        
```
vi Dockerfile
```

3. 도커 컨테이너 생성
```
sudo docker build -t hailo_docker:test .
#docker image:version
```
```
sudo docker run -it \
  --privileged \
  --ipc=host \
  -v /dev:/dev \
  -v /lib/modules:/lib/modules:ro \
  -v /usr/src:/usr/src:ro \
  -v /dev/bus/pci:/dev/bus/pci \
  -e HAILO_MONITOR=1 \
  -v /home/rpi2/npu:/app/tappas/npu \#-v 마운트하고싶은 로컬 디렉토리 경로:마운트할 도커 디렉토리 경로
  --name hailo_test hailo_docker:test /bin/bash #docker name : hailo_test (사용자 임의 변경)
```

4. 도커 컨테이너 진입
```
docker start hailo_test
docker exec -it hailo_test /bin/bash
```

------

1. .HEf 컴파일     
   See  [Compile](https://github.com/manjookim/Hailo8L/tree/main/compile/README.md) for more details
2. Inference
    See  [inference](https://github.com/manjookim/Hailo8L/tree/main/Inference/README.md) for more details                   
3. monitoring
    See  [Compile](https://github.com/manjookim/Hailo8L/tree/main/compile/README.md) for more details
4. accuracy
   See  [accuracy](https://github.com/manjookim/Hailo8L/tree/main/accuracy/README.md) for more details               




------
### References
https://github.com/hailo-ai/hailort       
https://github.com/hailo-ai/Hailo-Application-Code-Examples        
https://github.com/hailo-ai/tappas     
https://github.com/hailo-ai/hailo_model_zoo    
https://hailo.ai/developer-zone/documentation/
https://docs.ultralytics.com/ko/models/yolov8/
