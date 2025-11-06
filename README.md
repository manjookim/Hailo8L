# Hailo8L
Raspberry pi OS + Hailo 8L + AI inference 

Raspberry pi OS : debian bookworm           
Raspberry Pi version : Raspberry pi 5              
Hailo NPU version : Hailo-8L            
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

####  호스트/rpi 나눠서 한 이유   


------
### 1. PC 환경설정

1. [Hailo.ai]( 회원가입 및 hailo ai sw suite 설치
hailo.ai > developer zone > software downloads > hailo_ai_sw_suite_2025-10.run 다운로드
```
sudo apt update
sudo apt install -y python3-tk graphviz libgraphviz-dev python3.10-dev build-essential cmake libusb-1.0-0-dev
sudo apt install -y python3-pip
python3 -m pip install virtualenv
sudo apt install -y nodejs npm
./hailo8_ai_sw_suite_2025-07.run
```

2. hailo model zoo 깃허브 클론
```
git clone https://github.com/hailo-ai/hailo_model_zoo
```

3.  가상환경 생성 및 접속
```
cd hailo_model_zoo
sudo apt intsall python3.10-venv
python3 -m venv hailo_custom_venv
source hailo_custom_venv/bin/activate
```

4. dataflow compiler , hailort 설치
```
pip install /home/mjss/Downloads/hailo_ai_sw_suite/artifacts/hailo_dataflow_compiler-3.32.0-py3-none-linux_x86_64.whl
pip install /home/mjss/Downloads/hailo_ai_sw_suite/artifacts/hailort-4.22.0-cp310-cp310-linux_x86_64.whl
pip install -e .
```




### 2. Rpi 환경설정
1. Dockerfile 작성
2. 

------

1. .HEf 컴파일     
   See  [Model_Compilation](https://github.com/manjookim/Hailo8L/tree/main/Model_compilation/README.md) for more details
2. Inference                   
3. monitoring
4. accuracy                




------
### References
https://github.com/hailo-ai/hailort       
https://github.com/hailo-ai/Hailo-Application-Code-Examples        
https://github.com/hailo-ai/tappas     
https://github.com/hailo-ai/hailo_model_zoo    
https://hailo.ai/developer-zone/documentation/
