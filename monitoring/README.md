## HAILO NPU 성능 측정

### ProtoType 설정
1. 도커내부에 prototype 설치
```
sudo apt install protobuf-compiler
protoc --version

pip install protobuf
```

2. proto 파일을 파이썬으로 변환 후 이동
```
git clone https://github.com/hailo-ai/hailort
cd hailort/hailort/libhailort

protoc --python_out=. scheduler_mon.proto

mv scheduler_mon_pb2.py /app/tappas/npu/
```

3. npu 사용량 측정
- 터미널 1 (추론)
  ```
  # 1. 환경 변수 설정
  export HAILO_MONITOR=1
  # 2. NPU 작업 실행
  python3 npu-det.py
  또는 #hailortcli run yolov8n-det.hef 
  ```
- 터미널 2 (측정)
```
python3 hailo_utilization.py 
#옵션 필요할 때 사용 
#--update_period 0.1
```
