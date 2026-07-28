1. 코드 작성
```
nano main.cpp
```
- `main.cpp` : pre-proces, inference,  post-process (E2E)

2. 컴파일
```
sudo g++ -O3 main.cpp -o mps_cpp \
    `pkg-config --cflags --libs opencv4` \
    -lhailort -pthread
```
3. 코드 실행
```
./mps_cpp
```
