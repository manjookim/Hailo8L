Scheduling 



### HRTT 파일 생성
추론하는 터미널에서 설정
```
export HAILO_TRACE=scheduler #안하면 기록 안됨 
export HAILO_TRACE_TIME_IN_SECONDS_BOUNDED_DUMP=30 #값 조정은 자유 
export HAILO_TRACE_PATH=/to/your/path
export HAILO_MONITOR=1
```
(참고) dump값이 크거나 지정값이 없으면 fps 성능 저하 문제 발생 


### HRTT Profiling
Dataflow Compiler가 설치된 환경에서 실행
```
hailo runtime-profiler <.hrtt file>
```


< hrtt 파일 예시 >
<img width="1807" height="1301" alt="image" src="https://github.com/user-attachments/assets/2b9d8f1f-6ddd-4a8e-b15b-129d0fdc02bb" />

