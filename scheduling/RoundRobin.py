(hailo_venv2) root@rpi2:/app/tappas/rpi2/npu/monitoring_new/scheduling/online# cat test.py
# -*- coding: utf-8 -*-
import multiprocessing as mp
import concurrent.futures  # 🌟 추가된 모듈: 스레드 풀 관리를 위해 사용
import hailo_platform as hp
import numpy as np
import cv2 as cv
import glob, os, time, csv, re, datetime, sys
import threading
from scheduler_mon_pb2 import ProtoMon

DET_PATH  = "/app/tappas/rpi2/npu/yolov8s/yolov8s.hef"
SEG_PATH  = "/app/tappas/rpi2/npu/yolov8s/yolov8s_seg.hef"
POSE_PATH = "/app/tappas/rpi2/npu/yolov8s/yolov8s_pose.hef"

IMAGE_DIR = "/app/tappas/rpi2/npu/accuracy/coco/images/val2017"
NUM_IMAGES = 100
LATENCY_CSV = "latency_results_scheduling.csv"

class HailoMonitor:
    def __init__(self, directory="/tmp/hmon_files"):
        self.directory = directory
        self.data = {}
    def get_single_file(self):
        try:
            files = os.listdir(self.directory)
            return os.path.join(self.directory, files[0]) if files else None
        except: return None
    def read_stats(self):
        try:
            path = self.get_single_file()
            if path and os.path.exists(path):
                proto = ProtoMon()
                with open(path, "rb") as f:
                    proto.ParseFromString(f.read())
                if proto.device_infos:
                    self.data["utilization"] = proto.device_infos[0].utilization
                else: self.data["utilization"] = 0.0
            else: self.data["utilization"] = 0.0
        except: self.data["utilization"] = 0.0
        return self.data

def get_cpu_mem_stats():
    try:
        with open('/proc/meminfo', 'r') as f:
            m = f.read()
            tot = int(re.search(r'MemTotal:\s+(\d+)', m).group(1))
            avail = int(re.search(r'MemAvailable:\s+(\d+)', m).group(1))
        with open('/proc/stat', 'r') as f:
            cpu = [int(x) for x in f.readline().split()[1:]]
        return {'tot': sum(cpu), 'idle': cpu[3], 'mem': 100 * (1 - avail/tot)}
    except: return None

def preprocess_letterbox(img_path, target_shape=(640, 640)):
    img = cv.imread(img_path)
    if img is None: return None
    h, w = img.shape[:2]
    r = min(target_shape[0] / h, target_shape[1] / w)
    nh, nw = int(h * r), int(w * r)
    img = cv.resize(img, (nw, nh), interpolation=cv.INTER_LINEAR)
    pad = np.full((target_shape[0], target_shape[1], 3), 114, dtype=np.uint8)
    pad[(target_shape[0]-nh)//2:(target_shape[0]-nh)//2+nh, (target_shape[1]-nw)//2:(target_shape[1]-nw)//2+nw] = img
    return np.expand_dims(cv.cvtColor(pad, cv.COLOR_BGR2RGB), axis=0)

def monitor_resource(output_file, stop_event):
    npu_mon = HailoMonitor()
    prev = get_cpu_mem_stats()
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Time', 'NPU_util', 'CPU_util', 'Mem_util'])
        
        while not stop_event.is_set():
            time.sleep(0.5)
            npu_util = npu_mon.read_stats().get("utilization", 0.0)
            
            curr = get_cpu_mem_stats()
            if curr and prev:
                diff_tot = curr['tot'] - prev['tot']
                diff_idle = curr['idle'] - prev['idle']
                cpu_util = 100 * (1 - diff_idle / diff_tot) if diff_tot else 0
                
                timestamp = datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]
                writer.writerow([timestamp, f"{npu_util:.2f}", f"{cpu_util:.2f}", f"{curr['mem']:.2f}"])
                prev = curr

def run_inference(combo_name, paths, trial):
    params = hp.VDevice.create_params()
    params.scheduling_algorithm = hp.HailoSchedulingAlgorithm.ROUND_ROBIN
    with hp.VDevice(params) as vdevice:
        networks = []
        for p in paths:
            hef = hp.HEF(p)
            name = "DET" if "seg" not in p and "pose" not in p else ("SEG" if "seg" in p else "POSE")
            ng = vdevice.configure(hef)[0]
            in_p = hp.InputVStreamParams.make_from_network_group(ng, quantized=True, format_type=hp.FormatType.UINT8)
            out_p = hp.OutputVStreamParams.make_from_network_group(ng, quantized=False, format_type=hp.FormatType.FLOAT32)
            networks.append({"ng": ng, "in": in_p, "out": out_p, "name": hef.get_input_vstream_infos()[0].name})

        from contextlib import ExitStack
        with ExitStack() as stack:
            pipelines = [stack.enter_context(hp.InferVStreams(n["ng"], n["in"], n["out"])) for n in networks]
            images = sorted(glob.glob(f"{IMAGE_DIR}/*.jpg"))[:NUM_IMAGES]
            model_times = {n["name"]: [] for n in networks}

            def single_infer(pipe, in_data, model_name):
                t0 = time.time()
                pipe.infer(in_data)  
                t1 = time.time()
               
                hw_time = pipe.get_hw_time() * 1000
                return model_name, t0, t1, hw_time

            for idx, img_path in enumerate(images):
                frame = preprocess_letterbox(img_path)
                if frame is None: continue
                input_data = [{n["name"]: frame} for n in networks]

                with concurrent.futures.ThreadPoolExecutor(max_workers=len(pipelines)) as executor:
                    futures = []
                    for i, pipe in enumerate(pipelines):
                        futures.append(executor.submit(single_infer, pipe, input_data[i], networks[i]["name"]))
                    
                    for future in concurrent.futures.as_completed(futures):
                        model_name, t0, t1, hw_time = future.result()
                        latency = (t1 - t0) * 1000
                        model_times[model_name].append(t1 - t0)

            with open(LATENCY_CSV, 'a', newline='') as f:
                writer = csv.writer(f)
                for model_name, times in model_times.items():
                    avg_ms = np.mean(times) * 1000
                    fps = 1 / np.mean(times)
                    writer.writerow([combo_name, trial, model_name, f"{avg_ms:.2f}", f"{fps:.2f}"])

if __name__ == "__main__":
    if os.path.exists(LATENCY_CSV): os.remove(LATENCY_CSV)
    with open(LATENCY_CSV, 'w', newline='') as f:
        csv.writer(f).writerow(['Combination', 'Trial', 'Latency_ms', 'FPS'])

    combos = [
        #("DET", [DET_PATH]),
        #("SEG", [SEG_PATH]),
        #("POSE", [POSE_PATH]),
        #("DET_SEG", [DET_PATH, SEG_PATH]),
        #("DET_POSE", [DET_PATH, POSE_PATH]),
        ("SEG_POSE", [SEG_PATH, POSE_PATH]), 
        #("DET_SEG_POSE", [DET_PATH, SEG_PATH, POSE_PATH])
    ]

    total_time = time.time()
    for name, paths in combos:
        for t in range(1, 2):
            print(f"Running {name} Trial {t}...")
            stop_event = mp.Event()
            util_file = f"util_sched_{name}_trial{t}.csv"
            
            if os.path.exists("/tmp/hmon_files"):
                for f_rem in os.listdir("/tmp/hmon_files"):
                    try: os.remove(os.path.join("/tmp/hmon_files", f_rem))
                    except: pass

            m_proc = mp.Process(target=monitor_resource, args=(util_file, stop_event))
            m_proc.start()
            time.sleep(2) # 🌟 서비스가 파일을 생성할 시간 확보
            
            p_bench = mp.Process(target=run_inference, args=(name, paths, t))
            p_bench.start()
            p_bench.join()
            
            stop_event.set()
            m_proc.join()
            time.sleep(5)
    print("Total time : ", time.time()- total_time)

    print("ALL DONE.")
