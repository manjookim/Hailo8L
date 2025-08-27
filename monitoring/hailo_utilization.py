#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import time
from time import sleep
from scheduler_mon_pb2 import ProtoMon
import re
import datetime
import sys

class HailoMonitor:
    """Hailo NPU의 사용량을 protobuf 파일을 통해 읽어오는 클래스입니다."""
    def __init__(self, update_period=1, directory="/tmp/hmon_files"):
        self.interval = update_period
        self.directory = directory
        self.data = {}

    def get_single_file(self):
        try:
            files = os.listdir(self.directory)
            return os.path.join(self.directory, files[0]) if files else None
        except Exception:
            return None

    def read_stats(self):
        try:
            path = self.get_single_file()
            if path and os.path.exists(path):
                proto = ProtoMon()
                with open(path, "rb") as f:
                    proto.ParseFromString(f.read())
                self.data["utilization"] = proto.device_infos[0].utilization if proto.device_infos else 0.0
            else:
                self.data["utilization"] = 0.0
        except Exception:
            self.data["utilization"] = 0.0
        return self.data

def get_cpu_and_mem_stats():
    stats = {}
    try:
        with open('/proc/meminfo', 'r') as f:
            meminfo = f.read()
            stats['mem_total_kb'] = int(re.search(r'MemTotal:\s+(\d+)', meminfo).group(1))
            stats['mem_available_kb'] = int(re.search(r'MemAvailable:\s+(\d+)', meminfo).group(1))
        with open('/proc/stat', 'r') as f:
            cpu_times = [int(x) for x in f.readline().split()[1:]]
            stats['cpu_total_time'] = sum(cpu_times)
            stats['cpu_idle_time'] = cpu_times[3]
    except Exception:
        return None
    return stats

def calculate_cpu_usage(prev, curr):
    if not prev or not curr: return 0.0
    total_diff = curr['cpu_total_time'] - prev['cpu_total_time']
    idle_diff = curr['cpu_idle_time'] - prev['cpu_idle_time']
    return 100.0 * (1 - idle_diff / total_diff) if total_diff else 0.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--update_period", type=float, default=1)
    args = parser.parse_args()

    # 로그 파일 경로를 파이썬에서 동적으로 생성합니다.
    log_dir = "/home/rpi2/npu/logs"
    current_date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    log_filename = f"monitor-{current_date_str}.log"
    log_filepath = os.path.join(log_dir, log_filename)

    npu_monitor = HailoMonitor(args.update_period)
    prev_stats = get_cpu_and_mem_stats()

    try:
        while True:
            # 매일 자정이 지나면 새로운 파일에 로그를 기록하도록 파일 경로를 업데이트합니다.
            new_date_str = datetime.datetime.now().strftime("%Y-%m-%d")
            if new_date_str != current_date_str:
                current_date_str = new_date_str
                log_filename = f"monitor-{current_date_str}.log"
                log_filepath = os.path.join(log_dir, log_filename)

            npu_util = npu_monitor.read_stats().get('utilization', 0.0)
            curr_stats = get_cpu_and_mem_stats()
            if not curr_stats:
                time.sleep(args.update_period)
                continue

            cpu_util = calculate_cpu_usage(prev_stats, curr_stats)
            mem_used = curr_stats['mem_total_kb'] - curr_stats['mem_available_kb']
            mem_util = 100.0 * (mem_used / curr_stats['mem_total_kb'])

            log_msg = (
                f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}, "
                f"NPU_util: {npu_util:.2f}%, "
                f"CPU_util: {cpu_util:.2f}%, "
                f"Mem_util: {mem_util:.2f}%"
            )
            
            with open(log_filepath, "a") as f:
                f.write(log_msg + "\n")

            prev_stats = curr_stats
            time.sleep(args.update_period)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        # 에러 발생 시 로그 경로를 확인하고 기록합니다.
        log_dir_error = "/home/rpi2/npu/logs"
        log_filename_error = f"monitor-error-{datetime.datetime.now().strftime('%Y-%m-%d')}.log"
        log_filepath_error = os.path.join(log_dir_error, log_filename_error)
        with open(log_filepath_error, "a") as f:
            f.write(f"[{datetime.datetime.now()}] Monitoring script stopped with error: {e}\n")
        exit(1)

if __name__ == "__main__":
    main()
