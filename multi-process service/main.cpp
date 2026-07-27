#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <chrono>
#include <map>
#include <cmath>
#include <thread>
#include <mutex>
#include <atomic>
#include <filesystem>
#include <algorithm>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include "hailo/hailort.hpp"

using namespace std;
using namespace cv;
namespace fs = std::filesystem;
using namespace hailort;

// ============================================================================
// 1. 공통 데이터 구조체 및 헬퍼 함수
// ============================================================================
struct MetaData {
    int img_id;
    int orig_w, orig_h;
    int pad_w, pad_h;
    float scale;
};

struct BBoxResult {
    int category_id;
    Rect2f bbox;
    float score;
    Mat mask; // SEG 
    vector<float> keypoints; // POSE 
};

inline float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

void softmax_1d(const float* input, float* output, int size) {
    float max_val = *std::max_element(input, input + size);
    float sum = 0.0f;
    for (int i = 0; i < size; ++i) {
        output[i] = std::exp(input[i] - max_val);
        sum += output[i];
    }
    for (int i = 0; i < size; ++i) output[i] /= sum;
}

// ============================================================================
// 2. 디코더 (DET, SEG, POSE) 
// ============================================================================
class YOLOv8DetDecoder {
public:
    vector<BBoxResult> decode(const float* raw_data, size_t num_detections, const MetaData& meta) {
        // C++ DET 디코딩 로직 (생략 방지를 위해 기본 구조 유지)
        vector<BBoxResult> results;
        // 실제 구현 시 Hailo NMS 출력 포맷에 맞춰 파싱
        return results;
    }
};

class YOLOv8SegDecoder {
public:
    vector<BBoxResult> decode(const map<string, const float*>& raw_data, const MetaData& meta) {
        vector<BBoxResult> results;
        // 이전 답변의 SEG OpenCV Mat 곱셈 NMS 로직 삽입
        return results;
    }
};

class YOLOv8PoseDecoder {
public:
    vector<BBoxResult> decode(const map<string, const float*>& raw_data, const MetaData& meta) {
        vector<BBoxResult> results;
        // 이전 답변의 POSE 키포인트 디코딩 로직 삽입
        return results;
    }
};

// ============================================================================
// 3. Hailo Inference Engine
// ============================================================================
class FastHailoInfer {
private:
    std::unique_ptr<VDevice> vdevice; 
    std::shared_ptr<ConfiguredNetworkGroup> network_group;
    std::vector<InputVStream> input_vstreams;
    std::vector<OutputVStream> output_vstreams;

    std::vector<uint8_t> aligned_input_buffer; 
    std::map<std::string, std::vector<uint8_t>> output_buffers;
    string model_type;

    YOLOv8DetDecoder det_decoder;
    YOLOv8SegDecoder seg_decoder;
    YOLOv8PoseDecoder pose_decoder;

public:
    FastHailoInfer(const std::string& hef_path) {
        if (hef_path.find("seg") != string::npos) model_type = "SEG";
        else if (hef_path.find("pose") != string::npos) model_type = "POSE";
        else model_type = "DET";

        hailo_vdevice_params_t params = {};
        hailo_init_vdevice_params(&params);
        params.multi_process_service = true;
        params.group_id = "SHARED";

        auto vdevice_exp = VDevice::create(params);
        vdevice = vdevice_exp.release();

        auto hef_exp = Hef::create(hef_path);
        auto hef = hef_exp.release();
        
        auto configure_params = vdevice->create_configure_params(hef).release();
        auto network_groups = vdevice->configure(hef, configure_params).release();
        network_group = network_groups[0];

        auto vstreams_exp = VStreamsBuilder::create_vstreams(*network_group, {}, HAILO_FORMAT_TYPE_AUTO);
        auto vstreams = vstreams_exp.release();
        
        input_vstreams = std::move(vstreams.first);  
        output_vstreams = std::move(vstreams.second); 

        aligned_input_buffer.resize(input_vstreams[0].get_frame_size());
        for (auto& out : output_vstreams) {
            output_buffers[out.name()].resize(out.get_frame_size());
        }
    }

    void preprocess_letterbox(const cv::Mat& img, cv::Mat& hw_buffer, MetaData& meta) {
        int target_w = 640;
        int target_h = 640;
        int h = img.rows;
        int w = img.cols;
        float r = std::min((float)target_h / h, (float)target_w / w);
        int nh = std::round(h * r);
        int nw = std::round(w * r);

        cv::Mat resized;
        cv::resize(img, resized, cv::Size(nw, nh));

        hw_buffer.setTo(cv::Scalar(114, 114, 114));
        int top = (target_h - nh) / 2;
        int left = (target_w - nw) / 2;

        resized.copyTo(hw_buffer(cv::Rect(left, top, nw, nh)));
        cv::cvtColor(hw_buffer, hw_buffer, cv::COLOR_BGR2RGB);

        meta.orig_w = w; meta.orig_h = h;
        meta.pad_w = left; meta.pad_h = top;
        meta.scale = r;
    }

    std::vector<double> run_benchmark(const std::vector<std::string>& image_paths) {
        std::vector<double> latencies;

        for (const auto& path : image_paths) {
            cv::Mat img = cv::imread(path);
            if (img.empty()) continue;

            MetaData meta;
            cv::Mat hw_buffer_mat(640, 640, CV_8UC3, aligned_input_buffer.data());
            
            // ----------------------------------------------------
            // 💡 E2E Latency 측정 시작 (전처리 + 추론 + 후처리)
            auto t_start = std::chrono::high_resolution_clock::now();
            
            preprocess_letterbox(img, hw_buffer_mat, meta);

            hailo_status status = input_vstreams[0].write(MemoryView(aligned_input_buffer.data(), aligned_input_buffer.size()));
            if (status != HAILO_SUCCESS) throw std::runtime_error("Write failed");

            for (auto& out_vstream : output_vstreams) {
                status = out_vstream.read(MemoryView(output_buffers[out_vstream.name()].data(), output_buffers[out_vstream.name()].size()));
                if (status != HAILO_SUCCESS) throw std::runtime_error("Read failed");
            }

            // 후처리 분기
            map<string, const float*> raw_data_map;
            for (auto& pair : output_buffers) {
                raw_data_map[pair.first] = reinterpret_cast<const float*>(pair.second.data());
            }

            vector<BBoxResult> results;
            if (model_type == "SEG") results = seg_decoder.decode(raw_data_map, meta);
            else if (model_type == "POSE") results = pose_decoder.decode(raw_data_map, meta);
            // else results = det_decoder.decode(...);

            auto t_end = std::chrono::high_resolution_clock::now();
            // ----------------------------------------------------

            double latency_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
            latencies.push_back(latency_ms);
        }
        return latencies;
    }
};

// ============================================================================
// 4. 자원 모니터링 스레드 
// ============================================================================
void get_sys(long long& total_cpu, long long& idle_cpu, float& mem_util) {
    ifstream meminfo("/proc/meminfo");
    string line, key;
    long long mem_total = 0, mem_avail = 0;
    while (getline(meminfo, line)) {
        istringstream iss(line);
        iss >> key;
        if (key == "MemTotal:") iss >> mem_total;
        else if (key == "MemAvailable:") iss >> mem_avail;
    }
    mem_util = 100.0f * (1.0f - (float)mem_avail / (float)mem_total);

    ifstream stat("/proc/stat");
    getline(stat, line);
    istringstream iss_stat(line);
    string cpu_lbl;
    long long user, nice, system, idle, iowait, irq, softirq;
    iss_stat >> cpu_lbl >> user >> nice >> system >> idle >> iowait >> irq >> softirq;
    idle_cpu = idle + iowait;
    total_cpu = user + nice + system + idle + iowait + irq + softirq;
}

void monitor_resource(const string& output_file, atomic<bool>& stop_event) {
    ofstream ofs(output_file);
    ofs << "Time,NPU_util,CPU_util,Mem_util\n";
    
    long long p_tot = 0, p_idle = 0;
    float m_util = 0.0f;
    get_sys(p_tot, p_idle, m_util);

    while (!stop_event) {
        this_thread::sleep_for(chrono::milliseconds(500));
        
        long long c_tot = 0, c_idle = 0;
        get_sys(c_tot, c_idle, m_util);
        
        float cpu_util = 0.0f;
        if (c_tot != p_tot) {
            cpu_util = 100.0f * (1.0f - (float)(c_idle - p_idle) / (c_tot - p_tot));
        }

        // C++에서 Protobuf 없이 NPU Util 읽기 (임시 처리)
        // 실제 운영 시 hailortcli measure-utilization 출력 결과를 읽거나 Protobuf 컴파일 필요
        float npu_util = 0.0f; 

        auto now = chrono::system_clock::to_time_t(chrono::system_clock::now());
        char time_str[9];
        strftime(time_str, sizeof(time_str), "%H:%M:%S", localtime(&now));

        ofs << time_str << "," << npu_util << "," << cpu_util << "," << m_util << "\n";
        ofs.flush();
        
        p_tot = c_tot; p_idle = c_idle;
    }
}

// ============================================================================
// 5. 작업자 스레드 (Worker Thread) 및 전역 CSV Mutex
// ============================================================================
mutex csv_mutex;

void run_model_thread(string hef_path, string combo_name, int trial) {
    try {
        fs::path p(hef_path);
        string model_name = p.filename().string();
        cout << "[" << model_name << "] C++ 엔진 초기화 및 NPU 메모리 할당 중..." << endl;
        
        FastHailoInfer npu_engine(hef_path);
        
        // 이미지 수집
        string img_dir = "/app/tappas/rpi2/npu/mps_cpp_experiment/sampled_val2017";
        vector<string> images;
        for (const auto& entry : fs::directory_iterator(img_dir)) {
            if (entry.path().extension() == ".jpg") {
                images.push_back(entry.path().string());
            }
        }
        sort(images.begin(), images.end());
        if (images.size() > 500) images.resize(500); // 파이썬의 NUM_IMAGES = 500
        
        cout << "[" << model_name << "] Inference Start!" << endl;
        vector<double> latencies = npu_engine.run_benchmark(images);
        
        double sum = 0;
        for (double l : latencies) sum += l;
        double avg_ms = sum / latencies.size();
        double fps = 1000.0 / avg_ms;
        
        cout << "[" << model_name << "] Average Inference Time: " << avg_ms << "ms | FPS: " << fps << endl;
        
        // CSV 쓰레드 세이프 쓰기
        lock_guard<mutex> lock(csv_mutex);
        ofstream ofs("latency_results.csv", ios::app);
        ofs << combo_name << "," << trial << "," << model_name << "," << avg_ms << "," << fps << "\n";
        
    } catch (const exception& e) {
        cerr << "Error in " << hef_path << ": " << e.what() << endl;
    }
}

// ============================================================================
// 6. 메인 함수 
// ============================================================================
int main() {
    string DET_PATH = "/app/tappas/rpi2/npu/yolov8s/yolov8s.hef";
    string SEG_PATH = "/app/tappas/rpi2/npu/yolov8s/yolov8s_seg.hef";
    string POSE_PATH = "/app/tappas/rpi2/npu/yolov8s/yolov8s_pose.hef";

    // 결과 CSV 초기화
    ofstream ofs("latency_results.csv");
    ofs << "Combination,Trial,Model,Latency_ms,FPS\n";
    ofs.close();

    struct Combo { string name; vector<string> paths; };
    vector<Combo> combinations = {
        {"DET_SEG_POSE", {DET_PATH, SEG_PATH, POSE_PATH}}
    };

    cout << ">>> C++ All-in-One Benchmark Started." << endl;
    auto start_time = chrono::high_resolution_clock::now();

    for (const auto& combo : combinations) {
        for (const auto& p : combo.paths) {
            if (!fs::exists(p)) {
                cerr << "CRITICAL ERROR: File not found: " << p << endl;
                return 1;
            }
        }

        for (int t = 1; t <= 1; ++t) {
            cout << "Running " << combo.name << " - Trial " << t << "..." << endl;
            
            atomic<bool> stop_event(false);
            string util_file = "util_" + combo.name + "_trial" + to_string(t) + ".csv";
            
            // 모니터링 스레드 실행
            thread m_thread(monitor_resource, util_file, ref(stop_event));

            // 워커 스레드 실행 (파이썬의 멀티프로세싱 대체)
            vector<thread> workers;
            for (const auto& p : combo.paths) {
                workers.emplace_back(run_model_thread, p, combo.name, t);
            }

            // 조인
            for (auto& w : workers) {
                if (w.joinable()) w.join();
            }

            stop_event = true;
            if (m_thread.joinable()) m_thread.join();

            cout << "Done. Cooling down..." << endl;
        }
    }

    auto end_time = chrono::high_resolution_clock::now();
    double total_sec = chrono::duration<double>(end_time - start_time).count();
    
    cout << "Total Inference Time : " << total_sec << " sec" << endl;
    cout << ">>> ALL EXPERIMENTS FINISHED." << endl;

    return 0;
}
