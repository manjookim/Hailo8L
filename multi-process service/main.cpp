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
    int img_id; // 💡 누락되었던 image_id 추가 완료
    int orig_w, orig_h;
    int pad_w, pad_h;
    float scale;
};

struct BBoxResult {
    int image_id; // 💡 결과 구조체에도 추가 완료
    int category_id;
    Rect2f bbox;
    float score;
    Mat mask; 
    vector<float> keypoints; 
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
// --- DET 디코더 ---
class YOLOv8DetDecoder {
private:
    int model_w = 640;
    int model_h = 640;
    map<int, int> id_map; 

public:
    vector<BBoxResult> decode(const vector<vector<vector<float>>>& raw_detections, const MetaData& meta) {
        vector<BBoxResult> results;
        for (size_t class_id = 0; class_id < raw_detections.size(); ++class_id) {
            for (const auto& det : raw_detections[class_id]) {
                if (det.size() < 5) continue;
                
                float ymin_n = det[0], xmin_n = det[1];
                float ymax_n = det[2], xmax_n = det[3];
                float score = det[4];

                if (score < 0.25f) continue;

                float x1 = (xmin_n * model_w - meta.pad_w) / meta.scale;
                float y1 = (ymin_n * model_h - meta.pad_h) / meta.scale;
                float x2 = (xmax_n * model_w - meta.pad_w) / meta.scale;
                float y2 = (ymax_n * model_h - meta.pad_h) / meta.scale;

                BBoxResult res;
                res.image_id = meta.img_id;
                res.category_id = id_map.empty() ? class_id : id_map[class_id];
                res.bbox = Rect2f(std::max(0.0f, x1), std::max(0.0f, y1), std::abs(x2 - x1), std::abs(y2 - y1));
                res.score = score;
                results.push_back(res);
            }
        }
        return results;
    }
};

// --- SEG 디코더 ---
// --- SEG 디코더 (최적화 버전) ---
class YOLOv8SegDecoder {
private:
    // 💡 최적화 1: 불필요한 노이즈를 거르기 위해 DET와 동일하게 0.25로 상향
    float CONF_THRES = 0.25f; 
    float IOU_THRES = 0.65f;
    float MASK_THRES = 0.5f;
    int REG_MAX = 16;
    int NUM_CLS = 80;
    int NUM_COEFF = 32;
    int MAX_DET = 100; // 💡 최적화 2: 최대 탐지 객체 수 제한

    void dfl_decode(const float* box_raw, float cx, float cy, int stride, Rect2f& out_box) {
        float dist[4];
        for (int i = 0; i < 4; ++i) {
            float softmax_out[16];
            softmax_1d(box_raw + i * REG_MAX, softmax_out, REG_MAX);
            float sum = 0;
            for (int j = 0; j < REG_MAX; ++j) sum += softmax_out[j] * j;
            dist[i] = sum * stride;
        }
        out_box = Rect2f(cx - dist[0], cy - dist[1], dist[2] + dist[0], dist[3] + dist[1]);
    }

public:
    vector<BBoxResult> decode(const map<string, const float*>& raw_data, const MetaData& meta) {
        vector<Rect2f> all_boxes;
        vector<float> all_scores;
        vector<int> all_cls_ids;
        vector<vector<float>> all_coeffs;

        struct HeadConfig { string box, cls, coef; int stride; int size; };
        HeadConfig heads[] = {
            {"yolov8s_seg/conv44", "yolov8s_seg/conv45", "yolov8s_seg/conv46", 8, 80 * 80},
            {"yolov8s_seg/conv60", "yolov8s_seg/conv61", "yolov8s_seg/conv62", 16, 40 * 40},
            {"yolov8s_seg/conv73", "yolov8s_seg/conv74", "yolov8s_seg/conv75", 32, 20 * 20}
        };

        for (const auto& h : heads) {
            const float* box_feat = raw_data.at(h.box);
            const float* cls_feat = raw_data.at(h.cls);
            const float* coef_feat = raw_data.at(h.coef);

            int grid_w = 640 / h.stride;

            for (int i = 0; i < h.size; ++i) {
                float max_score = 0;
                int class_id = -1;
                for (int c = 0; c < NUM_CLS; ++c) {
                    float s = cls_feat[i * NUM_CLS + c];
                    if (s > max_score) { max_score = s; class_id = c; }
                }

                if (max_score > CONF_THRES) {
                    float cx = (i % grid_w + 0.5f) * h.stride;
                    float cy = (i / grid_w + 0.5f) * h.stride;
                    
                    Rect2f box;
                    dfl_decode(box_feat + i * 4 * REG_MAX, cx, cy, h.stride, box);
                    
                    all_boxes.push_back(box);
                    all_scores.push_back(max_score);
                    all_cls_ids.push_back(class_id);
                    all_coeffs.push_back(vector<float>(coef_feat + i * NUM_COEFF, coef_feat + (i + 1) * NUM_COEFF));
                }
            }
        }

        if (all_boxes.empty()) return {};

        vector<Rect> nms_boxes;
        for (const auto& b : all_boxes) nms_boxes.push_back(Rect(b.x, b.y, b.width, b.height));

        vector<int> keep_idx;
        dnn::NMSBoxes(nms_boxes, all_scores, CONF_THRES, IOU_THRES, keep_idx);
        
        // 💡 속도 방어: 아무리 많아도 MAX_DET 개수까지만 자름
        if (keep_idx.size() > MAX_DET) keep_idx.resize(MAX_DET);

        const float* proto_data = raw_data.at("yolov8s_seg/conv48");
        int Ph = 160, Pw = 160;
        Mat proto_mat(Ph * Pw, NUM_COEFF, CV_32F, (void*)proto_data); 
        
        int N = keep_idx.size();
        if (N == 0) return {};

        Mat coeffs_mat(N, NUM_COEFF, CV_32F);
        for (int i = 0; i < N; ++i) {
            memcpy(coeffs_mat.ptr<float>(i), all_coeffs[keep_idx[i]].data(), NUM_COEFF * sizeof(float));
        }

        Mat masks_raw = coeffs_mat * proto_mat.t();

        vector<BBoxResult> final_results;
        for (int i = 0; i < N; ++i) {
            int idx = keep_idx[i];
            
            Rect2f b = all_boxes[idx];
            float x1 = std::max(0.0f, (b.x - meta.pad_w) / meta.scale);
            float y1 = std::max(0.0f, (b.y - meta.pad_h) / meta.scale);
            float x2 = std::min((float)meta.orig_w, (b.x + b.width - meta.pad_w) / meta.scale);
            float y2 = std::min((float)meta.orig_h, (b.y + b.height - meta.pad_h) / meta.scale);
            
            if (x2 - x1 <= 0 || y2 - y1 <= 0) continue;

            float sx = Pw / 640.0f;
            float sy = Ph / 640.0f;
            
            int x1p = std::max(0, std::min(Pw - 1, (int)(b.x * sx)));
            int y1p = std::max(0, std::min(Ph - 1, (int)(b.y * sy)));
            int x2p = std::max(0, std::min(Pw, (int)((b.x + b.width) * sx)));
            int y2p = std::max(0, std::min(Ph, (int)((b.y + b.height) * sy)));

            int roi_w = std::max(0, x2p - x1p);
            int roi_h = std::max(0, y2p - y1p);

            Mat crop = Mat::zeros(Ph, Pw, CV_32F);
            if (roi_w > 0 && roi_h > 0) {
                Rect roi(x1p, y1p, roi_w, roi_h);
                
                Mat mask_1d = masks_raw.row(i);
                Mat mask_2d = mask_1d.reshape(1, Ph); 
                
                // 💡 최적화 3: 전체 영역이 아닌, 필요한 영역(ROI)만 먼저 복사한 뒤 Sigmoid 연산!
                Mat cropped_mask = mask_2d(roi).clone();
                exp(-cropped_mask, cropped_mask);
                add(cropped_mask, 1.0, cropped_mask);
                divide(1.0, cropped_mask, cropped_mask);
                
                // 연산이 끝난 영역을 다시 빈 캔버스에 얹음
                cropped_mask.copyTo(crop(roi));
            } else {
                continue;
            }

            Mat mask_640, mask_orig;
            resize(crop, mask_640, Size(640, 640), 0, 0, INTER_LINEAR);
            
            int w_unpad = std::round(meta.orig_w * meta.scale);
            int h_unpad = std::round(meta.orig_h * meta.scale);
            w_unpad = std::min(w_unpad, 640 - meta.pad_w);
            h_unpad = std::min(h_unpad, 640 - meta.pad_h);

            if (w_unpad > 0 && h_unpad > 0) {
                Mat mask_unpad = mask_640(Rect(meta.pad_w, meta.pad_h, w_unpad, h_unpad));
                resize(mask_unpad, mask_orig, Size(meta.orig_w, meta.orig_h), 0, 0, INTER_LINEAR);
            } else {
                mask_orig = Mat::zeros(meta.orig_h, meta.orig_w, CV_32F);
            }

            Mat binary_mask;
            threshold(mask_orig, binary_mask, MASK_THRES, 255, THRESH_BINARY);
            binary_mask.convertTo(binary_mask, CV_8U); 

            BBoxResult res;
            res.image_id = meta.img_id;
            res.category_id = all_cls_ids[idx]; 
            res.bbox = Rect2f(x1, y1, x2 - x1, y2 - y1);
            res.score = all_scores[idx];
            res.mask = binary_mask;
            final_results.push_back(res);
        }

        return final_results;
    }
};
// --- POSE 디코더 ---
class YOLOv8PoseDecoder {
private:
    float CONF_THRES = 0.3f;
    float IOU_THRES = 0.45f;
    int REG_MAX = 16;

    void dfl_decode(const float* box_raw, float cx, float cy, int stride, Rect2f& out_box) {
        float dist[4];
        for (int i = 0; i < 4; ++i) {
            float softmax_out[16];
            softmax_1d(box_raw + i * REG_MAX, softmax_out, REG_MAX);
            float sum = 0;
            for (int j = 0; j < REG_MAX; ++j) sum += softmax_out[j] * j;
            dist[i] = sum * stride;
        }
        out_box = Rect2f(cx - dist[0], cy - dist[1], dist[2] + dist[0], dist[3] + dist[1]); 
    }

public:
    vector<BBoxResult> decode(const map<string, const float*>& raw_data, const MetaData& meta) {
        vector<Rect2f> all_boxes;
        vector<float> all_scores;
        vector<vector<float>> all_kpts;

        struct HeadConfig { string reg, cls, kpt; int stride; int size; };
        HeadConfig heads[] = {
            {"yolov8s_pose/conv43", "yolov8s_pose/conv44", "yolov8s_pose/conv45", 8, 80 * 80},
            {"yolov8s_pose/conv57", "yolov8s_pose/conv58", "yolov8s_pose/conv59", 16, 40 * 40},
            {"yolov8s_pose/conv70", "yolov8s_pose/conv71", "yolov8s_pose/conv72", 32, 20 * 20}
        };

        for (const auto& h : heads) {
            const float* reg_feat = raw_data.at(h.reg);
            const float* cls_feat = raw_data.at(h.cls);
            const float* kpt_feat = raw_data.at(h.kpt);

            int grid_w = 640 / h.stride;

            for (int i = 0; i < h.size; ++i) {
                float score = cls_feat[i]; 
                if (score > CONF_THRES) {
                    float cx = (i % grid_w + 0.5f) * h.stride;
                    float cy = (i / grid_w + 0.5f) * h.stride;

                    Rect2f box;
                    dfl_decode(reg_feat + i * 4 * REG_MAX, cx, cy, h.stride, box);
                    all_boxes.push_back(box);
                    all_scores.push_back(score);

                    vector<float> kpts(17 * 3);
                    const float* kpt_base = kpt_feat + i * 17 * 3;
                    for (int k = 0; k < 17; ++k) {
                        float kx = (kpt_base[k * 3 + 0] * 2.0f + (i % grid_w)) * h.stride;
                        float ky = (kpt_base[k * 3 + 1] * 2.0f + (i / grid_w)) * h.stride;
                        float kv = kpt_base[k * 3 + 2];
                        kpts[k * 3 + 0] = kx;
                        kpts[k * 3 + 1] = ky;
                        kpts[k * 3 + 2] = kv;
                    }
                    all_kpts.push_back(kpts);
                }
            }
        }

        if (all_boxes.empty()) return {};

        // 💡 해결: NMSBoxes 정수 변환
        vector<Rect> nms_boxes;
        for (const auto& b : all_boxes) nms_boxes.push_back(Rect(b.x, b.y, b.width, b.height));

        vector<int> keep_idx;
        dnn::NMSBoxes(nms_boxes, all_scores, CONF_THRES, IOU_THRES, keep_idx);

        vector<BBoxResult> final_results;
        for (int idx : keep_idx) {
            Rect2f b = all_boxes[idx];
            float x1 = (b.x - meta.pad_w) / meta.scale;
            float y1 = (b.y - meta.pad_h) / meta.scale;
            float w = b.width / meta.scale;
            float h = b.height / meta.scale;

            BBoxResult res;
            res.image_id = meta.img_id;
            res.category_id = 1; 
            res.bbox = Rect2f(x1, y1, w, h);
            res.score = all_scores[idx];

            for (int k = 0; k < 17; ++k) {
                float raw_kx = all_kpts[idx][k * 3 + 0];
                float raw_ky = all_kpts[idx][k * 3 + 1];
                float raw_kv = all_kpts[idx][k * 3 + 2];

                float rx = (raw_kx - meta.pad_w) / meta.scale;
                float ry = (raw_ky - meta.pad_h) / meta.scale;
                float v_flag = sigmoid(raw_kv) > 0.5f ? 2.0f : 1.0f; 

                res.keypoints.push_back(rx);
                res.keypoints.push_back(ry);
                res.keypoints.push_back(v_flag);
            }
            final_results.push_back(res);
        }

        return final_results;
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
            // 💡 파일명에서 image_id 추출 (예: "000123.jpg" -> 123)
            try {
                meta.img_id = std::stoi(fs::path(path).stem().string());
            } catch (...) {
                meta.img_id = 0; 
            }

            cv::Mat hw_buffer_mat(640, 640, CV_8UC3, aligned_input_buffer.data());
            
            auto t_start = std::chrono::high_resolution_clock::now();
            
            preprocess_letterbox(img, hw_buffer_mat, meta);

            hailo_status status = input_vstreams[0].write(MemoryView(aligned_input_buffer.data(), aligned_input_buffer.size()));
            if (status != HAILO_SUCCESS) throw std::runtime_error("Write failed");

            for (auto& out_vstream : output_vstreams) {
                status = out_vstream.read(MemoryView(output_buffers[out_vstream.name()].data(), output_buffers[out_vstream.name()].size()));
                if (status != HAILO_SUCCESS) throw std::runtime_error("Read failed");
            }

            map<string, const float*> raw_data_map;
            for (auto& pair : output_buffers) {
                raw_data_map[pair.first] = reinterpret_cast<const float*>(pair.second.data());
            }

            vector<BBoxResult> results;
            if (model_type == "SEG") results = seg_decoder.decode(raw_data_map, meta);
            else if (model_type == "POSE") results = pose_decoder.decode(raw_data_map, meta);
            // DET 처리는 별도 로직 (NMS 포함) 필요 시 연동

            auto t_end = std::chrono::high_resolution_clock::now();

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
// 5. 작업자 스레드 (Worker Thread) 
// ============================================================================
mutex csv_mutex;

void run_model_thread(string hef_path, string combo_name, int trial) {
    try {
        fs::path p(hef_path);
        string model_name = p.filename().string();
        cout << "[" << model_name << "] C++ 엔진 초기화 및 NPU 메모리 할당 중..." << endl;
        
        FastHailoInfer npu_engine(hef_path);
        
        string img_dir = "/app/tappas/rpi2/npu/mps_cpp_experiment/sampled_val2017";
        vector<string> images;
        for (const auto& entry : fs::directory_iterator(img_dir)) {
            if (entry.path().extension() == ".jpg") {
                images.push_back(entry.path().string());
            }
        }
        sort(images.begin(), images.end());
        if (images.size() > 500) images.resize(500); 
        
        cout << "[" << model_name << "] Inference Start!" << endl;
        vector<double> latencies = npu_engine.run_benchmark(images);
        
        double sum = 0;
        for (double l : latencies) sum += l;
        double avg_ms = sum / latencies.size();
        double fps = 1000.0 / avg_ms;
        
        cout << "[" << model_name << "] Average Inference Time: " << avg_ms << "ms | FPS: " << fps << endl;
        
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

    ofstream ofs("latency_results.csv");
    ofs << "Combination,Trial,Model,Latency_ms,FPS\n";
    ofs.close();

    struct Combo { string name; vector<string> paths; };
    vector<Combo> combinations = {
        {"DET", {DET_PATH}},
        //{"SEG", {SEG_PATH}},
        //{"EST", {POSE_PATH}},
        //{"DET_SEG", {DET_PATH, SEG_PATH}},
        //{"DET_EST", {DET_PATH, POSE_PATH}},
        //{"SEG_EST", {SEG_PATH, POSE_PATH}},
        //{"DET_SEG_POSE", {DET_PATH, SEG_PATH, POSE_PATH}}
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
            
            thread m_thread(monitor_resource, util_file, ref(stop_event));

            vector<thread> workers;
            for (const auto& p : combo.paths) {
                workers.emplace_back(run_model_thread, p, combo.name, t);
            }

            for (auto& w : workers) {
                if (w.joinable()) w.join();
            }

            stop_event = true;
            if (m_thread.joinable()) m_thread.join();

            cout << "Done. Cooling down..." << endl;

            auto end_time = chrono::high_resolution_clock::now();
            double total_sec = chrono::duration<double>(end_time - start_time).count();
    
            cout << "Total Inference Time : " << total_sec << " sec" << endl;
            cout << ">>> ALL EXPERIMENTS FINISHED." << endl;

        }
    }


    return 0;
}
