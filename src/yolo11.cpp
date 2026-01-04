#include <memory>
#include <vector>
#include <algorithm>
#include <chrono>
#include <iostream>
#include "layer.h"
#include "net.h"
#include <opencv2/opencv.hpp>
#include <float.h>

#define MAX_STRIDE 32

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

static inline float intersection_area(const Object &a, const Object &b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void qsort_descent_inplace(std::vector<Object> &objects, int left, int right)
{
    int i = left, j = right;
    float p = objects[(left + right) / 2].prob;
    while (i <= j)
    {
        while (objects[i].prob > p)
            i++;
        while (objects[j].prob < p)
            j--;
        if (i <= j)
            std::swap(objects[i++], objects[j--]);
    }
#pragma omp parallel sections
    {
#pragma omp section
        if (left < j)
            qsort_descent_inplace(objects, left, j);
#pragma omp section
        if (i < right)
            qsort_descent_inplace(objects, i, right);
    }
}

static void qsort_descent_inplace(std::vector<Object> &objects)
{
    if (!objects.empty())
        qsort_descent_inplace(objects, 0, objects.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<Object> &faceobjects, std::vector<int> &picked, float nms_threshold, bool agnostic = false)
{
    picked.clear();
    const int n = faceobjects.size();
    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
        areas[i] = faceobjects[i].rect.area();

    for (int i = 0; i < n; i++)
    {
        const Object &a = faceobjects[i];
        int keep = 1;
        for (int j : picked)
        {
            const Object &b = faceobjects[j];
            if (!agnostic && a.label != b.label)
                continue;
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[j] - inter_area;
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }
        if (keep)
            picked.push_back(i);
    }
}

static inline float clampf(float d, float min, float max)
{
    return std::max(min, std::min(max, d));
}

static void parse_yolov11_detections(float *inputs, float conf_thres, int num_channels, int num_anchors, int num_labels, int img_w, int img_h, std::vector<Object> &objects)
{
    std::vector<Object> detections;
    cv::Mat output(num_channels, num_anchors, CV_32F, inputs);
    output = output.t();

    for (int i = 0; i < num_anchors; i++)
    {
        const float *row = output.ptr<float>(i);
        const float *bbox = row;
        const float *cls = row + 4;
        const float *max_cls = std::max_element(cls, cls + num_labels);
        float score = *max_cls;
        if (score > conf_thres)
        {
            float x = bbox[0], y = bbox[1], w = bbox[2], h = bbox[3];
            float x0 = clampf(x - 0.5f * w, 0.f, (float)img_w);
            float y0 = clampf(y - 0.5f * h, 0.f, (float)img_h);
            float x1 = clampf(x + 0.5f * w, 0.f, (float)img_w);
            float y1 = clampf(y + 0.5f * h, 0.f, (float)img_h);

            Object obj;
            obj.rect = cv::Rect_<float>(x0, y0, x1 - x0, y1 - y0);
            obj.label = max_cls - cls;
            obj.prob = score;
            detections.push_back(obj);
        }
    }
    objects = detections;
}

class YoloV11
{
private:
    ncnn::Net net;
    std::vector<std::string> class_names;
    std::unique_ptr<ncnn::Extractor> ex;
    float fconf_thres, fnms_thres;

public:
    YoloV11(const std::string &model_path, const std::vector<std::string> &names, bool useVulkan = true, bool int8=false, float fconf_thres = 0.25f, float fnms_thres = 0.45f)
    {
        class_names = names;
        net.opt.use_vulkan_compute = useVulkan; 
        printf("[CONFIG] INT8=%d conf=%.2f nms=%.2f\n", int8, fconf_thres, fnms_thres);
        net.opt.use_bf16_storage = true; 
        if(int8){
            net.opt.use_int8_inference = true;
            net.opt.use_fp16_arithmetic = false;
        }else{
            net.opt.use_int8_inference = false;
            net.opt.use_fp16_arithmetic = true;
        }      
        net.opt.use_packing_layout = true;      
        net.opt.num_threads = 4;

        net.load_param((model_path + ".param").c_str());
        net.load_model((model_path + ".bin").c_str());
        this->fconf_thres = fconf_thres;
        this->fnms_thres = fnms_thres;
    }

    int detect(const cv::Mat &bgr, std::vector<Object> &objects)
    {
        
        const int target_size = 480;
        const float conf_thres = fconf_thres;
        const float nms_thres = fnms_thres;
        int img_w = bgr.cols, img_h = bgr.rows;
        int w = img_w, h = img_h;
        float scale = (w > h) ? (float)target_size / w : (float)target_size / h;
        w = w * scale;
        h = h * scale;
        if (w > h)
            w = target_size;
        else
            h = target_size;

        ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, img_w, img_h, w, h);
        int wpad = (target_size + MAX_STRIDE - 1) / MAX_STRIDE * MAX_STRIDE - w;
        int hpad = (target_size + MAX_STRIDE - 1) / MAX_STRIDE * MAX_STRIDE - h;
        ncnn::Mat in_pad;
        ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 114.f);

        const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
        in_pad.substract_mean_normalize(0, norm_vals);

        if (!ex)
            ex = std::make_unique<ncnn::Extractor>(net.create_extractor());

        auto t0 = std::chrono::high_resolution_clock::now();
        ex->input("in0", in_pad);
        ncnn::Mat out;
        ex->extract("out0", out);

        auto t1 = std::chrono::high_resolution_clock::now();

        printf("[INFO] out shape: w=%d, h=%d, c=%d\n", out.w, out.h, out.c);

        std::vector<Object> proposals;
        parse_yolov11_detections((float *)out.data, conf_thres, out.h, out.w, out.h - 4, in_pad.w, in_pad.h, proposals);

        qsort_descent_inplace(proposals);
        std::vector<int> picked;
        nms_sorted_bboxes(proposals, picked, nms_thres);

        objects.resize(picked.size());
        for (size_t i = 0; i < picked.size(); i++)
        {
            objects[i] = proposals[picked[i]];
            float x0 = (objects[i].rect.x - wpad / 2) / scale;
            float y0 = (objects[i].rect.y - hpad / 2) / scale;
            float x1 = (objects[i].rect.x + objects[i].rect.width - wpad / 2) / scale;
            float y1 = (objects[i].rect.y + objects[i].rect.height - hpad / 2) / scale;
            x0 = clampf(x0, 0.f, img_w - 1.f);
            y0 = clampf(y0, 0.f, img_h - 1.f);
            x1 = clampf(x1, 0.f, img_w - 1.f);
            y1 = clampf(y1, 0.f, img_h - 1.f);
            objects[i].rect = cv::Rect_<float>(x0, y0, x1 - x0, y1 - y0);
        }

        auto t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> infer_ms = t1 - t0;
        std::chrono::duration<double, std::milli> post_ms = t2 - t1;
        printf("[TIME] Inference: %.2f ms | Postprocess: %.2f ms\n", infer_ms.count(), post_ms.count());
        return 0;
    }

    void save_result(const cv::Mat &bgr, const std::vector<Object> &objects)
    {
        cv::Mat image = bgr.clone();
        for (const auto &obj : objects)
        {
            cv::rectangle(image, obj.rect, cv::Scalar(0, 255, 0), 2);
            char text[128];
            sprintf(text, "%s %.1f%%", class_names[obj.label].c_str(), obj.prob * 100);
            cv::putText(image, text, cv::Point(obj.rect.x, obj.rect.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
        }
        cv::imwrite("output.jpg", image);
        printf("[INFO] Saved result as output.jpg (%zu objects)\n", objects.size());
    }
};

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        printf("Usage: %s [imagepath] [modelpath] [int8=0/1] [conf=0.25] [nms=0.45]\n", argv[0]);
        return -1;
    }

    std::string image_path = argv[1];
    std::string model_path = argv[2];
    bool use_int8 = false;
    float conf_thres = 0.25f;
    float nms_thres = 0.45f;
    if(argc>3) use_int8 = std::stoi(argv[3]);
    if(argc>4) conf_thres = std::stof(argv[4]);
    if(argc>5) nms_thres = std::stof(argv[5]);

    cv::Mat img = cv::imread(image_path);
    if (img.empty())
    {
        fprintf(stderr, "Failed to read image: %s\n", image_path.c_str());
        return -1;
    }

    std::vector<std::string> class_names = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush"};

    YoloV11 yolo(model_path, class_names, true, use_int8, conf_thres, nms_thres);
    std::vector<Object> objects;
    yolo.detect(img, objects);
    yolo.save_result(img, objects);
    return 0;
}
