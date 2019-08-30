#ifndef __YOLO_HELPER_H__
#define __YOLO_HELPER_H__

//#include "yolo.h"
#include <string>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <dnndk/dnndk.h>
#include "utils.h"

using namespace cv;

enum Color
{
    background = 0,
    green,
    yellow,
    red
};

struct BBox
{
    float x1, y1, x2, y2;
};

struct BBoxInfo
{
    BBox box;
    int label;
    int classId; // For coco benchmarking
    float prob;
};

class YoloHelper
{
#define INPUT_NODE "layer0_conv"
public:
    YoloHelper(/* args */);
    ~YoloHelper();

    std::vector<BBoxInfo> do_inference(const cv::Mat &image_org, bool simu = false);
    int judge_lights_color(cv::Mat test_img);
    int judge_lights_color(std::string full_imgfile);
    std::vector<BBoxInfo> judge_red_yellow_green(cv::Mat &image_org);

    void init();
    void runYOLO(DPUTask *task, Mat &img);
    void setInputImageForYOLO(DPUTask *task, const Mat &frame, float *mean);
    void postProcess(DPUTask *task, Mat &frame, int sWidth, int sHeight);
    inline DPUTask *get_task()
    {
        return m_task;
    }

    void detect(vector<vector<float>> &boxes, 
                  vector<float> result,
                  int channel, 
                  int height, 
                  int weight, 
                  int num, 
                  int sh, 
                  int sw);

    void correct_region_boxes(vector<vector<float>> &boxes,
                                       int n,
                                       int w, 
                                       int h, 
                                       int netw, 
                                       int neth, 
                                       int relative = 0);

private:
    DPUKernel *m_kernel = nullptr;
    DPUTask *m_task = nullptr;

//utils
private:
    std::string type2str(int type);
    float get_percentage(cv::Mat img_hsv,
                         int iLowH,
                         int iHighH,
                         int iLowS,
                         int iHighS,
                         int iLowV,
                         int iHighV);
};

#endif