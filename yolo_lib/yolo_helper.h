#ifndef __YOLO_HELPER_H__
#define __YOLO_HELPER_H__

//#include "yolo.h"
#include <string>
#include <vector>
#include <map>
#include <dnndk/dnndk.h>
#include "utils.h"

using namespace cv;
using namespace std;

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
    int classId; 
    float prob;
};

class YoloHelper
{
#define INPUT_NODE "layer0_conv"
public:
    YoloHelper();
    ~YoloHelper();

    std::vector<BBoxInfo> do_inference(const Mat &image_org, bool simu = false);
    void init();
    void release_resource();
    void runYOLO(DPUTask *task, Mat &img);
    inline DPUTask *get_task()
    {
        return m_task;
    }
    void test_parse_cfgfile(const string cfgfilepath);
    const vector<BBoxInfo>& get_inference_result()
    {
        return m_boxes;
    };
private:  
    vector<map<string,string>> parse_cfgfile(const string cfgfilepath);
    void setInputImageForYOLO(DPUTask *task, const Mat &frame, float *mean);
    void postProcess(DPUTask *task, Mat &frame, int sWidth, int sHeight);

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

    int m_classification_cnt = 4;
    int m_acnhor_count = 3;

    vector<BBoxInfo> m_boxes;
    
    
    
//simulation environment
public:
    int judge_lights_color(Mat test_img);
    int judge_lights_color(string full_imgfile);
    std::vector<BBoxInfo> judge_red_yellow_green(const Mat &image_org);    
    float get_percentage(cv::Mat img_hsv,
                     int iLowH,
                     int iHighH,
                     int iLowS,
                     int iHighS,
                     int iLowV,
                     int iHighV);   
};

#endif