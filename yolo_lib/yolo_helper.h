#ifndef __YOLO_HELPER_H__
#define __YOLO_HELPER_H__

#include "yolo.h"
#include <string>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

enum Color
{
    background = 0,
    green,
    yellow,
    red
};

class YoloHelper
{
private:
    /* data */
public:
    YoloHelper(/* args */);
    ~YoloHelper();

    std::vector<BBoxInfo> do_inference(const cv::Mat& image_org,bool simu=false);    
    std::vector<BBoxInfo> judge_red_yellow_green(const cv::Mat& image_org);  
    void parse_config_params(int argc, char** argv);

    cv::Mat get_marked_image(int imageIndex);

    int judge_lights_color(cv::Mat test_img);
    int judge_lights_color(std::string full_imgfile);
    std::vector<BBoxInfo> judge_red_yellow_green(cv::Mat& image_org);
public:
    std::unique_ptr<Yolo> m_inferNet=nullptr;
//config
private:
    bool m_decode;
    bool m_doBenchmark;
    bool m_viewDetections;
    bool m_saveDetections;
    std::string m_saveDetectionsPath;
    uint m_batchSize;
    bool m_shuffleTestSet;
//
private:
    std::vector<DsImage> dsImages;    

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