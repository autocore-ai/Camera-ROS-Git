#ifndef __YOLO_HELPER_H__
#define __YOLO_HELPER_H__

#include "yolo.h"

class YoloHelper
{
private:
    /* data */
public:
    YoloHelper(/* args */);
    ~YoloHelper();

    std::vector<BBoxInfo> do_inference(const cv::Mat& image_org);    
    void parse_config_params(int argc, char** argv);

    cv::Mat get_marked_image(int imageIndex);
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
};

#endif