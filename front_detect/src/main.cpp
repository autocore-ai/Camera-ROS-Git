#include "model_helper.h"
#include "front_detector.h"
#include <dirent.h>
#include <fstream>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace std;
using namespace cv;

void test()
{
    string co_model_path = "/home/user/workspace/src/Camera-ROS-Git/mobilenet_ssd_v1_coco_quant_postprocess_edgetpu.tflite";
    string tl_model_path = "/home/user/workspace/src/Camera-ROS-Git/trafficlights_mobilenetv1_edgetpu.tflite";

    MobilenetV1 mv1;
    mv1.init(tl_model_path);
    
    MobilenetV1SSD  mv1ssd;
    mv1ssd.init(co_model_path);

    string imgname = "/home/user/test_data/cat_720p.jpg";
    Mat img = imread(imgname,cv::IMREAD_UNCHANGED);
    cv::cvtColor(img, img, CV_RGB2BGR);

    Mat img_resized_tl;
    resize(img, img_resized_tl, Size(224,224));
    std::vector<uint8_t> input;
    if(img_resized_tl.isContinuous())
    {
        input.insert(input.end(),img_resized_tl.data,img_resized_tl.data + 224*224*3);
    }  

    auto begin = std::chrono::system_clock::now();
    int index = mv1.inference(input);
    auto end = std::chrono::system_clock::now();
    auto elsp = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
    std::cout << "mobilenetv1:inference one frame using :" << elsp.count() << " ms"<<std::endl;

    Mat img_resized_co;
    resize(img, img_resized_co, Size(300,300));
    std::vector<uint8_t> input2;
    if(img_resized_co.isContinuous())
    {
        input2.insert(input2.end(),img_resized_co.data,img_resized_co.data + 300*300*3);
    }  
    begin = std::chrono::system_clock::now();
    mv1ssd.inference(input);
    end = std::chrono::system_clock::now();
    elsp = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
    std::cout << "mv1ssd:inference one frame using :" << elsp.count() << " ms"<<std::endl;
}

int main(int argc, char *argv[])
{
    ros::init(argc, argv, "front_detect");

    // for(int i = 0;i<10;i++)
    // {
    //     test();
    // }
    // return 0;
    
    FrontDetector front_detector;
    front_detector.init();
    
    ros::spin();
    
    return 0;
}
