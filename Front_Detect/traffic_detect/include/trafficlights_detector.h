#ifndef __TRAFFIC_LIGHTS_DETECTOR__
#define __TRAFFIC_LIGHTS_DETECTOR__

#include <vector>
#include <string>
using namespace std;

#include <ros/ros.h>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "dashboard_msgs/cam_cmd.h"
#include "dashboard_msgs/Cmd.h"
#include "dashboard_msgs/Proc.h"
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include "std_msgs/UInt8.h"
#include <std_msgs/Int32.h>
#include <std_msgs/Int32MultiArray.h>

#include "yolo_helper.h"

struct LightsStatusSimu
{
    bool find_red = false;
    bool find_yellow = false;
    bool find_green = false;

    void clear()
    {
        find_red = false;
        find_yellow = false;
        find_green = false;
    }
};


class TrafficLightsDetector
{
public:
    TrafficLightsDetector();
    ~TrafficLightsDetector();

public:
    //
    void on_recv_frame(const sensor_msgs::Image& image_source);

    //
    void init(int argc,char** argv);
    
    //
    void set_current_frame(cv::Mat frame);

    //处理收到的待检测帧
    void process_frame();

private:
    //推理前预处理,不同模型可能会不一样
    void preprocess_frame();

    //roi参数解析
    std::vector<int> visplit(std::string str,std::string pattern);

    //.launch文件参数加载
    bool load_parameters();

    //ros subscriber/publisher
    bool init_ros(int argc,char** argv);
private:
    unsigned char status_encode();
private:
    //收到的待处理图像
    cv::Mat m_frame;
    //预处理后准备推理的图像
    cv::Mat m_frame_model_input;
private:    
    YoloHelper m_yolo_helper;

    //当前frame的最终分析结果
    LightsStatusSimu m_lights_status_simu;

private: 
  
    std::string m_image_source_topic;
    std::string m_status_code_topic;
    std::string m_image_detected_topic;

    ros::Publisher pub_status_code;
    ros::Publisher pub_image_detected;
    ros::Subscriber sub_image_raw;

    int m_seq = 0;
    bool m_simu_mode = false;
};


class RosMsgHelper
{
public:
    RosMsgHelper();
    ~RosMsgHelper();
private:    

};

#endif