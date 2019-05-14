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

struct LightsStatus
{
    bool go=false;
    bool goLeft=false;
    bool goRight=true;
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

    //判断图像是否可被裁剪
    bool roi_region_is_valid();
private:
    //roi参数解析
    std::vector<int> visplit(std::string str,std::string pattern);

    //.launch文件参数加载
    bool load_parameters();

    //ros subscriber/publisher
    bool init_ros(int argc,char** argv);
private:
    unsigned char status_encode(bool go_up,bool go_left,bool go_right);
private:
    //roi参数. 图像裁剪用.  含义:从m*n(width*high)的图片中,以(x,y)为起点,裁剪出w*h的图片:
    string m_roi_region;
    int m_x;int m_y;int m_w;int m_h;

    //当前待处理图像
    cv::Mat m_frame;
    //裁剪后的图像
    cv::Mat m_frame_roi;
private:    
    YoloHelper m_yolo_helper;
    //float m_prob_threshold = 0.5;//yolov3-tiny_for_trafficlights.txt中配置

    //当前frame的最终分析结果
    LightsStatus m_lights_status;

private: 
    std::string m_refresh_epoch;
    std::string m_image_source_topic;
    std::string m_cam_cmd_topic;
    std::string m_status_code_topic;
    std::string m_image_raw_topic;
    std::string m_traffic_status_topic;
    std::string m_traffic_active_topic;

    ros::Publisher pub_image_raw;
    ros::Publisher pub_status_code;
    ros::Publisher pub_traffic_status;
    ros::Publisher pub_traffic_active;
    ros::Subscriber sub_image_raw;
    ros::Subscriber sub_cam_cmd;

    int m_seq = 0;

};


class RosMsgHelper
{
public:
    RosMsgHelper();
    ~RosMsgHelper();
private:    

};

#endif