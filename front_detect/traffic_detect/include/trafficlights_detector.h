#ifndef __TRAFFIC_LIGHTS_DETECTOR__
#define __TRAFFIC_LIGHTS_DETECTOR__

#include <vector>
#include <string>
using namespace std;

#include <ros/ros.h>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
//#include "std_msgs/UInt8.h"
//#include <std_msgs/Int32.h>
//#include <std_msgs/Int32MultiArray.h>
#include "autoware_msgs/TrafficLight.h"
#include "autoware_msgs/Signals.h"

#include "model_helper.h"
#include "Context.h"

enum class LightColor
{
    red = 2,
    yellow = 1,
    green = 0
};

class TrafficLightsDetector
{
public:
    TrafficLightsDetector();
    TrafficLightsDetector(string model_path);
    ~TrafficLightsDetector();
public:
    //
    void on_recv_frame(const sensor_msgs::Image& image_source);

    //
    void on_recv_signal_roi(const autoware_msgs::Signals::ConstPtr &extracted_pos);

    //
    void init(int argc,char** argv);
    
    //
    void set_current_frame(cv::Mat frame);

    //
    void process_frame();
private:
    //
    void preprocess_frame(const cv::Mat& frame);

    //.launch文件参数加载
    bool load_parameters();

    //ros subscriber/publisher
    bool init_ros(int argc,char** argv);

    unsigned char status_encode();

    void publish_traffic_light(std::vector<Context> contexts);
private:
    //
    std_msgs::Header frame_header_;
    //
    cv::Mat frame_;
    //
    cv::Mat frame_model_input_;
    //
    string model_path_;
private:    
    MobilenetV1 mv1_;

    //result of model inference
    LightColor lights_color_;

    // The vector of data structure to save traffic light state, position, ...etc
    std::vector<Context> contexts_;

private:  
    std::string image_source_topic_;
    std::string signal_state_topic_;
    std::string image_detected_topic_;

    ros::Publisher signal_state_puber_;
    ros::Publisher image_detected_puber_;
    
    ros::Subscriber image_raw_suber_;
    ros::Subscriber roi_signal_suber_;

    const int32_t kTrafficLightRed = 0;
    const int32_t kTrafficLightGreen = 1;
    const int32_t kTrafficLightUnknown = 2;
};
#endif