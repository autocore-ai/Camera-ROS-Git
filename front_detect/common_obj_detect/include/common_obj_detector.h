#ifndef __COMMON_OBJ_DETECTOR__
#define __COMMON_OBJ_DETECTOR__

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
#include "std_msgs/UInt8.h"
#include <std_msgs/Int32.h>
#include <std_msgs/Int32MultiArray.h>
#include "model_helper.h"

class CommonObjDetector
{
public:
    CommonObjDetector();
    CommonObjDetector(string model_path);
    ~CommonObjDetector();
public:
    //
    void on_recv_frame(const sensor_msgs::Image& image_source);

    //
    void init(int argc,char** argv);
    
    //
    void set_current_frame(cv::Mat frame);

    //
    void process_frame();

private:
    void preprocess_frame();

    //
    bool load_parameters();

    //ros subscriber/publisher
    bool init_ros(int argc,char** argv);
private:
    unsigned char status_encode();
private:
    //
    std_msgs::Header frame_header_;
    //
    cv::Mat frame_;
    //
    cv::Mat frame_model_input_;
private:    
    MobilenetV1SSD mv1ssd_;
private:  
    std::string image_source_topic_;
    std::string image_detected_topic_;
    std::string detected_objs_pubtopic_;

    ros::Publisher pub_image_detected_;
    ros::Subscriber sub_image_raw_;
    ros::Publisher pub_detected_objs_;

    std::string model_path_;
};
#endif