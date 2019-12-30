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
#include "autoware_msgs/TrafficLight.h"
#include "autoware_msgs/Signals.h"

#include "model_helper.h"
#include "Context.h"

class TrafficLightsDetector
{
public:
    TrafficLightsDetector();
    ~TrafficLightsDetector();
public:
    //
    void init(int argc,char** argv);

    //
    void on_recv_frame(const sensor_msgs::Image& image_source);

    //
    void on_recv_signal_roi(const autoware_msgs::Signals::ConstPtr &extracted_pos);
private:
    //ros subscriber/publisher
    bool init_ros(int argc,char** argv);
    
    //read .launch file
    bool load_parameters();
    
    //
    void preprocess_frame(const cv::Mat& frame);

    void determin_state(LightState in_current_state,Context& in_out_signal_context);

    void publish_traffic_light(std::vector<Context> contexts);

    void publish_image(std::vector<Context> contexts);

    //
    void set_current_frame(cv::Mat frame);
private:
    //
    std_msgs::Header frame_header_;
    //
    cv::Mat frame_;
    //
    cv::Mat frame_model_input_;
    //
    string model_path_;
    //
    int change_state_threshold_=5;
private:    
    MobilenetV1 mv1_;

    // The vector of data structure to save traffic light state, position, ...etc
    std::vector<Context> contexts_;
private:  
    std::string image_source_topic_;
    std::string image_detected_topic_;

    ros::Publisher signal_state_puber_;
    ros::Publisher image_detected_puber_;

    ros::Subscriber image_raw_suber_;
    ros::Subscriber roi_signal_suber_;

    const int32_t kTrafficLightRed = 0;
    const int32_t kTrafficLightGreen = 1;
    const int32_t kTrafficLightUnknown = 2;

     /* Light state transition probably happen in Japanese traffic light */
    const LightState kStateTransitionMatrix[4][4] = {
        /* current: */
        /* GREEN   , YELLOW    , RED    , UNDEFINED  */
        /* -------------------------------------------  */
        {GREEN     , YELLOW    , YELLOW    , GREEN}  ,  /* | previous = GREEN */
        {UNDEFINED , YELLOW    , RED       , YELLOW} ,  /* | previous = YELLOW */
        {GREEN     , RED       , RED       , RED}    ,  /* | previous = RED */
        {GREEN     , YELLOW    , RED       , UNDEFINED} /* | previous = UNDEFINED */
    };
};

#endif