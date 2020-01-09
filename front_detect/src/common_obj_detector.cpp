#include "common_obj_detector.h"
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include "autoreg_msgs/obj.h"
#include "autoreg_msgs/obj_array.h"
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <chrono>
using namespace cv;

CommonObjDetector::CommonObjDetector()
{

}

CommonObjDetector::~CommonObjDetector()
{

}

void CommonObjDetector::set_current_frame(cv::Mat frame)
{
    frame_ = frame;
}

void CommonObjDetector::init()
{
    init_ros();

    mv1ssd_.init(model_path_);
} 

bool CommonObjDetector::init_ros()
{
    ros::NodeHandle node;

    bool ret = load_parameters();
    
    sub_image_raw_ = node.subscribe(image_source_topic_, 1, &CommonObjDetector::on_recv_frame,this);
    pub_image_detected_ = node.advertise<sensor_msgs::Image>(image_detected_topic_,1);
    pub_detected_objs_ = node.advertise<autoreg_msgs::obj_array>(detected_objs_pubtopic_,1);

    return ret;
}

void CommonObjDetector::on_recv_frame(const sensor_msgs::Image& image_source)
{
    ROS_INFO("CommonObjDetector::on_recv_frame!");

    frame_header_ = image_source.header;

    cv_bridge::CvImagePtr cv_image = cv_bridge::toCvCopy(image_source, "bgr8");
    cv::Mat frame = cv_image->image; 
    set_current_frame(frame);
    
    process_frame();
}

bool CommonObjDetector::load_parameters()
{
    ros::NodeHandle private_nh("~");
    
    ROS_INFO("****CommonObjDetector params****");

    private_nh.param<std::string>("co_image_source_topic", image_source_topic_, "/usb_cam/image_raw");
    ROS_INFO("Setting image_source_topic to %s", image_source_topic_.c_str());
     
    private_nh.param<std::string>("co_image_detected_topic", image_detected_topic_, "/common_obj/image_detected");
    ROS_INFO("Setting image_detected_topic to %s", image_detected_topic_.c_str());

    private_nh.param<std::string>("co_detected_objs_pubtopic", detected_objs_pubtopic_, "/common_obj/detected_objs");
    ROS_INFO("Setting detected_objs_pubtopic to %s", detected_objs_pubtopic_.c_str());

    private_nh.param<std::string>("co_model_path", model_path_, "");
    ROS_INFO("Setting model_path to %s", model_path_.c_str());

    return true;
}

void CommonObjDetector::preprocess_frame()
{
    int width = mv1ssd_.get_width();
    int height = mv1ssd_.get_height();
    //cv::cvtColor(frame_, frame_, CV_RGB2BGR);
    cv::resize(frame_,frame_model_input_,cv::Size(width,height));
    cv::cvtColor(frame_model_input_,frame_model_input_,CV_RGB2BGR);

    ROS_INFO("resize img to %d x %d", width,height);
}

//处理收到的待检测帧
void CommonObjDetector::process_frame()
{
    auto begin = std::chrono::system_clock::now();

    preprocess_frame();
    std::vector<uint8_t> input;
    int width = mv1ssd_.get_width();
    int height = mv1ssd_.get_height();
    if(frame_model_input_.isContinuous())
    {
        input.insert(input.end(),frame_model_input_.data,frame_model_input_.data + width*height*3);
    }

    autoreg_msgs::obj_array detected_objs_msg;
    vector<BBoxInfo> boxes = mv1ssd_.inference(input);
    int w = frame_.cols;
    int h = frame_.rows;
    for(const BBoxInfo& b : boxes)
    {
        float xmin = w * b.xmin_ + 1.0;
        float ymin = h * b.ymin_ + 1.0;
        float xmax = w * b.xmax_ + 1.0;
        float ymax = h * b.ymax_ + 1.0;

        rectangle(frame_, cvPoint(xmin, ymin), cvPoint(xmax, ymax), Scalar(0, 0, 255), 1, 1, 0);

        autoreg_msgs::obj obj_msg;
        obj_msg.xmin = xmin;
        obj_msg.ymin = ymin;
        obj_msg.xmax = xmax;
        obj_msg.ymax = ymax;
        obj_msg.cls_id = b.cls_id_;
        obj_msg.score = b.score_;

        detected_objs_msg.objs.push_back(obj_msg);
    }
    pub_detected_objs_.publish(detected_objs_msg);

    cv_bridge::CvImage  cv_img(frame_header_, "bgr8", frame_);
    sensor_msgs::ImagePtr img_with_objs = cv_img.toImageMsg();
    pub_image_detected_.publish(img_with_objs);

    auto end = std::chrono::system_clock::now();
    auto elsp = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
    std::cout << "process_frame:" << elsp.count() << " ms"<<std::endl;
}


