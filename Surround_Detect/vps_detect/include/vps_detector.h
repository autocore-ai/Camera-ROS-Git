#ifndef __VPS_DETECTOR__
#define __VPS_DETECTOR__

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


class VpsDetector
{
public:
    VpsDetector();
    ~VpsDetector();

    void init(int argc,char **argv)
    {
        m_yolo_helper.parse_config_params(argc,argv);

        init_ros(argc, argv);
    }
    
private:
    //
    void on_pose_msg(const geometry_msgs::PoseStamped &pose_stamp)
    {
        m_pose.pose.position = pose_stamp.pose.position;
        m_pose.pose.orientation = pose_stamp.pose.orientation;
    }

    //
    void on_recv_frame(const sensor_msgs::Image &image_source);

    void init_ros(int argc,char **argv)
    {
        ros::init(argc,argv,"yolo");
        ros::NodeHandle node;
    	ros::NodeHandle private_nh("~");

        private_nh.param<std::string>("image_source_topic", m_image_raw_topic, "/usb_cam/image_raw");
        m_sub_image_raw = node.subscribe(m_image_raw_topic, 1, &VpsDetector::on_recv_frame,this);

        private_nh.param<std::string>("image_pose_topic",m_image_pose_topic,"/gnss_pose");
        ROS_INFO("Setting image_pose_topic to %s",m_image_pose_topic.c_str());
        m_sub_image_pose = node.subscribe(m_image_pose_topic, 1, &VpsDetector::on_pose_msg,this);
        
    }
private:
    //当前待处理图像
    cv::Mat m_frame;
    //模型的输入
    cv::Mat m_frame_input;  
private:    
    YoloHelper m_yolo_helper;

    ATCMapper *p_atc_mapper=nullptr;
    ATCPark *p_new_park=nullptr;
    
    geometry_msgs::PoseStamped m_pose;

private:
    ros::Subscriber m_sub_image_raw;
    std::string     m_image_raw_topic;
    
    ros::Subscriber m_sub_image_pose;
    std::string     m_image_pose_topic;

private:    
    //裕兰环视车位数据集图片尺寸
    int  m_yulan_w = 810;
    int  m_yulan_h = 1080;
        
    float  m_delta_x = 0.0219;// yulan: 0.0219 simu:0.0285  
    float  m_delta_y = 0.0201; //yulan: 0.0201 simu 0.0287 
 private:
    //expandp_bndbox
    void expand_bndbox(cv::Rect src_rect,cv::Rect &dst_rect,float ratio,int img_w,int img_h);
    //通过iou判断车位是否有效 与任一无效车位iou超过阈值,则认为是无效车位
    bool is_effective_park(std::vector<Box> box_list,Box src_box,float iou_thresold=0.5);
    //get iou ratio between two bndingbox
    float get_box_overlap_ratio(Box bbox1,Box bbox2);
    //
    bool park_anchor_filter(ParkAnchor src_park,cv::Point bbox_center,float thresold,float max_lw_ratio,float min_lw_ratio/*=1.0*/);
    //
    void get_two_crossover_from_line_to_rect(float a,float b,float c,cv::Rect src_rect,std::vector<cv::Point> &crossover_list);
    //
private:
    //图像裁剪
    void img_decode(cv::Mat raw_data, cv::Mat &dst_img,float &dst_delta_x,float &dst_delta_y);

    // image angle expanding
    void imrotate(cv::Mat& img,cv::Mat& newIm, double angle);

    // get two line's crosspoint
    cv::Point2f getCrossPoint(cv::Vec4i LineA, cv::Vec4i LineB);

    // get line' length  获取两点之间的距离
    float get_len(cv::Point begin,cv::Point end);

    // 可视化 在图片上画出车位
    void tracker_vis(cv::Mat &img,std::vector<ATCVisPark> vis_trks);
};

#endif
