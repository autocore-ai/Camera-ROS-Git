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

#include "autoreg_msgs/park_obj.h"
#include "yolo_helper.h"
#include "carpark_mgr.h"
#include "auto_yulan_sdk.h"
#include "trt_utils.h"

struct ParkAnchor
{
    cv::Point ltop;
    cv::Point lbot;
    cv::Point rtop;
    cv::Point rbot;
   
    std::string to_string()
    {
        std::ostringstream stringStream;
        stringStream << "park anchor as below\n";
        stringStream << "ltop:"<<ltop.x<<","<<ltop.y<<'\n';
        stringStream << "lbot:"<<lbot.x<<","<<lbot.y<<'\n';
        stringStream << "rtop:"<<rtop.x<<","<<rtop.y<<'\n';
        stringStream << "rbot:"<<rbot.x<<","<<rbot.y<<'\n';

        return stringStream.str();   
    }
};

class CarParkMgr;

class VpsDetector
{
public:
    VpsDetector();
    ~VpsDetector();

    void init(int argc,char **argv);    

    void test();
private:
    //
    void on_pose_msg(const geometry_msgs::PoseStamped &pose_stamp);
    //
    void update_carpose();
    //
    void on_recv_frame(const sensor_msgs::Image &image_source);
    //
    void process_frame();
    //
    void init_ros(int argc,char **argv);
private:
    //当前待处理图像
    cv::Mat m_frame;
    //模型的输入
    cv::Mat m_frame_input;  
private:    
    YoloHelper m_yolo_helper;

    ATCMapper*      m_p_atc_mapper=nullptr;
    ParkInfo*        m_p_park=nullptr;  
    //ATCParkTracker* m_park_tracker = nullptr;
    CarParkMgr*     m_p_carpark_mgr = nullptr;
    
    geometry_msgs::PoseStamped m_pose;  //当前车身姿态 
private:
    ros::Subscriber m_sub_image_raw;
    std::string     m_sub_topic_image_raw_from_camera;
    
    ros::Subscriber m_sub_image_pose;
    std::string     m_image_pose_topic;

    ros::Publisher  m_pub_image_obj;    //发送park的坐标,分类,置信度等.
    std::string     m_image_object_topic;
    autoreg_msgs::park_obj m_parkobj_msg;

    
    ros::Publisher  m_pub_image_raw;    //可视化
    std::string     m_image_raw_topic;
    long m_seq = 0;

    ros::Publisher  m_pub_vps_active;    //心跳
    std::string     m_vps_active_topic;
private:    
    //裕兰环视车位数据集图片尺寸
    int  m_yulan_w = 810;
    int  m_yulan_h = 1080;

    //在 x 方向上每个像素代表 21.9mm， y 方向每个像素代表 20.1mm
    float  m_delta_x = 0.0219;// yulan: 0.0219 simu:0.0285  
    float  m_delta_y = 0.0201; //yulan: 0.0201 simu 0.0287 
 private:
    //在图片上绘制出车位框 
    bool draw_park_on_img(cv::Mat &img);
    
    //
    void get_nonfree_parks(const std::vector<BBoxInfo>& boxes_info,std::vector<BBoxInfo>& non_free_park_boxes);

    //
    void get_free_park_boxes(const std::vector<BBoxInfo>& boxes_info,const std::vector<BBoxInfo>& non_free_park_boxes,std::vector<BBoxInfo>& free_park_boxes);

    //expandp_bndbox
    void expand_bndbox(cv::Rect src_rect,cv::Rect &dst_rect,float ratio,int img_w,int img_h);

    //通过iou判断车位是否有效 与任一无效车位iou超过阈值,则认为是无效车位
    bool is_effective_park(const std::vector<BBoxInfo>& notfreebox_list,const BBox& freepark_box,float iou_thresold=0.5);

    //get iou ratio between two bndingbox
    float get_box_overlap_ratio(const BBox& bbox1,const BBox& bbox2);

    //
    bool park_anchor_filter(ParkAnchor src_park,cv::Point bbox_center,float thresold,float max_lw_ratio,float min_lw_ratio/*=1.0*/);

    //
    void get_two_crossover_from_line_to_rect(float a,float b,float c,cv::Rect src_rect,std::vector<cv::Point> &crossover_list);
    //
    //图像裁剪
    void img_decode(cv::Mat raw_data, cv::Mat &dst_img,float &dst_delta_x,float &dst_delta_y);

    // image angle expanding
    void imrotate(cv::Mat& img,cv::Mat& newIm, double angle);

    // get two line's crosspoint
    cv::Point2f getCrossPoint(cv::Vec4i LineA, cv::Vec4i LineB);

    // get line' length  获取两点之间的距离
    float get_len(cv::Point begin,cv::Point end);

private:
    //
    void get_lines(const cv::Mat& roi_img,std::vector<cv::Vec4i>& lines,int box_idx);
    //
    bool park_edge_detect(cv::Mat src_img, cv::Mat &dst_img);

    // remove image border edges
    bool depress_fringe_grad(cv::Mat src_img, cv::Mat &dst_img, int shrink);

    //
    void adjust_coordination(const cv::Rect& expand_rect,ParkAnchor& draw_anchor);
    //
    void process_bbox(const std::vector<BBoxInfo>& free_park_boxes);
    //
    bool process_carpark(const cv::Vec4i& line,const cv::Rect& expand_rect,ParkAnchor& draw_anchor);
    //
    void process_carpark1(const cv::Point& begin,const cv::Point& end,
                                 const cv::Rect& expand_rect,
                                 ParkAnchor& draw_anchor);
    //
    void process_carpark2(const cv::Point& begin,
                          const cv::Point& end,
                          const cv::Rect& expand_rect,
                          ParkAnchor& draw_anchor);
    //
    bool process_carpark3(const cv::Point& begin,
                          const cv::Point& end,
                          const cv::Rect& expand_rect,ParkAnchor& draw_anchor);

    //
    void process_curr_park(const ParkAnchor& draw_anchor,const BBoxInfo& box_info);


    //发送车位坐标
    void pub_parkobj_msg();

    //发送绘制了车位框的图片
    void pub_img(const cv::Mat& detect_show);

    //
    void insert_crossover(std::vector<cv::Point> &crossover_list,cv::Point src_point);

    // 
    bool get_park_anchor(std::vector<cv::Point> anchor_list, ParkAnchor &dst_park, float offset);
    
//for debug
public:
    long m_frame_counts = 0;
    int m_frame_counts_divide = 60;
    bool m_test = false;

    float m_bbox_expand_ratio = 0.1;

    string m_save_dir = "/home/nano/suchang/src/Camera-ROS-Git/Surround_Detect/vps_detect/models/data/detections/";
    void save_img(const string& img_name,const cv::Mat& img);
//park anchor filter param
private:
    float m_thresold = 200.0;
    float m_max_lw_ratio = 4.5;
    float m_min_lw_ratio = 1.2;

};

#endif
