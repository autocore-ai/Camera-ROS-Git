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
private:
    //
    void pose_callback(const geometry_msgs::PoseStamped &pose_stamp)
    {
        g_pose_stamp.pose.position = pose_stamp.pose.position;
        g_pose_stamp.pose.orientation = pose_stamp.pose.orientation;
    }

    //
    void on_recv_frame(const sensor_msgs :: Image & image_source)
    {
        ROS_INFO("VpsDetector:on_recv_frame begin!");

        cv_bridge::CvImagePtr cv_image = cv_bridge::toCvCopy(image_source, "bgr8");
        cv::Mat frame_input;
        cv::Mat src_frame; 
        cv::Mat raw_data = cv_image->image;
        
        geometry_msgs::Point pos;
        geometry_msgs::Quaternion ort;
        pos = g_pose_stamp.pose.position;
        ort = g_pose_stamp.pose.orientation;

        m_frame_roi =  m_frame(cv::Rect(m_x,m_y,m_w,m_h));

        //{"background","free_park", "forbid_park", "incar_park"};
        
        std::vector<Box> non_free_park_boxes;
        std::vector<BBoxInfo> boxes = m_yolo_helper.do_inference(m_frame_roi);
        int box_idx = 0;
        for (BBoxInfo b : boxes)
        {          
            cout<<"box_idx:"<< box_idx++ << endl;
            cout<<"boundingbox:"<<b.box.x1<<","<<b.box.y1<<","<<b.box.x2<<","<<b.box.y2<<endl;
            cout<<"label:"<< b.label<< endl;
            cout<<"classId:"<< b.classId <<endl;
            cout<<"prob:"<< b.prob <<endl;
            cout<<"class_name:"<<m_yolo_helper.m_inferNet->getClassName(b.label)<<endl; 
          
            string class_name =  m_yolo_helper.m_inferNet->getClassName(b.label);
            if(class_name == "forbid_park" || class_name == "incar_park")
            {
                non_free_park_boxes.push_back(b);
            }

            

        }
        
        ROS_INFO("VpsDetector:on_recv_frame end!");
    }

    void init_ros(int argc,char **argv)
    {
        ros::init(argc,argv,"mobilenet_ssd");
        ros::NodeHandle node;
    	ros::NodeHandle private_nh("~");

        private_nh.param<std::string>("proto_file", proto_file, "~/models/mobilenet_deploy.prototxt");
        ROS_INFO("Setting proto_file to %s", proto_file.c_str());

        private_nh.param<std::string>("model_file", model_file, "~/models/mobilenet.caffemodel");
        ROS_INFO("Setting model_file to %s", model_file.c_str());

        private_nh.param<std::string>("model_name", model_name, "mssd_300");
        ROS_INFO("Setting model_name to %s", model_name.c_str());

        private_nh.param<std::string>("image_source_topic", image_source_topic, "/usb_cam/image_raw");
        ROS_INFO("Setting image_source_topic to %s", image_source_topic.c_str());
        
        private_nh.param<std::string>("image_pose_topic",image_pose_topic,"/gnss_pose");
        ROS_INFO("Setting image_pose_topic to %s",image_pose_topic.c_str());

        private_nh.param<std::string>("cam_cmd_topic", cam_cmd_topic, "/cam/cmd"); 
        ROS_INFO("Setting cam_cmd_topic to %s", cam_cmd_topic.c_str());
        
        private_nh.param<std::string>("image_object_topic", image_object_topic, "/vps/park_obj");
        ROS_INFO("Setting image_object_topic to %s", image_object_topic.c_str());
        
        private_nh.param<std::string>("image_raw_topic", image_raw_topic, "/vps/image_raw");
        ROS_INFO("Setting image_raw_topic to %s", image_raw_topic.c_str());
       
        private_nh.param<std::string>("vps_status_topic", vps_status_topic, "/vps/status");
        ROS_INFO("Setting vps_status_topic to %s", vps_status_topic.c_str());

        private_nh.param<std::string>("vps_active_topic", vps_active_topic, "/vps/active"); 
        ROS_INFO("Setting vps_active_topic to %s", vps_active_topic.c_str());
        
        pub_image_obj = node.advertise<autoreg_msgs::park_obj>(image_object_topic, 1);
        pub_image_raw = node.advertise<sensor_msgs::Image>(image_raw_topic,1);
        pub_vps_status = node.advertise<dashboard_msgs::Cmd>(vps_status_topic,1);
        pub_vps_active = node.advertise<dashboard_msgs::Proc>(vps_active_topic,1);
        sub_image_raw = node.subscribe(image_source_topic, 1, image_callback);
        sub_image_pose = node.subscribe(image_pose_topic,1,pose_callback);
        sub_cam_cmd = node.subscribe(cam_cmd_topic,1,cmd_callback);
        gp_atc_tracker = new ATCParkTracker(g_init_clock,g_delta_x,g_delta_y,g_clock_thresh,g_center_thresh,g_iou_thresh,g_iou_level,g_send_epoch);
        //gp_atc_tracker->test_compute_iou();
    }
private:
    //roi参数. 图像裁剪用.  含义:从m*n(width*high)的图片中,以(x,y)为起点,裁剪出w*h的图片:
    string m_roi_region;
    int m_x;int m_y;int m_w;int m_h;

    //当前待处理图像
    cv::Mat m_frame;
    //裁剪后的图像
    cv::Mat m_frame_roi;

    geometry_msgs::PoseStamped g_pose_stamp;
    
private:    
    YoloHelper m_yolo_helper;

    ATCMapper *p_atc_mapper=nullptr;
//
private:
    //roslaunch 参数
    std::string proto_file;
    std::string model_file;
    std::string model_name;
    std::string image_source_topic;
    std::string image_pose_topic;
    std::string image_object_topic;
    std::string vps_status_topic;
    std::string cam_cmd_topic;
    std::string vps_active_topic;
    std::string image_raw_topic;

    //订阅发布topic
    ros::Publisher pub_image_obj;
    ros::Publisher pub_vps_status;
    ros::Publisher pub_vps_active;
    ros::Publisher pub_image_raw;
    ros::Subscriber sub_image_raw;
    ros::Subscriber sub_image_pose;
    ros::Subscriber sub_cam_cmd;
    
    // yulan mapper init params
    int           g_yulan_w = 810;
    int           g_yulan_h = 1080;
    float         g_delta_x = 0.0219;// yulan: 0.0219 simu:0.0285  
    float         g_delta_y = 0.0201; //yulan: 0.0201 imu 0.0287 

private:
    //expand_bndbox
    void expand_bndbox(cv::Rect src_rect,cv::Rect &dst_rect,float ratio,int img_w,int img_h)
    {
        
        if(ratio<0)
            return;
        int baseline = src_rect.height<src_rect.width?src_rect.height:src_rect.width;
        int exp_value =std::ceil(baseline*ratio);
        dst_rect.x = src_rect.x-exp_value;
        dst_rect.x = dst_rect.x>0?dst_rect.x:0;
        dst_rect.y = src_rect.y - exp_value;
        dst_rect.y = dst_rect.y>0?dst_rect.y:0;
        dst_rect.width = src_rect.width + 2* exp_value;
        dst_rect.width = (dst_rect.x + dst_rect.width)<img_w?dst_rect.width:(img_w-1-dst_rect.x);
        dst_rect.height = src_rect.height + 2*exp_value;
        dst_rect.height = (dst_rect.y+ dst_rect.height)<img_h?dst_rect.height:(img_h-1-dst_rect.y);    
    }

    // insert non-repeat crosss over in cross_list
    void insert_crossover(std::vector<cv::Point> &crossover_list,cv::Point src_point)
    {
        //std::cout<<"insert crossover--------->";
        int merge_thresold =8;
        bool is_existed = false;
        for(unsigned int i= 0;i<crossover_list.size();i++)
        {   
            cv::Point crossover = crossover_list[i];
            int distance = (src_point.x-crossover.x)*(src_point.x-crossover.x)+(src_point.y-crossover.y)*(src_point.y - crossover.y);
           // std::cout<<"distance is:"<<distance;
            if(distance<merge_thresold) //merge crosspoint
            {
                is_existed = true;
                break;
            }

        }
        //std::cout<<"leave insert/n";
        if(!is_existed)
            crossover_list.push_back(src_point);
    }


    // park tracker vis 
    void tracker_vis(cv::Mat &img,std::vector<ATCVisPark> vis_trks)
    {
        //const char* cls_names[]={"background:","free","forbidden","incar"};
        for(unsigned int i=0;i<vis_trks.size();i++)
        {
            ATCVisPark *vis_park = &vis_trks[i];
    	cv::line(img,cv::Point(vis_park->grid_data[0],vis_park->grid_data[1]),cv::Point(vis_park->grid_data[2],vis_park->grid_data[3]),cv::Scalar(0,255,0),2,CV_AA);
    	cv::line(img,cv::Point(vis_park->grid_data[2],vis_park->grid_data[3]),cv::Point(vis_park->grid_data[4],vis_park->grid_data[5]),cv::Scalar(0,255,0),2,CV_AA);
    	cv::line(img,cv::Point(vis_park->grid_data[4],vis_park->grid_data[5]),cv::Point(vis_park->grid_data[6],vis_park->grid_data[7]),cv::Scalar(0,255,0),2,CV_AA);
    	cv::line(img,cv::Point(vis_park->grid_data[6],vis_park->grid_data[7]),cv::Point(vis_park->grid_data[0],vis_park->grid_data[1]),cv::Scalar(0,255,0),2,CV_AA);
    	
    	//std::cout<<"--------------------draw bndbox ---------------------"<<vis_trks.size()<<std::endl;
    	std::ostringstream score_str;
    	std::ostringstream id_str;
    	score_str << vis_park->conf_score;
    	id_str<<vis_park->id;
    	//std::cout<<"cls_id:"<<vis_park[i].cls_id<<":score:"<<score_str.str()<<":id:"<<id_str.str()<<std::endl;
            //std::string label = std::string(cls_names[vis_park->cls_id])+"::"+id_str.str()+"_"+score_str.str();
    	std::string label = id_str.str();
    	int base_line=1;        
            cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &base_line);
            cv::rectangle(img, cv::Rect(cv::Point(vis_park->grid_data[0],vis_park->grid_data[1]- label_size.height),cv::Size(label_size.width, label_size.height + base_line)),cv::Scalar(255,255,0),CV_FILLED);
    	cv::putText(img,label,cv::Point(vis_park->grid_data[0],vis_park->grid_data[1]),cv::FONT_HERSHEY_SIMPLEX,0.8,cv::Scalar(0,0,0));
        }
        return ;
    }

    // capture nonvertical-horizonl line's crossover within rect
    void get_two_crossover_from_line_to_rect(float a,float b,float c,cv::Rect src_rect,std::vector<cv::Point> &crossover_list)
    {   
        float x,y;
        if(b<1e-6)
            return ; 
        //a*x + b*y + c = 0;
        //y =0, crossover in top horizonal line
        y = 0;
        x = -(b*y+c)/a;
        //std::cout<<"src_rect:"<<src_rect.width<<":"<<src_rect.height;
        //std::cout<<"top:"<<x<<":"<<y<<",";
        if((x>=0)&&(x<=(src_rect.width-1)))     
            insert_crossover(crossover_list,cv::Point(floor(x),y));
        
        // crossover in bottom horizonal line
        y = src_rect.height - 1;
        x = -(b*y+c)/a;
        //std::cout<<"bot:"<<x<<":"<<y<<",";
        if((x>=0)&&(x<=(src_rect.width-1)))     
            insert_crossover(crossover_list,cv::Point(floor(x),y));
        
        // crossover in left vectical line
        x =0;
        y = -(a*x+c)/b;
        //std::cout<<"left:"<<x<<":"<<y<<",";
        if((y>=0)&&(y<=(src_rect.height-1)))
             insert_crossover(crossover_list,cv::Point(x,floor(y)));
        
        // crossover in right vertical line
        x = src_rect.width-1;
        y = -(a*x+c)/b;
        //std::cout<<"right:"<<x<<":"<<y<<",";
        if((y>=0)&&(y<=(src_rect.height-1)))
            insert_crossover(crossover_list,cv::Point(x,floor(y)));
        //std::cout<<"finishing "<<std::endl;    
    } 



    //图像裁剪
    void img_decode(cv::Mat raw_data, cv::Mat &dst_img,float &dst_delta_x,float &dst_delta_y)
    {
        int raw_w = raw_data.cols;
        int raw_h = raw_data.rows;
        //std::cout<<"raw_w:"<<raw_w<<"raw_h:"
        // get raw_data from yulan device
        if((g_yulan_w<=raw_w)&&(g_yulan_h<=raw_h))
        {
            dst_img=raw_data(cv::Rect(0,0,g_yulan_w,g_yulan_h));
            dst_delta_x = g_delta_x;
    	    dst_delta_y = g_delta_y;
        }else// get raw_data from simulator
        {
        	dst_img = raw_data;
    	    dst_delta_x = 24.0/raw_w;
    	    dst_delta_y = 30.0/raw_h;
        }
    
    }

    //get iou ratio between two bndingbox
    float get_box_overlap_ratio(Box bbox1,Box bbox2)
    {
        float x_min,y_min,x_max,y_max;
        float area_bbox1,area_bbox2,area_intersect;
        float iou=0.0;
        if (bbox2.x0 > bbox1.x1 || bbox2.x1 < bbox1.x0 || bbox2.y0 > bbox1.y1 || bbox2.y1 < bbox1.y0) 
            return iou;
        x_min = std::max(bbox1.x0, bbox2.x0);
        y_min = std::max(bbox1.y0, bbox2.y0);
        x_max = std::min(bbox1.x1, bbox2.x1);   
        y_max =std::min(bbox1.y1, bbox2.y1);
        
        //
        area_bbox1 =(bbox1.x1 - bbox1.x0)*(bbox1.y1 - bbox1.y0);
        area_bbox2 =(bbox2.x1 - bbox2.x0)*(bbox2.y1 - bbox2.y0);
        area_intersect = (x_max - x_min)*(y_max -y_min);
        iou = area_intersect/(area_bbox1 + area_bbox2 - area_intersect);
        return iou;
    }


    //通过iou判断车位是否有效
    bool overlap_park_supperss(std::vector<Box> box_list,Box src_box,float iou_thresold/* =0.5*/)
    {
        bool suppressed = false;
        if(iou_thresold<0|| iou_thresold >1)
            iou_thresold = 0.5;//default iou
        float iou;
        for(unsigned int idx =0;idx<box_list.size();++idx)
        {
            Box box = box_list[idx];
            if(1!=box.class_idx) // forbidden_park or incar_park
            {
                iou =  get_box_overlap_ratio(box,src_box);
                if(iou>iou_thresold)
                {
                    suppressed = true;
                    break;
                }
            }
        }
        
        return suppressed; 
    }

    // get line' length  获取两点之间的距离
    float get_len(cv::Point begin,cv::Point end)
    {
        float len =0;
        len =sqrt((begin.x-end.x)*(begin.x-end.x)+(begin.y-end.y)*(begin.y-end.y));
        return len;
    }


    // filter abnormal park anchors 
    bool park_anchor_filter(ParkAnchor src_park,cv::Point bbox_center,float thresold,float max_lw_ratio,float min_lw_ratio/*=1.0*/)
    {
        float l_len,w_len,center_len,lw_ratio;
        cv::Point park_center;
        //if(4!=src_park.num)
        //    return false;
        l_len= get_len(src_park.rtop,src_park.rbot);
        w_len = get_len(src_park.ltop,src_park.rtop);
        lw_ratio  = l_len/w_len;
        //
        if(l_len<w_len)
            lw_ratio = 1/lw_ratio;
        if((lw_ratio<min_lw_ratio)||(lw_ratio>max_lw_ratio))
            return false;
        park_center.x = (src_park.ltop.x + src_park.rbot.x)/2;
        park_center.y = (src_park.ltop.y = src_park.rbot.y)/2;
        
        center_len = get_len(bbox_center, park_center);
        if(center_len>thresold)
        {
            //std::cout<<"Warnning: center shift is out of thresold\n";
            return false;
        }
        return true;  
    }

    
    // image angle expanding  旋转图像
    void imrotate(cv::Mat& img,cv::Mat& newIm, double angle)
    {
        //better performance 
        std::size_t rot_size=0;
        if(img.rows>img.cols)
            rot_size = img.rows;
        else
            rot_size = img.cols;
        cv::Point2f pt(img.cols/2.,img.rows/2.);
        cv::Mat r = cv::getRotationMatrix2D(pt,angle,1.0);
        warpAffine(img,newIm,r,cv::Size(rot_size,rot_size),cv::INTER_LINEAR,cv::BORDER_CONSTANT,cv::Scalar(127,127,127));
     }

    // get two line's crosspoint
    cv::Point2f getCrossPoint(cv::Vec4i LineA, cv::Vec4i LineB)
    {
        double ka, kb;
        ka = (double)(LineA[3] - LineA[1]) / (double)(LineA[2] - LineA[0]); 
        kb = (double)(LineB[3] - LineB[1]) / (double)(LineB[2] - LineB[0]); 

        cv::Point2f crossPoint;
        crossPoint.x = (ka*LineA[0] - LineA[1] - kb*LineB[0] + LineB[1]) / (ka - kb);
        crossPoint.y = (ka*kb*(LineA[0] - LineB[0]) + ka*LineB[1] - kb*LineA[1]) / (ka - kb);
        return crossPoint;
    }
};

#endif
