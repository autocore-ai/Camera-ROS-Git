/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * License); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * AS IS BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*
 * Copyright (c) 2018, Open AI Lab
 * Author: chunyinglv@openailab.com
 */

#include <unistd.h>
#include <iostream>
#include <iomanip>
#include <pthread.h>
#include <string>
#include <vector>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "tengine_c_api.h"
#include "auto_park.h"
#include "auto_yulan_sdk.h"
#include "auto_tracker_sdk.h"
#include "auto_transform.h"
#include <sys/time.h>

#include <ros/ros.h>
#include "autoreg_msgs/park_obj.h"
#include "dashboard_msgs/cam_cmd.h"
#include "dashboard_msgs/Cmd.h"
#include "dashboard_msgs/Proc.h"
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <std_msgs/Int32MultiArray.h>
#include <std_msgs/Int32.h>
#include <cv_bridge/cv_bridge.h>
//#include <geometry_msgs/PoseStamped.h>
//#include <geometry_msgs/Quaternion.h>
//#include <geometry_msgs/Point.h>
//#include <tf/transform_broadcaster.h>
//#include <tf/transform_datatypes.h>
//#include <tf/transform.h>
//#include <tf/vector3.h>
//#include <tf/quaternion.h>

#define DEF_PROTO "models/MobileNetSSD_deploy.prototxt"
#define DEF_MODEL "models/MobileNetSSD_deploy.caffemodel"
#define DEF_IMAGE "tests/images/ssd_dog.jpg"

using namespace cv;
using namespace std;

//posestamp.pose.position.x;y,z
//posestamp.orientation.x.y.z.w
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
static ros::Publisher pub_image_obj;
static ros::Publisher pub_vps_status;
static ros::Publisher pub_vps_active;
static ros::Publisher pub_image_raw;
static ros::Subscriber sub_image_raw;
static ros::Subscriber sub_image_pose;
static ros::Subscriber sub_cam_cmd;

//ssd算法参数
float        *g_input_data = NULL;
tensor_t      g_input_tensor;
graph_t       g_graph;

int           g_img_h  = 300;
int           g_img_w  = 300;
int    seq_counter = 0;

//cmd init params
int g_node_label = 0;// sleep(0),wakeup(1)
bool g_node_active = true;
bool g_win_show = false;
std_msgs::Header g_img_header;
//bool g_non_hang_label = 0;
pthread_mutex_t g_cmd_mutex;
//pthread_mutex_t g_active_mutex;

// yulan mapper init params
int           g_yulan_w = 810;
int           g_yulan_h = 1080;
float         g_delta_x = 0.0219;// yulan: 0.0219 simu:0.0285  
float         g_delta_y = 0.0201; //yulan: 0.0201 imu 0.0287 

// Autocore Park tracker  init params
ATCParkTracker *gp_atc_tracker;
pthread_mutex_t g_pose_mutex;
geometry_msgs::PoseStamped g_pose_stamp;
unsigned int   g_init_clock = 10;
unsigned int   g_clock_thresh =3 ;
float          g_center_thresh = 20.0;
float          g_iou_thresh = 0.15;
unsigned int   g_iou_level = 1;
unsigned int   g_send_epoch = 10;
//bool g_static_test =true;
struct Box
{
    float x0;
    float y0;
    float x1;
    float y1;
    int class_idx;
    float score;
};
struct ParkAnchor
{
    //std::vector<cv::Point> list;// 0:ltop,1:rtop;2,rbot,3:lbot;
    //int num =0;//
    cv::Point ltop;
    cv::Point rtop;
    cv::Point rbot;
    cv::Point lbot;
};
float get_box_overlap_ratio(Box bbox1,Box bbox2);
void tracker_vis(cv::Mat &img,std::vector<ATCVisPark> vis_trks);
bool overlap_park_supress(std::vector<Box> box_list,Box src_box,float iou_thresold =0.5); 
bool park_anchor_filter( ParkAnchor src_park,cv::Point center,float thresold,float max_lw_ratio,float min_lw_ratio=1.0);
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
    
    //
    return suppressed;
  
}

// image angle expanding
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

// get line' length  
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
//
bool get_park_anchor(std::vector<cv::Point> anchor_list, ParkAnchor &dst_park,float offset)
{
    cv::Point v10,v32,v_merge;
    
    if(offset<0||offset>1.0)
        return false;
    if(4!=anchor_list.size())
        return false;
    // line( anchor_list[1],anchor_list[2]) //the maximun distance line
    v10.x = anchor_list[1].x-anchor_list[0].x;
    v10.y = anchor_list[1].y - anchor_list[0].y;
    v32.x = anchor_list[3].x -anchor_list[2].x;
    v32.y = anchor_list[3].y-anchor_list[2].y;
    
    if ((v10.x*v10.x)<(v32.x*v32.x))
    {   
        v_merge.x = ceil(v10.x+offset*(v32.x-v10.x));
        v_merge.y = ceil(v10.y+offset*(v32.y-v10.y));
    }
    else
    {
        v_merge.x = ceil(v32.x+offset*(v10.x-v32.x));
        v_merge.y = ceil(v32.y+offset*(v10.y-v32.y));
    }
    dst_park.ltop =cv::Point(anchor_list[0].x + v_merge.x,anchor_list[0].y+v_merge.y);
    dst_park.rtop =anchor_list[0];
    dst_park.rbot = anchor_list[2];
    dst_park.lbot =cv::Point(anchor_list[2].x + v_merge.x,anchor_list[2].y+v_merge.y);
    return true;
    
}

// remove image border edges
bool depress_fringe_grad(cv::Mat src_img,cv::Mat &dst_img,int shrink)
{
    cv::Mat mid_img,mask;
    int dst_w,dst_h;

    //
    mask = cv::Mat::zeros(src_img.size(),src_img.type());
    dst_w = src_img.cols -2*shrink;
    dst_h = src_img.rows -2*shrink;
    if(shrink<0)
    {
        //src_img.copyTo(dst_img,mask);
        return false;
    }
    if(dst_w<1||dst_h<1)// bad shrink
    {
        //std::cout<<"Warnning: bad image shrink,please decrease shrink offset\n";
	//dst_img = mask;
        //src_img.copyTo(dst_img,mask);
        //std::cout<<"cout--------\n";
        return false;
    }
        
    mask(cv::Rect(shrink-1,shrink-1,dst_w,dst_h)).setTo(255);
    //std::cout<<"shrink:"<<shrink-1<<" dst_w:"<<dst_w<<" dst_h:"<<dst_h<<endl; 
    //imshow("mask",mask);
   
    src_img.copyTo(dst_img,mask);
    //std::cout<<"out offf drepressed \n";
    return true;
}

// input park patch image,
bool park_edge_detect(cv::Mat src_img,cv::Mat &dst_img)
{   
    cv::Mat mid_img,edge_img,depress_img,mask;
    cv::cvtColor(src_img,mid_img,CV_BGR2GRAY);
    
    //image enhance
   // cv::equalizeHist(mid_img,mid_img);
   // cv::imshow("gray",mid_img); 
    
    // canny operator
    //std::cout<<"try median filter!"<<std::endl;
    cv::Canny(mid_img, edge_img, 50, 200, 3);
    bool ret =depress_fringe_grad(edge_img,depress_img,5);
    if (!ret)
    	return ret;	    
    //get binary mask
    cv::equalizeHist(mid_img,mid_img);
    cv::threshold(mid_img,mask,180,255,CV_THRESH_BINARY);
    cv::Mat element = cv::getStructuringElement(MORPH_RECT, cv::Size(3, 3)); 
    // cv::erode(mask, mask, element);
    cv::dilate(mask,mask,element);
    cv::threshold(mask,mask,180,255,CV_THRESH_BINARY);

    //cv::imshow("bin_img",mask);
    //std::cout<<"width:"<<depress_img.cols<<" height:"<<depress_img.rows<<std::endl;
    //std::cout<<"height:"<<mask.cols<<" height:"<<mask.rows<<std::endl;
    depress_img.copyTo(dst_img,mask);
    //std::cout<<"out park edge detect\n";
    //cv::imshow("raw_edge",edge_img);
    //cv::imshow("dst_edge",dst_img);
    //cv::waitKey(100);
    return ret;

}
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

// feed data
void get_input_data_ssd(Mat& image_org, float* input_data, int img_h, int img_w)
{
    cv::Mat image_input = image_org.clone();
    cv::resize(image_input, image_input, cv::Size(img_h, img_w));
    image_input.convertTo(image_input, CV_32FC3);
    float *img_data = (float *)image_input.data;
    int hw = img_h * img_w;

    float mean[3]={127.5,127.5,127.5};
    for (int h = 0; h < img_h; h++)
    {
        for (int w = 0; w < img_w; w++)
        {
            for (int c = 0; c < 3; c++)
            {
                input_data[c * hw + h * img_w + w] = 0.007843* (*img_data - mean[c]);
                img_data++;
            }
        }
    }
}
void post_process_ssd(cv::Mat& image_input,ATCParkTracker *p_atc_tracker,ATCMapper *p_atc_mapper,float threshold,float* outdata,int num, cv::Mat& img,autoreg_msgs::park_obj& msg)
{
    //const char* class_names[] = {"background","free_park", "forbid_park", "incar_park"};
    int src_w = int(p_atc_mapper->get_width());
    int src_h = int(p_atc_mapper->get_height());

    img = image_input.clone();
    int raw_h = img.size().height;
    int raw_w = img.size().width;
    std::vector<Box> boxes;
    std::vector<Box> non_free_park_boxes;
    //printf("detect bndbox num: %d \n",num);
    
    /*---insert forbidden_park and incar_park in non-free park list---*/  
    for (int i=0;i<num;i++)
    {
        if((outdata[0]<1)||(outdata[0]>3))
	{
	    outdata +=6;
	    continue;
	} 
	if(outdata[1]>=threshold)
        {
            Box box;
            box.class_idx=outdata[0];		
            box.score=outdata[1];
            box.x0=outdata[2]*raw_w;	    	
            box.y0=outdata[3]*raw_h;
            box.x1=outdata[4]*raw_w;
            box.y1=outdata[5]*raw_h;            
            if((2==outdata[0])||(3==outdata[0]))
            {			                    
		non_free_park_boxes.push_back(box);
            }
        }
        outdata+=6;
    }

    /*---free parks will be supressed,when overlap with non-free parks more than 60 percent---*/
    /*--back to outdata head--*/
    outdata = outdata -6*num; 
    for (int i=0;i<num;i++)
    {	
	if((outdata[0]<1) or(outdata[0]>3))
	{
	    outdata +=6;
	    continue;
	} 
        if(outdata[1]>=threshold)
        {
            Box box;
            box.class_idx=outdata[0];
            box.score=outdata[1];
            box.x0=outdata[2]*raw_w;
            box.y0=outdata[3]*raw_h;
            box.x1=outdata[4]*raw_w;
            box.y1=outdata[5]*raw_h;
            
	    /*-clip bndingbox in the prefilled image region-*/
	    if(box.x0>=src_w)
		box.x0 = src_w -1;
	    if(box.x1>=src_w)
		box.x1 = src_w -1;
	    if(box.y0>=src_h)
		box.y0 = src_h -1;
	    if(box.y1>=src_h)
                box.y1 = src_h -1;
	    
            if(1==outdata[0])
            {   
                bool ret = overlap_park_supperss(non_free_park_boxes,box,0.6);
		/*-keep free parks which non-supressed-*/
                if(!ret) 
                {    
		    boxes.push_back(box);
                    //printf("%s\t:%.0f%%\n", class_names[box.class_idx], box.score * 100);
                    //printf("BOX:( %g , %g ),( %g , %g )\n",box.x0,box.y0,box.x1,box.y1);
                }
            }
            else 
            {
                boxes.push_back(box);
                //printf("%s\t:%.0f%%\n", class_names[box.class_idx], box.score * 100);
                //printf("BOX:( %g , %g ),( %g , %g )\n",box.x0,box.y0,box.x1,box.y1);
            }         
        }
        outdata +=6;
    }
    
    std::vector<ParkAnchor> dst_anchor_list;
    //int num_park =0;	
    /*---park location finetune---*/   
    for(int i =0;i<(int)boxes.size();i++)
    {
        cv::Mat roi_img, mid_img;
        cv::Rect expand_rect;
        Box box=boxes[i];
        bool is_valid =true;
	int box_cls_idx = box.class_idx;	
	float box_score = box.score;
	
	/*--enlarge bndingbox with o.1*width pixels--*/
        expand_bndbox(cv::Rect(box.x0,box.y0,(box.x1 - box.x0),(box.y1 - box.y0)),expand_rect,0.1,raw_w,raw_h);
	roi_img=image_input(expand_rect);
        //cv::rectangle(img, expand_rect,cv::Scalar(255, 0, 0),line_width);
	
       
        /*--two stage park line detection--*/  
        bool ret1 =park_edge_detect(roi_img,mid_img);
	if(!ret1)
	{
	    //std::cout<<"bad canidate skip."<<std::endl;
 	    continue;
	}
        std::vector<cv::Vec4i> lines;
        cv::HoughLinesP(mid_img, lines, 1, CV_PI/180, 20, 20, 10 );
        if(1>lines.size())
        {
            //std::cout<<"warnning can not detect lines\n";
            continue;
        }
	        
	/*- first stage: find the maximun length line!-*/
	int line_idx = 0;
        int max_line =0;
        for(unsigned int j =0;j<lines.size();j++)
        {
            cv::Vec4i l = lines[j];
            int line_len =(l[2]-l[0])*(l[2]-l[0])+(l[3]-l[1])*(l[3]-l[1]);            
            if(line_len>max_line)
            {
                max_line = line_len;
                line_idx = j;
            }            
        }

        cv::Vec4i result=lines[line_idx]; 
        cv::Point begin(result[0],result[1]);
        cv::Point center(expand_rect.width/2,expand_rect.height/2); 
        cv::Point end(result[2],result[3]);// ax+by+c =0;
        ParkAnchor draw_anchor;
        //cv::line( roi_img, begin, end,cv::Scalar(0,255,255), 5, CV_AA);

        float a,b,c;
	/*-second stage: find the short line-*/
	/* the long line direction cases:vectical line, horizonal line and other direction line*/	     
	if(begin.x == end.x)//vectical line
        {
            if(begin.x<center.x)// |*
            {
                draw_anchor.ltop=cv::Point(begin.x,0);
                draw_anchor.rtop=cv::Point(expand_rect.width-1,0);
                draw_anchor.rbot=cv::Point(expand_rect.width-1,expand_rect.height-1);
                draw_anchor.lbot=cv::Point(begin.x,expand_rect.height-1);
            }
            else // *|
            {
                draw_anchor.ltop=cv::Point(0,0);
                draw_anchor.rtop=cv::Point(begin.x,0);
                draw_anchor.rbot=cv::Point(begin.x,expand_rect.height-1);
                draw_anchor.lbot=cv::Point(0,expand_rect.height-1);
            }         
        }
        else if(begin.y == end.y)// horizonal line
        {
            if(begin.y<center.y)//V
            {
                draw_anchor.ltop=cv::Point(0,begin.y);
                draw_anchor.rtop=cv::Point(expand_rect.width-1,begin.y);
                draw_anchor.rbot=cv::Point(expand_rect.width-1,expand_rect.height -1);
                draw_anchor.lbot=cv::Point(0,expand_rect.height-1);
            }else//^
            {                   
                draw_anchor.ltop = cv::Point(0,0);
                draw_anchor.rtop = cv::Point(expand_rect.width-1,0);
                draw_anchor.rbot = cv::Point(expand_rect.width -1,begin.y);
                draw_anchor.lbot = cv::Point(0,begin.y);
            }
        }
        else // other direction line
        {            
            b =1.0;
            a = -(end.y-begin.y)*1.0/(end.x-begin.x);
            c = -a*begin.x - begin.y;
            std::vector<cv::Point> line_crossover;
            //std::cout<<"long line:"<<begin.x<<":"<<begin.y<<","<<end.x<<":"<<end.y<<"---->"<<a<<","<<b<<","<<c<<std::endl;
            
            /*catch two crossover between long line with expand bndingbox rect */
            get_two_crossover_from_line_to_rect(a,b, c,expand_rect,line_crossover);
            if(2!=line_crossover.size())
            {
                //std::cout<<"Warnning:can not get two crossover vs("<<line_crossover.size()<<"between maximun detected line with ROI rect \n";
                is_valid = false;
                continue;
            }
            
            /*catch perpendicular line based on two crossovers*/
            std::vector<cv::Point> rect_crossover;
            for(unsigned int idx=0;idx<line_crossover.size();idx++)
            {
                cv::Point crossover_point = line_crossover[idx];
                float p_a,p_b,p_c;//p_a*x+p_b*y+p_c=0
                p_a =-1/a;
                p_b = 1.0;
                p_c = -(crossover_point.y)-p_a*(crossover_point.x);
                                
                std::vector<cv::Point> next_crossover;
                next_crossover.push_back(crossover_point);
                get_two_crossover_from_line_to_rect(p_a,p_b,p_c,expand_rect,next_crossover);
                if(2!=next_crossover.size())
                {
                    //std::cout<<"Warnning:can not get two crossover vs("<<next_crossover.size()<<" in perpendicular line \n";
                    is_valid =false;
                    continue;   
                }
                for(unsigned int pline_idx=0;pline_idx<next_crossover.size();pline_idx++)
                {
                    rect_crossover.push_back(next_crossover[pline_idx]);
                }
            }

            /*-- translate 4 crossovers into ParkAnchor*/
            if(4==rect_crossover.size())
            {
                get_park_anchor( rect_crossover,draw_anchor,1);
                
            }
            else
            {
                is_valid =false;
            }
           
        }
        
        /*--drop unvalid park detection--*/ 
        if(!is_valid)
        {
            //std::cout<<"waring unvalid draw_anchor\n";
            continue;
        }
	/*--drop center-shift,abnormal shape parks--*/
        if(!park_anchor_filter(draw_anchor, center,20,4.5,1.2))
        {
            /*std::cout<<"bad detected park,skip......\n";            
            draw_anchor.ltop.x += expand_rect.x;
            draw_anchor.ltop.y += expand_rect.y;
            draw_anchor.rtop.x += expand_rect.x;
            draw_anchor.rtop.y += expand_rect.y;
            draw_anchor.rbot.x += expand_rect.x;
            draw_anchor.rbot.y += expand_rect.y;
            draw_anchor.lbot.x += expand_rect.x;
            draw_anchor.lbot.y += expand_rect.y;                   
            begin.x +=expand_rect.x;
            begin.y +=expand_rect.y;
            end.x +=expand_rect.x;
            end.y +=expand_rect.y;

            cv::line( img, begin, end,cv::Scalar(0,255,255),5, CV_AA);
            cv::line( img, draw_anchor.ltop, draw_anchor.rtop,cv::Scalar(0,0,255), 2, CV_AA);
            cv::line( img, draw_anchor.rtop, draw_anchor.rbot,cv::Scalar(0,0,255),2, CV_AA);
            cv::line( img, draw_anchor.rbot, draw_anchor.lbot,cv::Scalar(0,0,255), 2, CV_AA);
            cv::line( img, draw_anchor.lbot, draw_anchor.ltop,cv::Scalar(0,0,255), 2, CV_AA);
            */
            continue;
        }
	
	//--map park anchor from expand bnding box to raw image--
        draw_anchor.ltop.x += expand_rect.x;
        draw_anchor.ltop.y += expand_rect.y;
        draw_anchor.rtop.x += expand_rect.x;
        draw_anchor.rtop.y += expand_rect.y;
        draw_anchor.rbot.x += expand_rect.x;
        draw_anchor.rbot.y += expand_rect.y;
        draw_anchor.lbot.x += expand_rect.x;
        draw_anchor.lbot.y += expand_rect.y;
	
	//--add detected park msg
	autoreg_msgs::park_anchor park_obj;
	
	//cv::Point2f left_top,right_top,right_bot,left_bot;
	ATCPark* p_new_park =new ATCPark;
	p_new_park->points_in_img[0] = draw_anchor.ltop.x;
	p_new_park->points_in_img[1] = draw_anchor.ltop.y;
	p_new_park->points_in_img[2] = draw_anchor.rtop.x;
	p_new_park->points_in_img[3] = draw_anchor.rtop.y;
	p_new_park->points_in_img[4] = draw_anchor.rbot.x;
	p_new_park->points_in_img[5] = draw_anchor.rbot.y;
	p_new_park->points_in_img[6] = draw_anchor.lbot.x;
	p_new_park->points_in_img[7] = draw_anchor.lbot.y;
	p_new_park->conf_score =  box_score;
        p_new_park->id =0;
	p_new_park->cls_id =box_cls_idx;
	
	//mapping park from image axes to vector map  axes
	p_atc_mapper->convert_to_vecmap(p_new_park);

	//p_atc_mapper->convert
	bool res =p_atc_tracker->add_tracker(p_new_park);
	if(!res)
	{
	    ROS_WARN("add new tracker failed.");
	}
    }

    
    std::vector<ATCVisPark> vis_trks;
    std::vector<ATCPubPark> pub_trks;    
    unsigned int nums_pub;
    
    p_atc_tracker->get_vis_trackers(vis_trks);
    tracker_vis(img,vis_trks);
    nums_pub =p_atc_tracker->get_pub_trackers(pub_trks);
    
    // publish msg
    if(nums_pub>0)
    {
        //ROS_INFO("send parks:%d",nums_pub);
	
	for(unsigned int i=0;i<pub_trks.size();i++)
	{
	    autoreg_msgs::park_anchor park_obj;
	    ATCPubPark pub_park = pub_trks[i];
	    park_obj.x0 = pub_park.grid_data[0];
	    park_obj.y0 = pub_park.grid_data[1];
	    park_obj.x1 = pub_park.grid_data[2];
	    park_obj.y1 = pub_park.grid_data[3];
	    park_obj.x2 = pub_park.grid_data[4];
	    park_obj.y2 = pub_park.grid_data[5];
	    park_obj.x3 = pub_park.grid_data[6];
	    park_obj.y3 = pub_park.grid_data[7];
	    park_obj.cls_id = pub_park.cls_id;
	    park_obj.id = pub_park.id;
	    park_obj.score = pub_park.conf_score;
            msg.obj.push_back(park_obj);
	}
	pub_image_obj.publish(msg);	
    }
}

void inverse_tf(geometry_msgs::PoseStamped pose_stamp)
{
    tf::Transform transform;
    //geometry_msgs::Point    
    //tf::Quaternion tfqt();
    geometry_msgs::Point pt= pose_stamp.pose.position;
    geometry_msgs::Quaternion ort = pose_stamp.pose.orientation;
    tf::Vector3  vec_w(pt.x,pt.y,pt.z);
    tf::Quaternion tfqt(ort.x,ort.y,ort.z,ort.w);
    //transform.setRotation(tfqt);
    transform.setOrigin(vec_w);
    transform.setRotation(tfqt);
    //tf::Transform inv_transform =transform.inverse();
    tf::Vector3 center(0,1,0);
    center = transform*center;
    std::cout<<"inv:"<<center.x()<<","<<center.y()<<","<<center.z()<<std::endl;
    //get RPY to X-Y-Z axes
}

//camera module cmd call back
static void cmd_callback(const dashboard_msgs::cam_cmd &cmd)
{   
    
    unsigned int func_id = cmd.func_id;
    int cam_id = cmd.cam_id;
    int cmd_id = cmd.cmd_id;
    //std::cout<<"cam_id: "<<cam_id<<" cmd_id: "<<cmd_id<<std::endl;
    //int cmd_accept_label = 1;
    if(cam_id ==2)
    {
	int ret = cmd_id + 1;
	int cmd_accept_label =1;
	if(-1 == cmd_id)
        {   
	    ROS_INFO("EX: shut down vps_detect node!");
	    pthread_mutex_lock(&g_cmd_mutex);
	    g_node_active =false;
	    g_node_label = 0;	    
	    // cmd response publish 
	    std_msgs::Int32MultiArray val;
	    val.data.push_back(cmd_id);
	    //val.data.push_back(-1);
	    val.data.push_back(cmd_accept_label);
            pthread_mutex_unlock(&g_cmd_mutex);
	}
	else
	{
	    //int node_label;
	    if((cmd_id>-1)&&(cmd_id<2))
	    {
	        pthread_mutex_lock(&g_cmd_mutex);
	        g_node_label = cmd_id;
		pthread_mutex_unlock(&g_cmd_mutex);
		//node_label = cmd_id;
                if(cmd_id ==0)
		    ROS_INFO("EX: ensleep vps_detect node!");
		else
		    ROS_INFO("EX: wake up vps_detect node!");
	    }
	    else
	    {
                cmd_accept_label = 0;
		ret = ret -1;
	        ROS_INFO("EX: unknown cmd!");
	    }
	    // publish 
	    dashboard_msgs::Cmd val;
	    val.header = g_img_header;
	    val.func_id =func_id;
	    val.value = unsigned(ret);
	    val.retry = 0;
	    pub_vps_status.publish(val);
	}
    }  
}

// car pose call back
static void pose_callback(const geometry_msgs::PoseStamped &pose_stamp)
{
    pthread_mutex_lock(&g_pose_mutex);
    g_pose_stamp.pose.position = pose_stamp.pose.position;
    g_pose_stamp.pose.orientation = pose_stamp.pose.orientation;
    pthread_mutex_unlock(&g_pose_mutex);
}

// ros raw image call back
static void image_callback(const sensor_msgs::Image& image_source)
{
    int wake_up =0;
    //bool node_active =false;    
    pthread_mutex_lock(&g_cmd_mutex);  
    wake_up = g_node_label;
    g_img_header = image_source.header;
    pthread_mutex_unlock(&g_cmd_mutex);
    
   std::string win_name = "vps_show";
    // wake up
    if(wake_up>0)
    {
	if(!g_win_show)// new win
	{
	    cv::namedWindow(win_name,WINDOW_AUTOSIZE);
	    cv::moveWindow(win_name,0,0);
	    g_win_show = true;
	    ROS_INFO("new vps window!");
	}
    }
    else //sleep
    {
	if(g_win_show)//
	{
	    cv::destroyWindow(win_name);
	    g_win_show = false;
	    ROS_INFO("delete vps window!");
            gp_atc_tracker->clear();
	}
	return;
    }    
    cv_bridge::CvImagePtr cv_image = cv_bridge::toCvCopy(image_source, "bgr8");
    cv::Mat frame_input;
    cv::Mat src_frame; 
    cv::Mat raw_data = cv_image->image;
    
    if(raw_data.empty())
	    return ;
    geometry_msgs::Point pos;
    geometry_msgs::Quaternion ort;
    
    // load pose dataset
    pthread_mutex_lock(&g_pose_mutex);
    pos = g_pose_stamp.pose.position;
    ort = g_pose_stamp.pose.orientation;
    pthread_mutex_unlock(&g_pose_mutex);
    
    // decode img from raw data
    img_decode(raw_data,src_frame,g_delta_x,g_delta_y);
    int src_w = src_frame.cols;
    int src_h = src_frame.rows;
    ATCMapper *p_atc_mapper= new ATCMapper();
    p_atc_mapper->update(g_delta_x,g_delta_y,src_w,src_h,pos,ort);
    	    
    // prefill input image to square with cv::Scalar(127,127,127) pixels
    imrotate(src_frame, frame_input,0);     

    //msg header 
    autoreg_msgs::park_obj msg;
    msg.header = image_source.header;
    msg.header.frame_id = "park obj";
    msg.type = "park_obj";
    
    if(g_input_data == NULL)
    {
        // allocate input buffer
        int img_size = g_img_h * g_img_w * 3;
        g_input_data = (float *)malloc(sizeof(float) * img_size);
        if(g_input_data == NULL)
        {
            ROS_WARN("malloc input data failed");;
            exit(1);
        }
    }

    get_input_data_ssd(frame_input, g_input_data, g_img_h,  g_img_w);
    set_tensor_buffer(g_input_tensor, g_input_data, g_img_h * g_img_w * 3 * 4);
    run_graph(g_graph, 1);
    tensor_t out_tensor = get_graph_output_tensor(g_graph, 0,0);
    int out_dim[4];
    get_tensor_shape( out_tensor, out_dim, 4);
    float *outdata = (float *)get_tensor_buffer(out_tensor);

    int num = out_dim[1];
    float show_threshold=0.6;
    cv::Mat frame_show;
    
    /*unsigned int old_trks = gp_atc_tracker->update();
    ROS_INFO("Old trackers:%d",old_trks);
    */
    gp_atc_tracker->update();
    post_process_ssd(frame_input,gp_atc_tracker,p_atc_mapper,show_threshold, outdata, num, frame_show, msg);
    delete p_atc_mapper; 
    
    // detect show
    cv::Mat detect_show = frame_show(cv::Rect(0,0,src_w,src_h));    
    // publish roi raw msg 
    sensor_msgs::ImagePtr roi_msg =cv_bridge::CvImage(std_msgs::Header(), "bgr8", detect_show).toImageMsg();
    roi_msg->header.seq = seq_counter++;
    roi_msg->header.frame_id = "vps image";
    roi_msg->header.stamp = ros::Time::now();
    pub_image_raw.publish(roi_msg);

    cv::imshow(win_name, detect_show);
    cv::waitKey(10);
   
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

// publish heart beat
void *auto_active_pub(void *arg)
{

    bool active_loop =true; 
    while(active_loop)// 
    {
	int wake_up;
        pthread_mutex_lock(&g_cmd_mutex);
	wake_up = g_node_label;
	active_loop = g_node_active;
	pthread_mutex_unlock(&g_cmd_mutex);
        
	dashboard_msgs::Proc active_msg;
        active_msg.proc_name = "vpsdetect";
	active_msg.set = unsigned(wake_up);
        pub_vps_active.publish(active_msg);
	//std::cout<<"heart beat\n";
	sleep(1);
    }
    std::cout<<"VPS QUIT\n";
    sleep(3);
    delete gp_atc_tracker;
    exit(0);
}
int main(int argc, char *argv[])
{
    pthread_mutex_init(&g_pose_mutex, NULL);
    pthread_mutex_init(&g_cmd_mutex,NULL);
    //ros初始化
    init_ros(argc, argv);

    // init tengine
    init_tengine_library();
    if (request_tengine_version("0.1") < 0)
        return 1;
    if (load_model(model_name.c_str(), "caffe", proto_file.c_str(), model_file.c_str()) < 0)
        return 1;
    ROS_INFO("load model done!");
    // create graph
    g_graph = create_runtime_graph("graph", model_name.c_str(), NULL);
    if (!check_graph_valid(g_graph))
    {
        ROS_WARN("create graph0 failed");
        return 1;
    }
    //tensor
    int node_idx = 0;
    int tensor_idx = 0;
    g_input_tensor = get_graph_input_tensor(g_graph, node_idx, tensor_idx);
    if(!check_tensor_valid(g_input_tensor))
    {
        ROS_INFO("Get input node failed : node_idx: %d, tensor_idx: %d",node_idx,tensor_idx);
        return 1;
    }
    // prerun graph
    int dims[] = {1, 3, g_img_h, g_img_w};
    set_tensor_shape(g_input_tensor, dims, 4);
    
    prerun_graph(g_graph);

    //申请内存，注意释放
    int img_size = g_img_h * g_img_w * 3;
    g_input_data = (float *)malloc(sizeof(float) * img_size);
    if(g_input_data == NULL)
    {
        ROS_WARN("malloc input data failed");
        return 1;
    }
    ROS_INFO("graph is ready,waiting for image raw");
    pthread_t pub_active_tid;
    if(pthread_create(&pub_active_tid,NULL,auto_active_pub,NULL)!=0)
    {
	ROS_INFO("error in pthread_create");
	exit(0);
    }

    ros::spin();
    
    //thread end
    if(pthread_join(pub_active_tid,NULL)!=0)
    {
        ROS_INFO("error in pthread!");
    }
    /*while(restd    {
        ros::spinOnce();
	publish(heart);
    }*/

    postrun_graph(g_graph);
    if(g_input_data)
    {
        free(g_input_data);
        g_input_data = NULL;
        printf("[SSD]free input data ok\n");
    }
    destroy_runtime_graph(g_graph);
    remove_model(model_name.c_str());
   // delete gp_atc_tracker;
    pthread_mutex_destroy(&g_cmd_mutex);
    pthread_mutex_destroy(&g_pose_mutex);
 
    return 0;
}

