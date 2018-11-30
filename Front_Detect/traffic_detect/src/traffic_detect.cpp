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
#include <string>
#include <vector>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "tengine_c_api.h"
#include <sys/time.h>
#include "status_machine.h"

#include <ros/ros.h>
//#include "autoware_msgs/image_obj.h"
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include "std_msgs/UInt8.h"

#define DEF_PROTO "models/MobileNetSSD_deploy.prototxt"
#define DEF_MODEL "models/MobileNetSSD_deploy.caffemodel"
#define DEF_IMAGE "tests/images/ssd_dog.jpg"
#define DEF_ROI   "550,250,300,300"

using namespace cv;
using namespace std;

typedef unsigned char UINT8;

struct Box
{
    float x0;
    float y0;
    float x1;
    float y1;
    int class_idx;
    float score;
};

//roslaunch 参数
std::string proto_file;
std::string model_file;
std::string model_name;
std::string roi_region;
std::string refresh_epoch;
std::string image_source_topic;
std::string status_code_topic;
std::string image_pub_topic;

// taffic light signals status maintainer
TFSMaintainer *p_auto_maintainer;

// image transport
static image_transport::ImageTransport *p_pub_it=NULL;

//订阅发布topic
static image_transport::Publisher pub_roi_raw;
static ros::Publisher pub_status_code;
static ros::Publisher pub_image_obj;
static ros::Subscriber sub_image_raw;

//ssd算法参数
float        *g_input_data = NULL;
tensor_t      g_input_tensor;
graph_t       g_graph;

int           g_img_h  = 300;
int           g_img_w  = 300;
int           g_roi_x ;
int           g_roi_y ;
int           g_roi_w ;
int           g_roi_h;

UINT8 status_encode(bool go_up,bool go_left,bool go_right)
{
    UINT8 up =0;
    UINT8 left = 0;
    UINT8 right = 0;
    UINT8 code;
    if(go_up)
	up = 1;
    if(go_left)
	left = 1;
    if(go_right)
	right =1;
    code = 4*up+2*left+1*right;  
    return code;	
	 
}
// set traffic light status according detected traffic light signals
void set_machine_status(TFSMachine *p_status_machine,std::vector<Box> boxes)
{   // status machine default green red
    bool stop_red_open =false;
    bool go_green_left = false;
    bool go_green_right = false;
    bool stop_red_left = false;
    bool stop_red_right = false;
    for(unsigned int i=0;i<boxes.size();i++)
    {
        Box box = boxes[i];
        if(4==box.class_idx) //red stop
            stop_red_open = true;              
        if(5==box.class_idx)
            stop_red_left = true;
        if(6 == box.class_idx)
            stop_red_right =true;
        if(2==box.class_idx)
            go_green_left = true;
        if(3==box.class_idx)
            go_green_right =true;

    }
    if(stop_red_open) // red light, just need check ooLeft, goRight status
    {
        p_status_machine->add_round_tl_signal(!stop_red_open);
        
        if(go_green_left) //arrow left green open
            p_status_machine->add_arrow_tl_signal("goLeft");
        if(go_green_right)
            p_status_machine->add_arrow_tl_signal("goRight");    

    }else //go green open && stop red or red green light missing
    {
       if(stop_red_left)
            p_status_machine->add_arrow_tl_signal("stopLeft");
        if(stop_red_right)
            p_status_machine->add_arrow_tl_signal("stopRight");
    }
}

// ---- split string
std::vector<int> visplit(std::string str,std::string pattern)
{
    std::string::size_type pos;
    std::vector<int> result;
    str += pattern;
    unsigned int size = str.size();
    for (unsigned int i = 0; i < size; i++)
    {
        pos = str.find(pattern, i);
        if (pos < size)
        {
            std::string s = str.substr(i, pos - i);
            result.push_back(atoi(s.c_str()));
            i = pos + pattern.size() - 1;
        }
    }
    return result;
}
//  ROI region valid check
bool roi_region_is_valid(int src_w,int src_h,int roi_x,int roi_y,int roi_w,int roi_h)
{
    if((0>roi_x) ||((src_w-roi_w)<roi_x))
    {
        ROS_INFO("Error: ROI x0 or x1 is out of image X-axes.");
        return false;
    }
    if((0>roi_y) ||((src_h-roi_h)<roi_y))
    {
        ROS_INFO("Error: ROI y0 or y1 is out of image Y-axes.");
        return false;
    }
    return true;

}
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

void post_process_ssd(cv::Mat& image_input,float threshold,float* outdata,int num, cv::Mat& img)
{
    const char* class_names[] = {"background",
                        "go", "goLeft", "goRight", "stop",
                        "stopLeft", "stopRight", "car", "cat", "chair",
                        "cow", "diningtable", "dog", "horse",
                        "motorbike", "person", "pottedplant",
                        "sheep", "sofa", "train", "tvmonitor"};
    img = image_input;
    int raw_h = img.size().height;
    int raw_w = img.size().width;
    //std::vector<Box> boxes;
    std::vector<Box> traffic_boxes;
    int line_width=raw_w*0.01;

    //printf("detect ruesult num: %d \n",num);
    for (int i=0;i<num;i++)
    {
        if(outdata[1]>=threshold)
        {
            if((1<=outdata[0])&&(6>=outdata[0])) // get traffic light bnding box
            {   
                Box box;
                box.class_idx=outdata[0];
                box.score=outdata[1];
                box.x0=outdata[2]*raw_w;
                box.y0=outdata[3]*raw_h;
                box.x1=outdata[4]*raw_w;
                box.y1=outdata[5]*raw_h;
                traffic_boxes.push_back(box);

                printf("%s\t:%.0f%%\n", class_names[box.class_idx], box.score * 100);
                printf("BOX:( %g , %g ),( %g , %g )\n",box.x0,box.y0,box.x1,box.y1);
            }         
        }
        outdata+=6;
    }

    // show traffic lights signals
    for(int i=0;i<(int)traffic_boxes.size();i++)
    {
        Box box=traffic_boxes[i];
        cv::Scalar bnd_color;
        if(box.class_idx <=3) //green go
            bnd_color = cv::Scalar(0,255,0);
        else
            bnd_color =cv::Scalar(0,0,255);

        cv::rectangle(img, cv::Rect(box.x0, box.y0,(box.x1-box.x0),(box.y1-box.y0)),bnd_color,line_width);
        std::ostringstream score_str;
        score_str<<box.score;
        std::string label = std::string(class_names[box.class_idx]) + ": " + score_str.str();
        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        cv::rectangle(img, cv::Rect(cv::Point(box.x0,box.y0- label_size.height),
                                  cv::Size(label_size.width, label_size.height + baseLine)),
                      bnd_color, CV_FILLED);
        cv::putText(img, label, cv::Point(box.x0, box.y0),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }

    // ----- create traffic status machine to parse current signals  -------
    TFSMachine auto_status_machine;
    set_machine_status(&auto_status_machine,traffic_boxes);
    bool go_up,go_left,go_right;
    auto_status_machine.get_tls_go_code(go_up,go_left,go_right);
    
    // ----- update traffic status maintainer and get stable status order
    // update 
    p_auto_maintainer->update_status(go_up,go_left,go_right);
    // get stable go status
    p_auto_maintainer->get_stable_go_status(go_up,go_left,go_right);
    std::string status_info = p_auto_maintainer->get_stable_status_label();
    cv::putText(img,status_info,cv::Point(20,50),cv::FONT_HERSHEY_SIMPLEX,0.7,cv::Scalar(0,255,0,2));    
    // publish roi raw msg 
    //sensor_msgs::ImagePtr roi_msg =cv_bridge::CvImage(std_msgs::Header(), "bgr8", img).toImageMsg();
    //pub_roi_raw.publish(roi_msg);
    //ros::Time time = ros
    cv_bridge::CvImage cvi;
    cvi.image = img;
    sensor_msgs::Image im;
    cvi.toImageMsg(im);
    pub_roi_raw.publish(im);
   
    // publish traffic status code
    std_msgs::UInt8 status_msg;
    status_msg.data = status_encode(go_up,go_left,go_right);
    pub_status_code.publish(status_msg);
          
  
}

static void image_callback(const sensor_msgs::Image& image_source)
//static void image_callback(const sensor_msgs::ImageConstPtr &image_source)
{
    cv_bridge::CvImagePtr cv_image = cv_bridge::toCvCopy(image_source, "bgr8");
    cv::Mat frame = cv_image->image;
    //cv::Mat frame = cv_bridge::toCvShare(image_source,"bgr8")->image;
    int src_w = frame.cols;
    int src_h =frame.rows;
    bool ret =roi_region_is_valid(src_w,src_h,g_roi_x,g_roi_y,g_roi_w,g_roi_h);
    
    // unvalid ROI,skip...
    if(!ret)
        return;
   // get roi data
   cv::Mat frame_input =  frame(cv::Rect(g_roi_x,g_roi_y,g_roi_w,g_roi_h));

    //消息发布准备
    //autoware_msgs::image_obj msg;
    //msg.header = image_source.header;
    //msg.header.frame_id = "image obj";

    if(g_input_data == NULL)
    {
        //申请内存，注意释放
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
    float show_threshold=0.5;
    cv::Mat frame_show;

    post_process_ssd(frame_input, show_threshold, outdata, num, frame_show);
    cv::rectangle(frame,cv::Point(g_roi_x,g_roi_y),cv::Point(g_roi_x+g_roi_w,g_roi_y+g_roi_h),cv::Scalar(255,0,0));
    //imshow("demo_traffic_detect", frame);
    //cvWaitKey(15);
   
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
    
    private_nh.param<std::string>("roi_region", roi_region, "550,250,300,300");
    ROS_INFO("Setting model_name to %s", roi_region.c_str());
    
    private_nh.param<std::string>("refresh_epoch",refresh_epoch,"10");
    ROS_INFO("Setting refresh_epoch to %s",refresh_epoch.c_str());

    private_nh.param<std::string>("image_source_topic", image_source_topic, "/usb_cam/image_raw");
    ROS_INFO("Setting image_source_topic to %s", image_source_topic.c_str());

    private_nh.param<std::string>("status_code_topic", status_code_topic, "/traffic/tls_code");
    ROS_INFO("Setting staus_code_topic to %s", status_code_topic.c_str());
    
    private_nh.param<std::string>("image_pub_topic", image_pub_topic, "/traffic/image_raw");
    ROS_INFO("Setting staus_code_topic to %s", image_pub_topic.c_str());    
    

    //pub_image_obj = node.advertise<autoware_msgs::image_obj>(image_object_topic, 1);
    pub_status_code = node.advertise<std_msgs::UInt8>(status_code_topic,1);
    sub_image_raw = node.subscribe(image_source_topic, 1, image_callback);
    //
    p_pub_it = new image_transport::ImageTransport(node);
    pub_roi_raw = p_pub_it->advertise(image_pub_topic,1);
}

int main(int argc, char *argv[])
{
    //ros初始化
    init_ros(argc, argv);
    
    // roi region init
    std::vector<int> roi_vec =visplit(roi_region,",");
    if(roi_vec.size()<3)
    {
        ROS_INFO("Error: roi_region at least specified 3 integers");
        return -1;
    }else if(roi_vec.size()<4)
    {
        roi_vec.push_back(roi_vec[2]);       
    }
    g_roi_x = roi_vec[0];
    g_roi_y = roi_vec[1];
    g_roi_w = roi_vec[2];
    g_roi_h = roi_vec[3];
    
    // TFSMaintainer init
    int epoch_frames = atoi(refresh_epoch.c_str());
    if(1>epoch_frames)
    {
	epoch_frames =10;
        ROS_INFO(" Warning:refresh_epoch value should >=1,using default value(%d)",epoch_frames);
    }
    p_auto_maintainer = new TFSMaintainer(epoch_frames);
    
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
    ros::spin();

    postrun_graph(g_graph);
    if(g_input_data)
    {
        free(g_input_data);
        g_input_data = NULL;
        printf("[SSD]free input data ok\n");
    }
    destroy_runtime_graph(g_graph);
    remove_model(model_name.c_str());
    delete p_pub_it; 
    delete p_auto_maintainer;
    return 0;
}
