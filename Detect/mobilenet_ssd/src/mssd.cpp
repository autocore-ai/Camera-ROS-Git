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

#include <ros/ros.h>
#include "autoware_msgs/image_obj.h"
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>

#define DEF_PROTO "models/MobileNetSSD_deploy.prototxt"
#define DEF_MODEL "models/MobileNetSSD_deploy.caffemodel"
#define DEF_IMAGE "tests/images/ssd_dog.jpg"

using namespace cv;
using namespace std;

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
std::string image_source_topic;
std::string image_object_topic;
//订阅发布topic
static ros::Publisher pub_image_obj;
static ros::Subscriber sub_image_raw;
//ssd算法参数
float        *g_input_data = NULL;
tensor_t      g_input_tensor;
graph_t       g_graph;

int           g_img_h  = 300;
int           g_img_w  = 300;


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
void post_process_ssd(cv::Mat& image_input,float threshold,float* outdata,int num, cv::Mat& img,autoware_msgs::image_obj& msg)
{
    const char* class_names[] = {"background",
                        "aeroplane", "bicycle", "bird", "boat",
                        "bottle", "bus", "car", "cat", "chair",
                        "cow", "diningtable", "dog", "horse",
                        "motorbike", "person", "pottedplant",
                        "sheep", "sofa", "train", "tvmonitor"};

    img = image_input.clone();
    int raw_h = img.size().height;
    int raw_w = img.size().width;
    std::vector<Box> boxes;
    int line_width=raw_w*0.005;
    ROS_INFO("detect ruesult num: %d ",num);
    for (int i=0;i<num;i++)
    {
        if(outdata[1]>=threshold)
        {
            Box box;
            box.class_idx=outdata[0];
            box.score=outdata[1];
            box.x0=outdata[2]*raw_w;
            box.y0=outdata[3]*raw_h;
            box.x1=outdata[4]*raw_w;
            box.y1=outdata[5]*raw_h;
            boxes.push_back(box);
        }
        outdata+=6;
    }
    for(int i=0;i<(int)boxes.size();i++)
    {
        Box box=boxes[i];
        cv::rectangle(img, cv::Rect(box.x0, box.y0,(box.x1-box.x0),(box.y1-box.y0)),cv::Scalar(255, 255, 0),line_width);
        std::ostringstream score_str;
        score_str<<box.score;
        std::string label = std::string(class_names[box.class_idx]) + ": " + score_str.str();
        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        cv::rectangle(img, cv::Rect(cv::Point(box.x0,box.y0- label_size.height),
                                  cv::Size(label_size.width, label_size.height + baseLine)),
                      cv::Scalar(255, 255, 0), CV_FILLED);
        cv::putText(img, label, cv::Point(box.x0, box.y0),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }
    //发布/image_obj
    for (int i = 0; i < (int)boxes.size(); ++i) 
    {
		autoware_msgs::image_rect rect;
		Box box = boxes[i];

		rect.x = box.x0;
		rect.y = box.y0;
		rect.width = box.x1-box.x0;
		rect.height = box.y1-box.y0;
		rect.score = box.score;
        msg.type = "car";//class_names[box.class_idx];
		msg.obj.push_back(rect);
	}
    if(boxes.size() > 0)
        pub_image_obj.publish(msg);
}

static void image_callback(const sensor_msgs::Image& image_source)
{
    cv_bridge::CvImagePtr cv_image = cv_bridge::toCvCopy(image_source, "bgr8");
    cv::Mat frame_input = cv_image->image;

    //消息发布准备
    autoware_msgs::image_obj msg;
    msg.header = image_source.header;
    msg.header.frame_id = "image obj";

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

    post_process_ssd(frame_input, show_threshold, outdata, num, frame_show, msg);
    
    imshow("Mobilenet_SSD", frame_show);
    cvWaitKey(15);
   
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

    private_nh.param<std::string>("image_source_topic", image_source_topic, "/image_raw");
    ROS_INFO("Setting image_source_topic to %s", image_source_topic.c_str());

    private_nh.param<std::string>("image_object_topic", image_object_topic, "/image_obj");
    ROS_INFO("Setting image_object_topic to %s", image_object_topic.c_str());

    pub_image_obj = node.advertise<autoware_msgs::image_obj>(image_object_topic, 1);
    sub_image_raw = node.subscribe(image_source_topic, 1, image_callback);
}

int main(int argc, char *argv[])
{
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
 
    return 0;
}
