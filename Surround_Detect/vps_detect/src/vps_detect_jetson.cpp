#include "trafficlights_detector.h"
#include "yolo_helper.h"

#include <unistd.h>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <pthread.h>

#include <sys/time.h>
#include "status_machine.h"

#include <ros/ros.h>
//#include "autoware_msgs/image_obj.h"
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

using namespace cv;
using namespace std;



// active loop
/*
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
        active_msg.proc_name = "trafficdetect";
	active_msg.set = unsigned(wake_up);	
        pub_traffic_active.publish(active_msg);
	//std::cout<<"heart beat\n";
	sleep(1);
    }
    std::cout<<"TRAFFIC QUIT\n";
    sleep(3);

    exit(0);
}
*/


int main(int argc, char *argv[])
{
    bool test=false;    
    if(test)
    {
        YoloHelper yolo_helper;
        yolo_helper.parse_config_params(argc,argv);

        string img_path="/home/nano/Downloads/test_img/0.jpeg";
        cv::Mat test_img = cv::imread(img_path, CV_LOAD_IMAGE_COLOR);

        yolo_helper.do_inference(test_img);
        return 0;
    }

    VpsDetector detector;
    detector.init(argc,argv);
  
    // int epoch_frames = atoi(refresh_epoch.c_str());
    // if(1>epoch_frames)
    // {
	//     epoch_frames =10;
    //     ROS_INFO(" Warning:refresh_epoch value should >=1,using default value(%d)",epoch_frames);
    // }
    
    //create active thread
    /*
    pthread_t pub_active_tid;
    if(pthread_create(&pub_active_tid,NULL,auto_active_pub,NULL)!=0)
    {
	    ROS_INFO("error in pthread_create");
	    exit(0);
    }
    */
    ros::spin();

    //thread exit
    /*
    if(pthread_join(pub_active_tid,NULL)!=0)
    {
        ROS_INFO("error in pthread!");
    }*/
    
    return 0;
}
