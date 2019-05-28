#include "vps_detector.h"
#include "yolo_helper.h"

#include <unistd.h>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <pthread.h>

#include <sys/time.h>

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

int main(int argc, char *argv[])
{
    /*
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
    */

    CarParkMgr cpm;
    
    ros::spin();
    
    return 0;
}
