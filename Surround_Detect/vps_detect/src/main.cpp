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
    VpsDetector detector;
    detector.init(argc,argv);

    if(detector.m_test)
    {
        detector.test();

        return 0;
    }
    
    ros::spin();
    
    return 0;
}
