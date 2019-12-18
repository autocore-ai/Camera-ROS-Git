#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <unistd.h>

#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/Image.h>
#include <stdio.h>

using namespace std;
using namespace cv;



int main(int argc,char **argv)
{

	ros::init(argc, argv, "opencv_ros");
	ros::NodeHandle nh;
	
	ros::Publisher  camera_pub = nh.advertise<sensor_msgs::Image>("/usb_cam/image_raw", 1);

	cv::VideoCapture captRefrnc("/dev/video0");
        if (!captRefrnc.isOpened())
        {
           std::cout  << "Could not open video " << argv[1] << std::endl;
           return -1;
   	}

       captRefrnc.set(CV_CAP_PROP_FRAME_WIDTH,1960);
       captRefrnc.set(CV_CAP_PROP_FRAME_HEIGHT,1280);
	cv::Mat frame;
	captRefrnc >> frame;

	ros::Rate loop_rate(1);
	while((nh.ok())&&(!frame.empty()))
	{
	    sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", frame).toImageMsg();
        msg->header.frame_id = "camera";
	    msg->header.stamp.sec = ros::Time::now().sec;
	    msg->header.stamp.nsec = ros::Time::now().nsec;
        camera_pub.publish(msg);
		
		ros::spinOnce();
	    loop_rate.sleep();
        //cv::imshow("demo",frame);
        //cv::waitKey(10);
	    captRefrnc >> frame;
	}
	return 0;
}
