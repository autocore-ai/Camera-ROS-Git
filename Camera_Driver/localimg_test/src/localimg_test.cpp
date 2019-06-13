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

    ros::NodeHandle private_nh("~");

    string img_src_topic;
    private_nh.param<std::string>("image_source_topic", img_src_topic, "/usb_cam/image_raw");
    ROS_INFO("Setting img_src_topic to %s",img_src_topic.c_str());

    string test_img_path;
    private_nh.param<std::string>("test_img", test_img_path, "");
    ROS_INFO("Setting test_img_path to %s",test_img_path.c_str());
    
	ros::Publisher  camera_pub = nh.advertise<sensor_msgs::Image>("/usb_cam/image_raw", 1);

    //string img = "/home/nano/suchang/fuck_raw.jpg";
	cv::Mat frame = cv::imread(test_img_path,CV_LOAD_IMAGE_COLOR);
    if(frame.empty())
    {
        cout<<"empty frame:"<<test_img_path<<endl;
        return -1;       
    }
	//cv::resize(frame, frame, cv::Size(1080, 1080));  
	//captRefrnc >> frame;

	ros::Rate loop_rate(5);
	while((nh.ok())&&(!frame.empty()))
	//while(1)
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
	   // captRefrnc >> frame;
	}
	return 0;
}
