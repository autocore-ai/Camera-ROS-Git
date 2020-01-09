#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <unistd.h>
#include <dirent.h>

#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/Image.h>
#include <stdio.h>
#include <iostream>
#include <csignal>
#include <unistd.h>

using namespace std;
using namespace cv;

void signal_handle(int signum)
{
    cout<<"ctrl + c,exit"<<endl;
    exit(0);
}

int main(int argc,char **argv)
{
    signal(SIGINT,signal_handle);
    signal(SIGTERM,signal_handle);

    ros::init(argc, argv, "localimg_test");
	ros::NodeHandle nh;

    ros::NodeHandle private_nh("~");

    string co_image_source_topic;
    string tl_image_source_topic;
    string test_img_dir;

    private_nh.param<std::string>("co_image_source_topic", co_image_source_topic, "");
    ROS_INFO("Setting co_image_source_topic to %s",co_image_source_topic.c_str());

    private_nh.param<std::string>("tl_image_source_topic", tl_image_source_topic, "");
    ROS_INFO("Setting tl_image_source_topic to %s",tl_image_source_topic.c_str());

    private_nh.param<std::string>("test_img_dir", test_img_dir, "");
    ROS_INFO("Setting test_img_dir to %s",test_img_dir.c_str());
    
	ros::Publisher  co_image_puber = nh.advertise<sensor_msgs::Image>("/usb_cam/image_raw", 1);
    ros::Publisher  tl_image_puber = nh.advertise<sensor_msgs::Image>("/image_raw", 1);

    struct dirent *files_in_dir;  // Pointer for directory entry
    while(1)
    {
        DIR *dir = opendir(test_img_dir.c_str());
        if(dir == NULL)
        {
            cout<<"can not open dir"<<endl;
            return -1;
        }

        while ((files_in_dir = readdir(dir)) != NULL) 
        {
            bool jpg_file = (strstr(files_in_dir->d_name, "jpg") != NULL);
            bool png_file = (strstr(files_in_dir->d_name, "png") != NULL);
            if(!jpg_file && !png_file)
            {
                continue;
            }
            
            string full_file_name = test_img_dir + '/' + files_in_dir->d_name;
            cout<<full_file_name<<endl;

            cv::Mat frame = cv::imread(full_file_name,CV_LOAD_IMAGE_COLOR);
            if(frame.empty())
            {
                cout<<"empty frame:"<<full_file_name<<endl;
                return -1;       
            }

            ros::Rate loop_rate(5);
            //while((nh.ok())&&(!frame.empty()))
            {
                sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", frame).toImageMsg();
                msg->header.frame_id = "camera";
                msg->header.stamp.sec = ros::Time::now().sec;
                msg->header.stamp.nsec = ros::Time::now().nsec;
                co_image_puber.publish(msg); //pub to common obj detector 
                tl_image_puber.publish(msg); //pub to traffic lights detector

                cout << "publish " << full_file_name << endl;
                ros::spinOnce();
                loop_rate.sleep();
            }
        }

        cout << "send done" <<endl;

        sleep(30);
    }

    return 0;
}
