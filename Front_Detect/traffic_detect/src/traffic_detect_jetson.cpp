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

#include <dirent.h>
#include <fstream>



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
    //test_bgr(3);
    //return 0;
    
    if(test)
    {
        YoloHelper yolo_helper;
        yolo_helper.parse_config_params(argc,argv);
        
        string dir_path = "/home/nano/workspace_sc/failed/";
        struct dirent *files_in_dir;  // Pointer for directory entry
        DIR *dir = opendir(dir_path.c_str());

        std::map<int,std::pair<int,int>> detect_map;
        FILE * f_fail =  fopen("/home/nano/workspace_sc/failed.txt","w"); 
        while ((files_in_dir = readdir(dir)) != NULL) 
        {   
            if(strstr(files_in_dir->d_name, "png") == NULL) 
            {
                continue;
            }

            //cout<<files_in_dir->d_name<<endl;  

            string filename = files_in_dir->d_name;
            string labelname = filename;
            labelname.replace(labelname.end()-4,labelname.end(),".txt",4);

            //cout<<labelname<<endl;

            string full_imgfile = dir_path + '/' + filename;
            string full_labelfile = dir_path + '/' + labelname;

            std::ifstream infile(full_labelfile);
            std::string line;
            int true_class_index;float x;float y;float w;float h;
            while (std::getline(infile, line))
            {
                std::istringstream iss(line);
                
                if (!(iss >> true_class_index >> x >> y >> w >> h)) 
                { 
                    cout<<"error!!!!!!!!!!!!!"<<endl;
                    break; 
                } 

                //cout<<true_class_index<<endl;
            }

            if(true_class_index != 3)
            {
                //continue;
            }
            
            cout<<"test_img:"<<full_imgfile<<endl;
            //cout<<"full_labelfile:"<<full_labelfile<<endl;
            cv::Mat test_img = cv::imread(full_imgfile, CV_LOAD_IMAGE_COLOR);
            int detect_class_index = -1;
/*            
            std::vector<BBoxInfo> boxes = yolo_helper.do_inference(test_img);
            for(BBoxInfo b:boxes)
            {
                detect_class_index = b.label;
                break;
            }
*/  
            double inferElapsed = 0;
            struct timeval inferStart, inferEnd;
            gettimeofday(&inferStart, NULL);

            detect_class_index = yolo_helper.judge_lights_color(test_img);
            
            gettimeofday(&inferEnd, NULL);
            inferElapsed += ((inferEnd.tv_sec - inferStart.tv_sec) + (inferEnd.tv_usec - inferStart.tv_usec) / 1000000.0) * 1000;
            std::cout << " Inference time per image : " << inferElapsed  << " ms" << endl;
            
            //break;
            
            if(detect_class_index == true_class_index)
            {
                detect_map[true_class_index].first += 1;
            }
            else
            {
                detect_map[true_class_index].second += 1;
                cout<<"fail test_img:"<<full_imgfile
                    <<"detect_class_index:"<<detect_class_index
                    <<"true_class_index"<<true_class_index<<endl;
                //break;
                fwrite(full_imgfile.c_str(), full_imgfile.size(), 1, f_fail);
            }

            
        }

        for(auto iter = detect_map.begin();iter != detect_map.end();iter++)
        {
            int true_index = iter->first;
            int correct_num = iter->second.first;
            int error_num = iter->second.second;
            cout<<true_index<<":"<<correct_num<<","<<error_num<<endl;
        }
        
        fclose(f_fail);
        return 0;
    }

    TrafficLightsDetector detector;
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
