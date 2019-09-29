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


void test_dnndk()
{
        YoloHelper yolo_helper;
        //yolo_helper.init();
        dpuOpen();

        Mat img = imread("/root/yolov3_deploy/coco_test.jpg");

        DPUKernel *kernel = dpuLoadKernel("yolo");

        DPUTask *task = dpuCreateTask(kernel, 0);

        auto begin = std::chrono::system_clock::now();

        yolo_helper.runYOLO(task, img);

        auto end = std::chrono::system_clock::now();

        auto elsp = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);


        std::cout << "elsp:" << elsp.count() << std::endl;

        imwrite("result.jpg", img);

        imshow("Xilinx DPU", img);

        waitKey(0);
}

void test_yolohelper(const string cfgfile_path)
{
    YoloHelper yolo_helper;
    yolo_helper.test_parse_cfgfile(cfgfile_path);
}

void test_xlinx(string dir_path,string filename)
{
    YoloHelper yolo_helper;
    yolo_helper.init();
   
    std::map<int,std::pair<int,int>> detect_map;
    struct dirent *files_in_dir;  // Pointer for directory entry
    DIR *dir = opendir(dir_path.c_str());
    FILE * f_fail =  fopen(filename.c_str(),"w");
    int success = 0;int fail = 0;
    while ((files_in_dir = readdir(dir)) != NULL) 
    {
        if(strstr(files_in_dir->d_name, "png") == NULL) 
        {
            continue;
        }

        string filename = files_in_dir->d_name;
        string labelname = filename;
        labelname.replace(labelname.end()-4,labelname.end(),".txt",4);
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
        }

        cv::Mat test_img = cv::imread(full_imgfile, CV_LOAD_IMAGE_COLOR);
        cv::resize(test_img,test_img,cv::Size(608,608));

        int detect_class_index = -1;
        double inferElapsed = 0;
        struct timeval inferStart, inferEnd;
        gettimeofday(&inferStart, NULL);

        DPUTask* task = yolo_helper.get_task();
        yolo_helper.runYOLO(task, test_img);
        vector<BBoxInfo> v_boxes = yolo_helper.get_inference_result();
        BBoxInfo t;
        t.classId = -1;
        cout << "predict "<<v_boxes.size()<<" boxes"<<endl;
        for(auto b : v_boxes)
        {
            t = b;
            break;
        }
        if(t.classId != true_class_index)
        {
            fail += 1;

            int classid = t.classId;
            cout << filename << ":fail!!!" <<endl;
            cout << "predict:"<<classid<<" truth:"<<true_class_index
                 << "predict prob"<<t.prob<<endl;
                
            fwrite(full_imgfile.c_str(), full_imgfile.size(), 1, f_fail); 
            fwrite("\n",1,1,f_fail);
        }
        else
        {
            success += 1;
            //cout << filename << ":correct!!!" <<endl;
        }
        
        gettimeofday(&inferEnd, NULL);
        inferElapsed += ((inferEnd.tv_sec - inferStart.tv_sec) + (inferEnd.tv_usec - inferStart.tv_usec) / 1000000.0) * 1000;
        std::cout << " Inference time per image : " << inferElapsed  << " ms" << endl;

        cout << "success:"<< success <<" fail:"<< fail <<endl;
   } 

   fclose(f_fail);
   cout << "success:"<< success <<" fail:"<< fail <<endl;
}
         
int main(int argc, char *argv[])
{
    for(int i = 0; i<argc; i++)
    {
        cout<<argv[i]<<endl;
    }

    string dir_path = "/root/sc/data/test";
    string filename = "/root/sc/data/detect_failed.txt";

    //test_yolohelper(argv[1]);
    test_xlinx(dir_path,filename);
    //test_dnndk();
    
    return 0;
 
    TrafficLightsDetector detector;
    detector.init(argc,argv);
  
    ros::spin();
    
    return 0;
}
