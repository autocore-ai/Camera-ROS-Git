#include "model_helper.h"
#include "trafficlights_detector.h"
#include <dirent.h>
#include <fstream>

using namespace std;

//test model accuracy
int test()
{
    string model_path ="/home/user/sc_ws/src/trafficlights_mobilenetv1_edgetpu.tflite";
    MobilenetV1 mv1;
    mv1.init(model_path);

    map<string,int> cls2id_map={
        {"green",0},
        {"yellow",1},
        {"red",2},
        {"unknow",3}
    };

    string root_dir = "/home/user/sc_ws/src/test_data/tl_data2/";
    for(auto e : cls2id_map)
    {
        int correct = 0;
        int error = 0;

        string color = e.first;
        int true_class_index = e.second;

        string dir_path = root_dir + color;
        struct dirent *files_in_dir;  // Pointer for directory entry
        DIR *dir = opendir(dir_path.c_str());
        while ((files_in_dir = readdir(dir)) != NULL)
        {
            if(strstr(files_in_dir->d_name, "png") == NULL) 
            {
                continue;
            }

            string filename = files_in_dir->d_name;
            string full_imgfile = dir_path + '/' + filename;

            Mat img = imread(full_imgfile,cv::IMREAD_UNCHANGED);
            cv::cvtColor(img, img, CV_RGB2BGR);
            Mat img_resized;
            resize(img, img_resized, Size(224,224));

            std::vector<uint8_t> input;
            if(img_resized.isContinuous())
            {
                input.insert(input.end(),img_resized.data,img_resized.data + 224*224*3);
            }
               
            int index = mv1.inference(input);
            if(index != true_class_index)
            {
                std::cout <<" error : " <<" pre_index:"<<index
                          <<" true_index:"<<true_class_index
                          <<endl;
                        
                error += 1;          
            }
            else
            {
                correct += 1;
            }       
        }

        cout<<color<<" correct:"<<correct<<" error:"<<error<<endl; 
    }

    return 0;
}

int main(int argc, char *argv[])
{
    //test();
    //return 0;

    TrafficLightsDetector detector;
    detector.init(argc,argv);
    
    ros::spin();
    
    return 0;
}
