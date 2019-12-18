#include "model_helper.h"
#include "trafficlights_detector.h"
#include <dirent.h>
#include <fstream>

using namespace std;

int test()
{
    string filename ="/home/user/sc_ws/src/output_tflite_graph_edgetpu.tflite";
    MobilenetV1 mv1;
    mv1.init(filename);

    int correct = 0;
    int error = 0;
    string dir_path = "/home/user/sc_ws/src/test_data/tl_data";
    struct dirent *files_in_dir;  // Pointer for directory entry
    DIR *dir = opendir(dir_path.c_str());
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
                cout<<"no label file"<<endl;
                break; 
            } 
        }
        
        Mat img = imread(full_imgfile,cv::IMREAD_UNCHANGED);
        cv::cvtColor(img, img, CV_RGB2BGR);
        Mat img_resized;
        resize(img, img_resized, Size(224,224));

        std::vector<uint8_t> input;
        if(img_resized.isContinuous())
        {
            input.insert(input.end(),img_resized.data,img_resized.data + 224*224*3);
        }
   
        cout<<"input.size():"<<input.size()<<endl;
        
        int index = mv1.inference(input);
        //int index = -1;
        if(index != true_class_index - 1)
        {
            std::cout <<" error : "
                      <<" pre_index:"<<index
                      <<" true_index:"<<true_class_index
                      <<endl;
                      
            error += 1;          
        }
        else
        {
            correct += 1;
        }
        
        cout<<"correct:"<<correct<<" error:"<<error<<endl;       
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
