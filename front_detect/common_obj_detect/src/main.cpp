#include "model_helper.h"
#include "common_obj_detector.h"

#include <dirent.h>
#include <fstream>

using namespace tflite;
using namespace std;

int main(int argc, char *argv[])
{
    CommonObjDetector detector;
    detector.init(argc,argv);
  
    ros::spin();
    
    return 0;
}
