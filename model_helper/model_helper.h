#ifndef __MODEL_HELPER_H__
#define __MODEL_HELPER_H__

#include <iostream>
#include <string>
#include <vector>
#include <map>
#include "edgetpu.h"
#include "tensorflow/lite/builtin_op_data.h"
#include "tensorflow/lite/kernels/register.h"
#include <chrono> 
#include <sys/time.h>
//using namespace std::chrono; 

using namespace std;
//using namespace cv;

std::unique_ptr<tflite::Interpreter> BuildEdgeTpuInterpreter(
    const tflite::FlatBufferModel& model,
    edgetpu::EdgeTpuContext* edgetpu_context);

/*************************************************************************/
class MobilenetV1
{
public:
    MobilenetV1();
    virtual ~MobilenetV1();

    void init(const string& model_path);
    int inference(const std::vector<uint8_t>& input);
    
    inline int get_width()
    {
        return width_;
    }

    inline int get_height()
    {
        return height_;
    }
private: 
    int inference(const std::vector<uint8_t>& input,const std::unique_ptr<tflite::Interpreter>& interpreter);
private:
    std::unique_ptr<tflite::Interpreter> interpreter_;
    std::unique_ptr<tflite::FlatBufferModel> model_;

    int width_=224;
    int height_=224;

};


/***************************************************************************/
struct BBoxInfo
{
    float xmin_;
    float ymin_;
    float xmax_;
    float ymax_;
    int cls_id_;
    float score_;

    void output()
    {
    }
};


class MobilenetV1SSD
{
public:    
    MobilenetV1SSD();
    ~MobilenetV1SSD();

    void init(const string& model_path);
    vector<BBoxInfo> inference(const std::vector<uint8_t>& input_data);

    inline int get_width()
    {
        return width_;
    }

    inline int get_height()
    {
        return height_;
    }
private:
    int width_=300;
    int height_=300;

    float score_threshold_=0.7;
    float iou_threshold_=0.8;

    std::unique_ptr<tflite::Interpreter> interpreter_;
    std::unique_ptr<tflite::FlatBufferModel> model_;
};
#endif