#include "front_detector.h"

FrontDetector::FrontDetector()
{

}

FrontDetector::~FrontDetector()
{

}

void FrontDetector::init()
{
    lights_detector_.init();
    common_obj_detector_.init();
}