#ifndef FRONT_DETECTOR_H_
#define FRONT_DETECTOR_H_

#include "trafficlights_detector.h"
#include "common_obj_detector.h"

class FrontDetector
{
public:
    FrontDetector();
    ~FrontDetector();

    void init();
private:
    TrafficLightsDetector lights_detector_;
    CommonObjDetector common_obj_detector_;
};

#endif