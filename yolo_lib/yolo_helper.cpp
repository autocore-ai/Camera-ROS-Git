#include "yolo_helper.h"

#include "ds_image.h"
#include "trt_utils.h"
#include "yolo.h"
#include "yolo_config_parser.h"
#include "yolov2.h"
#include "yolov3.h"

#include <experimental/filesystem>
#include <fstream>
#include <string>
#include <sys/time.h>
#include <iostream>

#include <ros/ros.h>

using namespace std;


YoloHelper::YoloHelper(/* args */)
{
}

YoloHelper::~YoloHelper()
{
}


void YoloHelper::parse_config_params(int argc, char** argv)
{
    // parse config params
    yoloConfigParserInit(argc, argv);
    NetworkInfo yoloInfo = getYoloNetworkInfo();
    InferParams yoloInferParams = getYoloInferParams();
    uint64_t seed = getSeed();
    std::string networkType = getNetworkType();
    std::string precision = getPrecision();
    std::string testImages = getTestImages();
    std::string testImagesPath = getTestImagesPath();
    m_decode = getDecode();
    m_doBenchmark = getDoBenchmark();
    m_viewDetections = getViewDetections();
    m_saveDetections = getSaveDetections();
    m_saveDetectionsPath = getSaveDetectionsPath();
    m_batchSize = getBatchSize();
    m_shuffleTestSet = getShuffleTestSet();

    srand(unsigned(seed));

    //std::unique_ptr<Yolo> inferNet{nullptr};
    if ((networkType == "yolov2") || (networkType == "yolov2-tiny"))
    {
        m_inferNet = std::unique_ptr<Yolo>{new YoloV2(m_batchSize, yoloInfo, yoloInferParams)};
    }
    else if ((networkType == "yolov3") || (networkType == "yolov3-tiny"))
    {
        m_inferNet = std::unique_ptr<Yolo>{new YoloV3(m_batchSize, yoloInfo, yoloInferParams)};
    }
    else
    {
        assert(false && "Unrecognised network_type. Network Type has to be one among the following : yolov2, yolov2-tiny, yolov3 and yolov3-tiny");
    }

/*
    if (testImages.empty())
    {
        std::cout << "Enter a valid file path for test_images config param" << std::endl;
        return -1;
    }
*/
    ROS_INFO("m_saveDetections:%s",m_saveDetections?"true":"false");
    ROS_INFO("m_saveDetectionsPath:%s",m_saveDetectionsPath.c_str());
}
   
std::vector<BBoxInfo> YoloHelper::do_inference(const cv::Mat& image_org)
{
    //std::vector<DsImage> dsImages;
    dsImages.clear();

    dsImages.emplace_back(image_org, m_inferNet->getInputH(),m_inferNet->getInputW());
    cv::Mat trtInput = blobFromDsImages(dsImages, m_inferNet->getInputH(), m_inferNet->getInputW());
    
    double inferElapsed = 0;
    struct timeval inferStart, inferEnd;
    gettimeofday(&inferStart, NULL);

    m_inferNet->doInference(trtInput.data, dsImages.size());
    
    gettimeofday(&inferEnd, NULL);
    inferElapsed += ((inferEnd.tv_sec - inferStart.tv_sec) + (inferEnd.tv_usec - inferStart.tv_usec) / 1000000.0) * 1000;
    std::cout << " Inference time per image : " << inferElapsed  << " ms" << endl;

    std::vector<BBoxInfo> boxes;

    for (uint imageIdx = 0; imageIdx < dsImages.size(); ++imageIdx)
    {
        auto & curImage = dsImages.at(imageIdx);
        auto binfo = m_inferNet->decodeDetections(imageIdx, curImage.getImageHeight(),
                                                curImage.getImageWidth());
        boxes  = nmsAllClasses(m_inferNet->getNMSThresh(), binfo, m_inferNet->getNumClasses());
        
        // box_idx = 0
        // for (auto b : remaining)
        // {
        //     cout<<"box_idx:"<< box_idx << endl;
        //     cout<<"boundingbox:"<<b.box.x1<<","<<b.box.y1<<","<<b.box.x2<<","<<b.box.y2<<endl;
        //     cout<<"label:"<< b.label<< endl;
        //     cout<<"classId:"<< b.classId <<endl;
        //     cout<<"prob:"<< b.prob <<endl;
        //     cout<<"class_name:"<< m_inferNet->getClassName(b.label)<<endl;
        // }

        for (auto b : boxes)
        {
            if (m_inferNet->isPrintPredictions())
            {
                printPredictions(b, m_inferNet->getClassName(b.label));
            }
            curImage.addBBox(b, m_inferNet->getClassName(b.label));
        }

         if (m_viewDetections)
         {
             curImage.showImage();
         }

         if(m_saveDetections)
         {
            curImage.saveImageJPEG(m_saveDetectionsPath);
         }
     }

     return boxes;
}


cv::Mat YoloHelper::get_marked_image(int imageIndex)
{
    DsImage &curImage = dsImages.at(imageIndex);

    return curImage.get_marked_image();
}

