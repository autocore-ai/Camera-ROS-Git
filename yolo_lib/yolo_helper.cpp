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
using namespace cv;

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
   
std::vector<BBoxInfo> YoloHelper::do_inference(const cv::Mat& image_org,bool simu)
{
    //std::vector<DsImage> dsImages;
    dsImages.clear();
    cv::Mat trtInput;
    
    if(simu)
    {
        return judge_red_yellow_green(image_org);
    }
    //else
    {
        dsImages.emplace_back(image_org, m_inferNet->getInputH(),m_inferNet->getInputW());
        trtInput = blobFromDsImages(dsImages, m_inferNet->getInputH(), m_inferNet->getInputW());
    }


    double inferElapsed = 0;
    struct timeval inferStart, inferEnd;
    gettimeofday(&inferStart, NULL);

    m_inferNet->doInference(trtInput.data,dsImages.size());
    
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


string YoloHelper::type2str(int type) {
  string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}


float YoloHelper::get_percentage(Mat img_hsv,
                           int iLowH,
                           int iHighH,
                           int iLowS, 
                           int iHighS,
                           int iLowV,
                           int iHighV)
{
    
    Mat imgThresholded;
    inRange(img_hsv, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), imgThresholded); //Threshold the image

    //cout<<imgThresholded.rows<<","<<imgThresholded.cols<<endl;
    string ty =  type2str( imgThresholded.type() );
    //printf("Matrix: %s %dx%d \n", ty.c_str(), imgThresholded.cols, imgThresholded.rows );

    int counts = 0;
    for(int y = 0; y < imgThresholded.rows; y++)
    {
        for(int x = 0; x < imgThresholded.cols; x++)
        {
            int pixel = (int)imgThresholded.at<uchar>(y,x);
            if ( pixel == 255)
            {
                 counts++;
            }
        }
    }
    //cout<<"counts="<<counts<<endl; 
    float percentage = (float)counts/(imgThresholded.cols * imgThresholded.rows);
    //cout<<"percentage="<<percentage<<endl;

    //im_show("img_hsv",imgThresholded);
    return percentage;
}

int YoloHelper::judge_lights_color(cv::Mat test_img)
{
     cv::resize(test_img, test_img, cv::Size(150,300) );
     //im_show("test_img",test_img);
    
    int src_w = test_img.cols;
    int src_h = test_img.rows;
    int roi_x = 0 * src_w;
    int roi_y = 0 * src_h;
    int roi_w = 1 * src_w;
    int roi_h = 1 * src_h;
    
    cv::Rect roi(roi_x, roi_y, roi_w, roi_h); 
    cv::Mat roi_img  = test_img(roi);

    //im_show("roi_img",roi_img);
    //waitKey(0);

    //因为我们读取的是彩色图，直方图均衡化需要在HSV空间做
    Mat img_hsv;
    cvtColor(roi_img,img_hsv,CV_BGR2HSV);
    vector<Mat> hsvSplit;
    split(img_hsv, hsvSplit);
    equalizeHist(hsvSplit[2],hsvSplit[2]);
    merge(hsvSplit,img_hsv);

   int iLowH = 0;
   int iHighH = 10;
 
   int iLowS = 43; 
   int iHighS = 255;
 
   int iLowV = 46;
   int iHighV = 255;

   float r_percent = get_percentage(img_hsv,iLowH,iHighH,iLowS,iHighS,iLowV,iHighV);

   iLowH = 35;
   iHighH = 77;
 
   iLowS = 43; 
   iHighS = 255;
 
   iLowV = 46;
   iHighV = 255;

   float g_percent = get_percentage(img_hsv,iLowH,iHighH,iLowS,iHighS,iLowV,iHighV);

   iLowH = 26;
   iHighH = 34;
   float y_percent = get_percentage(img_hsv,iLowH,iHighH,iLowS,iHighS,iLowV,iHighV);
    
   cout<<r_percent<<","<<g_percent<<","<<y_percent<<endl;
   waitKey(100);

   if(r_percent < 0.01 && g_percent < 0.01 && y_percent < 0.01 )
   {
        cout<<"background"<<endl;
        return Color::background;
   }
   else if(r_percent > 0.03 || ((r_percent > 2*g_percent) && (r_percent > 2*y_percent)))
   {
        cout<<"red"<<endl;
        return Color::red;
   }
   else if(g_percent > 0.03 || ((g_percent > 2*r_percent) && (g_percent > 2*y_percent)))
   {
        cout<<"green"<<endl;
        return Color::green;
   }
   else if(y_percent > 0.03 || ((y_percent > 2*r_percent) && (y_percent > 2*g_percent)))
   {
        cout<<"yellow"<<endl;
        return Color::yellow;
   }
   else
   {
        cout<<"background"<<endl;
        return Color::background;
   }
}

int YoloHelper::judge_lights_color(string full_imgfile)
{
    cv::Mat test_img = cv::imread(full_imgfile, CV_LOAD_IMAGE_COLOR);
    return judge_lights_color(test_img);
}

std::vector<BBoxInfo> YoloHelper::judge_red_yellow_green(const cv::Mat& image_org)
{
    cout<<"judge_red_yellow_green"<<endl;
    std::vector<BBoxInfo> vec_boxes;
    vec_boxes.clear();
    BBoxInfo box;

    box.label = judge_lights_color(image_org);
    
    vec_boxes.push_back(box);

    return vec_boxes;
}


