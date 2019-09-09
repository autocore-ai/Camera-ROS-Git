#include "yolo_helper.h"

#include <experimental/filesystem>
#include <fstream>
#include <string>
#include <sys/time.h>
#include <iostream>
#include <ros/ros.h>

YoloHelper::YoloHelper()
{
}

YoloHelper::~YoloHelper()
{
}
   
std::vector<BBoxInfo> YoloHelper::do_inference(const cv::Mat& image_org,bool simu)
{
    std::vector<BBoxInfo> vBoxes;
    
    if(simu)
    {
        return judge_red_yellow_green(image_org);
    }
    else
    {
        return vBoxes;
    }
}

/******************************************************************************/
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
 
    float percentage = (float)counts/(imgThresholded.cols * imgThresholded.rows);
   
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

   int iLowH = 0;int iHighH = 10;
   int iLowS = 43;int iHighS = 255;
   int iLowV = 46;int iHighV = 255;
   float r_percent = get_percentage(img_hsv,iLowH,iHighH,iLowS,iHighS,iLowV,iHighV);

   iLowH = 35;iHighH = 77;
   float g_percent = get_percentage(img_hsv,iLowH,iHighH,iLowS,iHighS,iLowV,iHighV);

   iLowH = 26;iHighH = 34;
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


//---------------------------------------------------------------------------------------
void YoloHelper::init()
{
    dpuOpen();
    m_kernel = dpuLoadKernel("yolo");
    m_task = dpuCreateTask(m_kernel, 0);
}

void YoloHelper::release_resource()
{
    dpuDestroyTask(m_task);

    /* Destroy DPU Kernels & free resources */
    dpuDestroyKernel(m_kernel);

    /* Dettach from DPU driver & free resources */
    dpuClose();
}

void YoloHelper::runYOLO(DPUTask *task, Mat &img)
{
    // mean values for YOLO-v3 
    float mean[3] = {0.0f, 0.0f, 0.0f};
    int height = dpuGetInputTensorHeight(task, INPUT_NODE);
    int width = dpuGetInputTensorWidth(task, INPUT_NODE);
  
    // feed input frame into DPU Task with mean value 
    setInputImageForYOLO(task, img, mean);
    
    // invoke the running of DPU for YOLO-v3 
    auto begin = std::chrono::system_clock::now();
    dpuRunTask(task);
    auto end = std::chrono::system_clock::now();
    auto elsp = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
    //std::cout << "run task elsp:" << elsp.count() << std::endl;

    postProcess(task, img, width, height);
}

void YoloHelper::setInputImageForYOLO(DPUTask *task, const Mat &frame, float *mean)
{
    int height = dpuGetInputTensorHeight(task, INPUT_NODE);
    int width = dpuGetInputTensorWidth(task, INPUT_NODE);
    int size = dpuGetInputTensorSize(task, INPUT_NODE);
    int8_t *data = dpuGetInputTensorAddress(task, INPUT_NODE);

    cv::Mat flip_img ;
    cv::flip(frame,flip_img ,1); 
    unsigned char *img_data = flip_img.data;  //h*w*c rgb

    float scale = dpuGetInputTensorScale(task, INPUT_NODE);
    for (int i = 0; i < size; ++i)
    {
        data[i] = int(img_data[i]/255. * scale);

        if (data[i] < 0)
            data[i] = 127;
    }
}

/*
void YoloHelper::setInputImageForYOLO(DPUTask *task, const Mat &frame, float *mean)
{
    Mat img_copy;
    int height = dpuGetInputTensorHeight(task, INPUT_NODE);
    int width = dpuGetInputTensorWidth(task, INPUT_NODE);
    int size = dpuGetInputTensorSize(task, INPUT_NODE);

    int8_t *data = dpuGetInputTensorAddress(task, INPUT_NODE);
    image img_new = load_image_cv(frame);  // c*h*w rgb
    image img_yolo = letterbox_image(img_new, width, height);

    vector<float> bb(size);
    //bb重新转换成ｈ*w*c的顺序
    for (int b = 0; b < height; ++b)
    {
        for (int c = 0; c < width; ++c)
        {
            for (int a = 0; a < 3; ++a)
            {
                //  
                bb[b * width * 3 + c * 3 + a] = img_yolo.data[a * height * width + b * width + c];
            }
        }
    }

    float scale = dpuGetInputTensorScale(task, INPUT_NODE);
    for (int i = 0; i < size; ++i)
    {
        data[i] = int(bb.data()[i] * scale);

        if (data[i] < 0)
            data[i] = 127;
    }

    free_image(img_new);
    free_image(img_yolo);
}
*/

//
void YoloHelper::postProcess(DPUTask *task, Mat &frame, int sWidth, int sHeight)
{
    m_boxes.clear();

    //output nodes of YOLO-v3 
    const vector<string> outputs_node = {"layer81_conv", "layer93_conv", "layer105_conv"};

    vector<vector<float>> boxes;
    for (size_t i = 0; i < outputs_node.size(); i++)
    {
        string output_node = outputs_node[i];
        //cout<<"postProcess: "<<output_node<<endl;
        
        int channel = dpuGetOutputTensorChannel(task, output_node.c_str());
        int width = dpuGetOutputTensorWidth(task, output_node.c_str());
        int height = dpuGetOutputTensorHeight(task, output_node.c_str());

        int sizeOut = dpuGetOutputTensorSize(task, output_node.c_str());
        int8_t *dpuOut = dpuGetOutputTensorAddress(task, output_node.c_str());
        float scale = dpuGetOutputTensorScale(task, output_node.c_str());

        //ROS_INFO("(w,h,c):(%d,%d,%d)",width,height,channel);
        
        vector<float> result(sizeOut);
        boxes.reserve(sizeOut);

        // Store every output node results 
        get_output(dpuOut, sizeOut, scale, channel, height, width, result);

        // Store the object detection frames as coordinate information  
        detect(boxes, result, channel, height, width, i, sHeight, sWidth);
    }

    // Restore the correct coordinate frame of the original image 
    correct_region_boxes(boxes, boxes.size(), frame.cols, frame.rows, sWidth, sHeight);

    // Apply the computation for NMS    
    //cout << "boxes size: " << boxes.size() << endl;
    vector<vector<float>> res = applyNMS(boxes, m_classification_cnt, NMS_THRESHOLD);

    float h = frame.rows;
    float w = frame.cols;
    for (size_t i = 0; i < res.size(); ++i)
    {
        float xmin = (res[i][0] - res[i][2] / 2.0) * w + 1.0;
        float ymin = (res[i][1] - res[i][3] / 2.0) * h + 1.0;
        float xmax = (res[i][0] + res[i][2] / 2.0) * w + 1.0;
        float ymax = (res[i][1] + res[i][3] / 2.0) * h + 1.0;
        
        //cout << res[i][res[i][4] + 6] << " ";
        //cout << xmin << " " << ymin << " " << xmax << " " << ymax << endl;

        //x,y,c_x,c_y,index,confidence,prob1,prob2...
        int class_id = res[i][4];
        int classprob_index = res[i][4] + 6;
        float classprob = res[i][classprob_index];
        if (classprob > CONF)
        {
            int type = res[i][4];

            if (type == 0)
            {
                rectangle(frame, cvPoint(xmin, ymin), cvPoint(xmax, ymax), Scalar(0, 0, 255), 1, 1, 0);
            }
            else if (type == 1)
            {
                rectangle(frame, cvPoint(xmin, ymin), cvPoint(xmax, ymax), Scalar(255, 0, 0), 1, 1, 0);
            }
            else
            {
                rectangle(frame, cvPoint(xmin, ymin), cvPoint(xmax, ymax), Scalar(0, 255, 255), 1, 1, 0);
            }

            BBoxInfo b;
            b.classId = class_id;
            b.prob = classprob;
            m_boxes.emplace_back(b);
        }
    }
    
    //cout<<"postProcess end "<<endl;
}


void YoloHelper::detect(vector<vector<float>> &boxes, vector<float> result,
    int channel, int height, int width, int num, int sHeight, int sWidth) 
{

    //printf("c=%d,h=%d,w=%d,num=%d,sH=%d,sW=%d\n",channel,height,width,num,sHeight,sWidth);
    
    vector<float> biases{116,90, 156,198, 373,326, 30,61, 62,45, 59,119, 10,13, 16,30, 33,23};
    int conf_box = 5 + m_classification_cnt;
    float swap[height * width][m_acnhor_count][conf_box]; //[s*s,3,1+4+class]

    for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
            for (int c = 0; c < channel; ++c) {
                int temp = c * height * width + h * width + w;
                //printf("temp=%d,width=%d,with_addr=%p\n",temp,width,&width);
                swap[h * width + w][c / conf_box][c % conf_box] = result[temp];
            }
        }
    }
 
    for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
            for (int c = 0; c < m_acnhor_count; ++c) {
                float obj_score = sigmoid(swap[h * width + w][c][4]);
                
                if (obj_score < CONF)
                    continue;
                vector<float> box;
                
                box.push_back((w + sigmoid(swap[h * width + w][c][0])) / width);
                box.push_back((h + sigmoid(swap[h * width + w][c][1])) / height);
                box.push_back(exp(swap[h * width + w][c][2]) * biases[2 * c + 2 * m_acnhor_count * num] / float(sWidth));
                box.push_back(exp(swap[h * width + w][c][3]) * biases[2 * c + 2 * m_acnhor_count * num + 1] / float(sHeight));
                box.push_back(-1);
                box.push_back(obj_score);
                for (int p = 0; p < m_classification_cnt; p++) {
                    box.push_back(obj_score * sigmoid(swap[h * width + w][c][5 + p]));
                }
                boxes.push_back(box);
            }
        }
    }
}

void YoloHelper::correct_region_boxes(vector<vector<float>>& boxes, int n,
    int w, int h, int netw, int neth, int relative) 
{
    //printf("%s begin \n",__FUNCTION__);
    
    int new_w=0;
    int new_h=0;

    if (((float)netw/w) < ((float)neth/h)) {
        new_w = netw;
        new_h = (h * netw)/w;
    } else {
        new_h = neth;
        new_w = (w * neth)/h;
    }
    for (int i = 0; i < n; ++i){
        boxes[i][0] =  (boxes[i][0] - (netw - new_w)/2./netw) / ((float)new_w/(float)netw);
        boxes[i][1] =  (boxes[i][1] - (neth - new_h)/2./neth) / ((float)new_h/(float)neth);
        boxes[i][2] *= (float)netw/new_w;
        boxes[i][3] *= (float)neth/new_h;
    }

    //printf("%s end \n",__FUNCTION__);
}


 vector<map<string,string>> YoloHelper::parse_cfgfile(const string cfgfilepath)
 {
    assert(file_exists(cfgfilepath));
    std::ifstream file(cfgfilepath);
    assert(file.good());
    std::string line;
    std::vector<std::map<std::string, std::string>> blocks;
    std::map<std::string, std::string> block;

    while (getline(file, line))
    {
        if (line.size() == 0) continue;
        if (line.front() == '#') continue;
        line = trim(line);
        if (line.front() == '[')
        {
            if (block.size() > 0)
            {
                blocks.push_back(block);
                block.clear();
            }
            std::string key = "type";
            std::string value = trim(line.substr(1, line.size() - 2));
            block.insert(std::pair<std::string, std::string>(key, value));
        }
        else
        {
            int cpos = line.find('=');
            std::string key = trim(line.substr(0, cpos));
            std::string value = trim(line.substr(cpos + 1));
            block.insert(std::pair<std::string, std::string>(key, value));
        }
    }
    blocks.push_back(block);
    return blocks;
}

 void YoloHelper::test_parse_cfgfile(const string cfgfilepath)
 {
    vector<map<string,string>> blocks = parse_cfgfile(cfgfilepath);
    vector<float> anchors;
    int number_classes = -1;
    
    for(auto block : blocks)
    {
        string section = block["type"];
        if(section == "yolo")
        {
            number_classes = std::stoi(block["classes"]);
            anchors.clear();
            
            std::string anchor_string = block.at("anchors");
            while (!anchor_string.empty())
            {
                int npos = anchor_string.find_first_of(',');
                if (npos != -1)
                {
                    float anchor = std::stof(trim(anchor_string.substr(0, npos)));
                    anchors.push_back(anchor);
                    anchor_string.erase(0, npos + 1);
                }
                else
                {
                    float anchor = std::stof(trim(anchor_string));
                    anchors.push_back(anchor);
                    break;
                }
            }
        }
    }

    cout << number_classes <<endl;
    for(auto &anchor : anchors)
    {
        cout<<anchor<<endl;
    }
 }
