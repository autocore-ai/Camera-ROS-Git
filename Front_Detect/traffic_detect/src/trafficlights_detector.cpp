#include "trafficlights_detector.h"
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

TrafficLightsDetector::TrafficLightsDetector()
{

}

TrafficLightsDetector::~TrafficLightsDetector()
{

}


void TrafficLightsDetector::set_current_frame(cv::Mat frame)
{
    m_frame = frame;
}

void TrafficLightsDetector::init(int argc,char** argv)
{
    //load yolov3 model param
    //m_yolo_helper.parse_config_params(argc,argv);
    //
    m_yolo_helper.init(argv[1]);
    init_ros(argc,argv);
} 

bool TrafficLightsDetector::init_ros(int argc,char** argv)
{
    ros::init(argc,argv,"yolov3");
    ros::NodeHandle node;

    bool ret = load_parameters();
    
    sub_image_raw = node.subscribe(m_image_source_topic, 1, &TrafficLightsDetector::on_recv_frame,this);
    //sub_cam_cmd = node.subscribe(m_cam_cmd_topic,1,cmd_callback);
    pub_status_code = node.advertise<std_msgs::UInt8>(m_status_code_topic,1);

    return ret;
}

void TrafficLightsDetector::on_recv_frame(const sensor_msgs::Image& image_source)
{
    ROS_INFO("traffic_detect:image_callback!!!!!!!");

    cv_bridge::CvImagePtr cv_image = cv_bridge::toCvCopy(image_source, "bgr8");

    //这里不能直接用m_frame = cv_image->image;
    cv::Mat frame = cv_image->image; 
    set_current_frame(frame);
    
    process_frame();
}

std::vector<int> TrafficLightsDetector::visplit(std::string str,std::string pattern)
{
    std::string::size_type pos;
    std::vector<int> result;
    str += pattern;
    unsigned int size = str.size();
    for (unsigned int i = 0; i < size; i++)
    {
        pos = str.find(pattern, i);
        if (pos < size)
        {
            std::string s = str.substr(i, pos - i);
            result.push_back(atoi(s.c_str()));
            i = pos + pattern.size() - 1;
        }
    }
    return result;
}

bool TrafficLightsDetector::load_parameters()
{
    ros::NodeHandle private_nh("~");
    
    private_nh.param<std::string>("image_source_topic", m_image_source_topic, "/usb_cam/image_raw");
    ROS_INFO("Setting image_source_topic to %s", m_image_source_topic.c_str());

    private_nh.param<std::string>("status_code_topic", m_status_code_topic, "/traffic/tls_code");
    ROS_INFO("Setting staus_code_topic to %s", m_status_code_topic.c_str());
     
    private_nh.param<bool>("simu_mode", m_simu_mode, false);
    ROS_INFO("Setting simu mode to %d", m_simu_mode);

    return true;
}

unsigned char TrafficLightsDetector::status_encode()
{
    //3 bit means go forward,left,right,like 100 means go forward,000 means can not go forward
    unsigned char code = 4;
    int red = m_lights_status_simu.find_red?1:0;
    int yellow = m_lights_status_simu.find_yellow?1:0;
    int green = m_lights_status_simu.find_green?1:0;

    printf("red=%d,yellow=%d,green=%d\n",red,yellow,green);
    
    if(red)
    {
        code = 0;
    }
    else if(green || yellow)
    {
        code = 6;
    }
    else 
    {
        code = 7;
    }
    
    cout<<"code="<<(int)code<<endl;
    
    return code;
    
}

void TrafficLightsDetector::preprocess_frame()
{
    int width = m_yolo_helper.get_width();
    int height = m_yolo_helper.get_height();
    cv::resize(m_frame,m_frame_model_input,cv::Size(width,height));

    cv::imshow("model input", m_frame_model_input);
    cv::waitKey(30);

    ROS_INFO("resize img to %d x %d", width,height);
}

//处理收到的待检测帧
void TrafficLightsDetector::process_frame()
{
    m_lights_status_simu.clear();
    
    preprocess_frame();
    std::vector<BBoxInfo> boxes = m_yolo_helper.do_inference(m_frame_model_input,m_simu_mode);

    cout<<"box_size="<<boxes.size()<<endl;
    if(boxes.size() == 0)
    {
        ROS_WARN("has not found box");
    }
    
    int box_idx = 0;
    for (BBoxInfo b : boxes)
    {
        //cout<<"boundingbox:"<<b.box.x1<<","<<b.box.y1<<","<<b.box.x2<<","<<b.box.y2<<endl;
        //cout<<"label:"<< b.label<< endl;
        cout<<"classId:"<< b.classId <<endl;

        //红绿灯4class red/yellow/green/background
        int classid = b.classId;
        if(classid == 3)
        {
            cout<<"background"<<endl;
        }
        else if(classid == 0)
        {
            m_lights_status_simu.find_red = true;
            break;
        }
        else if(classid == 1)
        {
            m_lights_status_simu.find_yellow = true;
            break;
        }
        else if(classid == 2)
        {
            m_lights_status_simu.find_green = true;
            break;
        }
    }

    //send marked img with boundingbox
    int imageIndex = 0;

    // publish traffic status code
    std_msgs::UInt8 status_msg;
    status_msg.data = status_encode();
    pub_status_code.publish(status_msg); 
}
