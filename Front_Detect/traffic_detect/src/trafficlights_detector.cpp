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
    pub_traffic_status = node.advertise<dashboard_msgs::Cmd>(m_traffic_status_topic,1);
    pub_traffic_active = node.advertise<dashboard_msgs::Proc>(m_traffic_active_topic,1);  
    pub_image_raw = node.advertise<sensor_msgs::Image>(m_image_raw_topic,1);

    return ret;
}

void TrafficLightsDetector::on_recv_frame(const sensor_msgs::Image& image_source)
{
    ROS_INFO("traffic_detect:image_callback!!!!!!!");

    cv_bridge::CvImagePtr cv_image = cv_bridge::toCvCopy(image_source, "bgr8");
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

bool TrafficLightsDetector::roi_region_is_valid()
{
    int src_w = m_frame.cols;
    int src_h = m_frame.rows;

    if( src_w < (m_x + m_w) )
    {
        ROS_INFO("Error: ROI x0 or x1 is out of image X-axes.");
        return false;
    }

    if( src_h < (m_y + m_h) )
    {
        ROS_INFO("Error: ROI y0 or y1 is out of image Y-axes.");
        return false;
    }

    return true;
}

bool TrafficLightsDetector::load_parameters()
{
    ros::NodeHandle private_nh("~");
    
    private_nh.param<std::string>("roi_region", m_roi_region, "0,0,640,480");
    ROS_INFO("Setting roi_region to %s", m_roi_region.c_str());
    
    std::vector<int> roi_vec =visplit(m_roi_region,",");
    if(roi_vec.size()<3)
    {
        ROS_INFO("Error: roi_region at least specified 3 integers");
        return false;
    }else if(roi_vec.size()<4)
    {
        roi_vec.push_back(roi_vec[2]);       
    }
    m_x = roi_vec[0];
    m_y = roi_vec[1];
    m_w = roi_vec[2];
    m_h = roi_vec[3];
    ROS_INFO("m_x=%d,m_y=%d,m_w=%d,m_h=%d",m_x,m_y,m_w,m_h);

    private_nh.param<std::string>("refresh_epoch",m_refresh_epoch,"10");
    ROS_INFO("Setting refresh_epoch to %s",m_refresh_epoch.c_str());

    private_nh.param<std::string>("image_source_topic", m_image_source_topic, "/usb_cam/image_raw");
    ROS_INFO("Setting image_source_topic to %s", m_image_source_topic.c_str());

    private_nh.param<std::string>("cam_cmd_topic", m_cam_cmd_topic, "/cam/cmd"); 
    ROS_INFO("Setting cam_cmd_topic to %s", m_cam_cmd_topic.c_str());    

    private_nh.param<std::string>("status_code_topic", m_status_code_topic, "/traffic/tls_code");
    ROS_INFO("Setting staus_code_topic to %s", m_status_code_topic.c_str());
    
    private_nh.param<std::string>("image_raw_topic", m_image_raw_topic, "/traffic/image_raw");
    ROS_INFO("Setting image_raw_topic to %s", m_image_raw_topic.c_str());

    private_nh.param<std::string>("traffic_status_topic", m_traffic_status_topic, "/traffic/status");
    ROS_INFO("Setting traffic_status_topic to %s", m_traffic_status_topic.c_str());
    
    private_nh.param<std::string>("traffic_active_topic", m_traffic_active_topic, "/traffic/active"); 
    ROS_INFO("Setting traffic_active_topic to %s", m_traffic_active_topic.c_str());   

    private_nh.param<bool>("simu_mode", m_simu_mode, false);
    ROS_INFO("Setting simu mode to %d", m_simu_mode);

    return true;
}


unsigned char TrafficLightsDetector::status_encode(bool go_up,bool go_left,bool go_right)
{
    unsigned char up = 0;
    unsigned char left = 0;
    unsigned char right = 0;
    unsigned char code;
    if (go_up)
        up = 1;
    if (go_left)
        left = 1;
    if (go_right)
        right = 1;
    
    code = 4 * up + 2 * left + 1 * right;
    return code;
}

unsigned char TrafficLightsDetector::status_encode_simu()
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

//处理收到的待检测帧
void TrafficLightsDetector::process_frame()
{
    m_lights_status_simu.clear();
    m_lights_status.clear();
    
    //m_frame_roi =  m_frame(cv::Rect(m_x,m_y,m_w,m_h));
    m_frame_roi =  m_frame;
    std::vector<BBoxInfo> boxes = m_yolo_helper.do_inference(m_frame_roi,m_simu_mode);

    cout<<"box_size="<<boxes.size()<<endl;
    if(boxes.size() == 0)
    {
        ROS_WARN("has not found box");
    }
    
    int box_idx = 0;
    for (BBoxInfo b : boxes)
    {
        //cout<<"box_idx:"<< box_idx++ << endl;
        cout<<"boundingbox:"<<b.box.x1<<","<<b.box.y1<<","<<b.box.x2<<","<<b.box.y2<<endl;
        //cout<<"label:"<< b.label<< endl;
        //cout<<"classId:"<< b.classId <<endl;
        //cout<<"prob:"<< b.prob <<endl;
        //cout<<"class_name:"<<m_yolo_helper.m_inferNet->getClassName(b.label)<<endl;

        bool go_up,go_left,go_right;//给出对当前frame的判断           
        //string class_name =  m_yolo_helper.m_inferNet->getClassName(b.label);

        string class_name = "";
        
        if(m_simu_mode)
        {
            if(class_name == "background")
            {
                cout<<"background"<<endl;

                string failed_img = "/home/nano/workspace_sc/failed/" + std::to_string(m_seq)+".png";
                m_seq++;
                
                cv::imwrite(failed_img,m_frame_roi);
                break;
            }
            else if(class_name == "red")
            {
                m_lights_status_simu.find_red = true;
                break;
            }
            else if(class_name == "yellow")
            {
                m_lights_status_simu.find_yellow = true;
                break;
            }
            else if(class_name == "green")
            {
                m_lights_status_simu.find_green = true;
                break;
            }
        }
        else
        {
            if(class_name == "stop")
            {
                m_lights_status.go = false;
            }
            else if(class_name == "go")
            {
                m_lights_status.go = true;
            }
            else if(class_name == "goLeft")
            {
                m_lights_status.goLeft = true;
            }
            else if(class_name == "stopLeft")
            {
                m_lights_status.goLeft = false;
            }
            else
            {
                ROS_INFO("not intrested class:%s",class_name.c_str());
            }
        }

    }


    //send marked img with boundingbox
    int imageIndex = 0;
/*    
    cv::Mat img = m_yolo_helper.get_marked_image(0);
    sensor_msgs::ImagePtr roi_msg =cv_bridge::CvImage(std_msgs::Header(), "bgr8", img).toImageMsg();
    roi_msg->header.seq = m_seq++;
    roi_msg->header.frame_id = "traffic image";
    roi_msg->header.stamp = ros::Time::now();
    pub_image_raw.publish(roi_msg);
*/
    // publish traffic status code
    std_msgs::UInt8 status_msg;
    if(m_simu_mode)
    {
        status_msg.data = status_encode_simu();
    }
    else
    {
        status_msg.data = status_encode(m_lights_status.go,m_lights_status.goLeft,m_lights_status.goRight);
    }
    
    pub_status_code.publish(status_msg); 
}
