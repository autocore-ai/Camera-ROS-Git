#include "vps_detector.h"
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
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
using namespace cv;

#define MYDEBUG ROS_INFO("%s-%d",__FUNCTION__,__LINE__);
#define LOG_FUNC_BEGIN ROS_INFO("%s begin",__FUNCTION__);
#define LOG_FUNC_END ROS_INFO("%s end",__FUNCTION__);


VpsDetector::VpsDetector()
{

}

VpsDetector::~VpsDetector()
{

}

void VpsDetector::init(int argc,char** argv)
{
    //load yolov3 model param
    m_yolo_helper.parse_config_params(argc,argv);
    //
    init_ros(argc,argv);

    /*
    unsigned int init_clock = 10; 
    unsigned int clock_thresh = 3;
    float center_thresh = 20.0;
    float iou_thresh = 0.15;
    unsigned int iou_level = 1;
    unsigned int send_epoch = 10;
    m_park_tracker = new ATCParkTracker(init_clock, m_delta_x, m_delta_y, clock_thresh, center_thresh, iou_thresh, iou_level, send_epoch);
    */
    m_p_carpark_mgr = new CarParkMgr;
    m_p_atc_mapper = new ATCMapper;
    m_p_park = new ParkInfo;


    m_pose.pose.position.x = 1.0;
    m_pose.pose.position.y = 1.0;
    m_pose.pose.position.z = 1.0;
    m_pose.pose.orientation.x = 2.0;
    m_pose.pose.orientation.y = 3.0;
    m_pose.pose.orientation.z = 4.0;
    m_pose.pose.orientation.w = 1.0;
} 

void VpsDetector::init_ros(int argc,char **argv)
{
    ros::init(argc,argv,"yolo");
    ros::NodeHandle node;
    ros::NodeHandle private_nh("~");

    private_nh.param<std::string>("image_source_topic", m_sub_topic_image_raw_from_camera, "/usb_cam/image_raw");
    ROS_INFO("Setting image_source_topic to %s",m_sub_topic_image_raw_from_camera.c_str());
    m_sub_image_raw = node.subscribe(m_sub_topic_image_raw_from_camera, 1, &VpsDetector::on_recv_frame,this);

    private_nh.param<std::string>("image_pose_topic",m_image_pose_topic,"/gnss_pose");
    ROS_INFO("Setting image_pose_topic to %s",m_image_pose_topic.c_str());
    m_sub_image_pose = node.subscribe(m_image_pose_topic, 1, &VpsDetector::on_pose_msg,this);

    private_nh.param<std::string>("image_object_topic", m_image_object_topic, "/vps/park_obj");
    ROS_INFO("Setting image_object_topic to %s", m_image_object_topic.c_str());
    m_pub_image_obj = node.advertise<autoreg_msgs::park_obj>(m_image_object_topic, 1);

    private_nh.param<std::string>("image_raw_topic", m_image_raw_topic, "/vps/image_raw");
    ROS_INFO("Setting image_raw_topic to %s", m_image_raw_topic.c_str());
    m_pub_image_raw = node.advertise<sensor_msgs::Image>(m_image_raw_topic, 1);

    private_nh.param<std::string>("vps_active_topic", m_vps_active_topic, "/vps/active");
    ROS_INFO("Setting vps_active_topic to %s", m_vps_active_topic.c_str());
    m_pub_vps_active = node.advertise<dashboard_msgs::Proc>(m_vps_active_topic, 1);

    private_nh.param<bool>("test_mode", m_test, false);
    ROS_INFO("Setting test mode to %d", m_test);

    private_nh.param<int>("frame_counts_divide", m_frame_counts_divide, 60);
    ROS_INFO("Setting frame_counts_divide to %d", m_frame_counts_divide);  

    private_nh.param<float>("bbox_expand_ratio", m_bbox_expand_ratio, 0.1);
    ROS_INFO("Setting bbox_expand_ratio to %f", m_bbox_expand_ratio); 

    private_nh.param<float>("min_lw_ratio",m_min_lw_ratio , 1.2);
    ROS_INFO("Setting m_min_lw_ratio to %f", m_min_lw_ratio); 

    private_nh.param<float>("max_lw_ratio", m_max_lw_ratio, 5.5);
    ROS_INFO("Setting m_max_lw_ratio to %f", m_max_lw_ratio); 

}

//lines的坐标是在roi_img这张图内的坐标.
void VpsDetector::get_lines(const cv::Mat& roi_img,std::vector<cv::Vec4i>& lines,int box_idx)
{
      LOG_FUNC_BEGIN
        
      cv::Mat tmp_img;
      if(park_edge_detect(roi_img, tmp_img))
      {
         cv::HoughLinesP(tmp_img, lines, 1, CV_PI / 180, 20, 20, 10);
      }

      //cv::imshow("lines",tmp_img);
      //cv::waitKey(0);

      std::ostringstream img_name;
      img_name << "expandbbox_edge" << box_idx<<".jpg";
      box_idx += 1;
      save_img(img_name.str(), tmp_img);

      LOG_FUNC_END
}

// input park patch image,
bool VpsDetector::park_edge_detect(cv::Mat src_img, cv::Mat &dst_img)
{
    cv::Mat mid_img, edge_img, depress_img, mask;
    cv::cvtColor(src_img, mid_img, CV_BGR2GRAY);

    //image enhance
    // cv::equalizeHist(mid_img,mid_img);
    //cv::imshow("gray",mid_img);

    // canny operator
    //std::cout<<"try median filter!"<<std::endl;
    cv::Canny(mid_img, edge_img, 50, 200, 3);
    //cv::imshow("edge_img",edge_img);
    
    bool ret = depress_fringe_grad(edge_img, depress_img, 5);
    if (!ret)
        return ret;
    //get binary mask
    cv::equalizeHist(mid_img, mid_img);
    cv::threshold(mid_img, mask, 180, 255, CV_THRESH_BINARY);
    cv::Mat element = cv::getStructuringElement(MORPH_RECT, cv::Size(3, 3));
    // cv::erode(mask, mask, element);
    cv::dilate(mask, mask, element);
    cv::threshold(mask, mask, 180, 255, CV_THRESH_BINARY);

    //cv::imshow("bin_img",mask);
    //std::cout<<"width:"<<depress_img.cols<<" height:"<<depress_img.rows<<std::endl;
    //std::cout<<"height:"<<mask.cols<<" height:"<<mask.rows<<std::endl;
    depress_img.copyTo(dst_img, mask);

    //std::cout<<"out park edge detect\n";
    //cv::imshow("raw_edge",edge_img);
    //cv::imshow("dst_edge",dst_img);
    //cv::waitKey(100);

    return ret;
}

bool VpsDetector::depress_fringe_grad(cv::Mat src_img, cv::Mat &dst_img, int shrink)
{
    cv::Mat mid_img, mask;
    int dst_w, dst_h;

    //
    mask = cv::Mat::zeros(src_img.size(), src_img.type());
    dst_w = src_img.cols - 2 * shrink;
    dst_h = src_img.rows - 2 * shrink;
    if (shrink < 0)
    {
        //src_img.copyTo(dst_img,mask);
        return false;
    }
    if (dst_w < 1 || dst_h < 1) // bad shrink
    {
        //std::cout<<"Warnning: bad image shrink,please decrease shrink offset\n";
        //dst_img = mask;
        //src_img.copyTo(dst_img,mask);
        //std::cout<<"cout--------\n";
        return false;
    }

    mask(cv::Rect(shrink - 1, shrink - 1, dst_w, dst_h)).setTo(255);
    //std::cout<<"shrink:"<<shrink-1<<" dst_w:"<<dst_w<<" dst_h:"<<dst_h<<endl;
    //imshow("mask",mask);

    src_img.copyTo(dst_img, mask);
    //std::cout<<"out offf drepressed \n";
    return true;
}

//--map park anchor from expand bnding box to raw image--
//Coordinate adjustment. 
void VpsDetector::adjust_coordination(const cv::Rect& expand_rect,ParkAnchor& draw_anchor)
{
    //roi_img的(0,0)在原始图像frame_input中的坐标
    long ltop_x = expand_rect.x;
    long ltop_y = expand_rect.y;

    draw_anchor.ltop.x += ltop_x;
    draw_anchor.ltop.y += ltop_y;
    
    draw_anchor.rtop.x += ltop_x;
    draw_anchor.rtop.y += ltop_y;

    draw_anchor.rbot.x += ltop_x;
    draw_anchor.rbot.y += ltop_y;

    draw_anchor.lbot.x += ltop_x;
    draw_anchor.lbot.y += ltop_y;

    return;
}

//长边沿x轴方向
void VpsDetector::process_carpark1(const cv::Point& begin,
                                          const cv::Point& end,
                                          const cv::Rect& expand_rect,
                                          ParkAnchor& draw_anchor)
{
    ROS_INFO("process_carpark1 begin"); 

    
    
    cv::Point center(expand_rect.width / 2, expand_rect.height / 2);
    if (begin.x < center.x)  //long line at right side
    {
        draw_anchor.ltop = cv::Point(begin.x, 0);
        draw_anchor.rtop = cv::Point(expand_rect.width - 1, 0);
        draw_anchor.rbot = cv::Point(expand_rect.width - 1, expand_rect.height - 1);
        draw_anchor.lbot = cv::Point(begin.x, expand_rect.height - 1);
    }
    else   //long line at left side
    {
        draw_anchor.ltop = cv::Point(0, 0);
        draw_anchor.rtop = cv::Point(begin.x, 0);
        draw_anchor.rbot = cv::Point(begin.x, expand_rect.height - 1);
        draw_anchor.lbot = cv::Point(0, expand_rect.height - 1);
    }

    adjust_coordination(expand_rect,draw_anchor);

    ROS_INFO("process_carpark1 end");
}

//长边沿y轴方向
 void VpsDetector::process_carpark2(const cv::Point& begin,
                                           const cv::Point& end,
                                           const cv::Rect& expand_rect,
                                           ParkAnchor& draw_anchor)
 {
     ROS_INFO("process_carpark2 begin"); 
 
     cv::Point center(expand_rect.width / 2, expand_rect.height / 2);
     
     if (begin.y < center.y) 
     {
         draw_anchor.ltop = cv::Point(0, begin.y);
         draw_anchor.rtop = cv::Point(expand_rect.width - 1, begin.y);
         draw_anchor.rbot = cv::Point(expand_rect.width - 1, expand_rect.height - 1);
         draw_anchor.lbot = cv::Point(0, expand_rect.height - 1);
     }
     else 
     {
         draw_anchor.ltop = cv::Point(0, 0);
         draw_anchor.rtop = cv::Point(expand_rect.width - 1, 0);
         draw_anchor.rbot = cv::Point(expand_rect.width - 1, begin.y);
         draw_anchor.lbot = cv::Point(0, begin.y);
     }

     adjust_coordination(expand_rect,draw_anchor);

     ROS_INFO("process_carpark2 end"); 
 }

//长边不沿x轴,y轴方向.
bool VpsDetector::process_carpark3(const cv::Point& begin,
                                        const cv::Point& end,
                               const cv::Rect& expand_rect,ParkAnchor& draw_anchor)
{
      ROS_INFO("process_carpark3 begin");  
      bool valid_cakpark = true;

      //ax+by+c=0
      float b = 1.0;
      float a = -(end.y - begin.y) * 1.0 / (end.x - begin.x);
      float c = -a * begin.x - begin.y;

      std::vector<cv::Point> line_crossover;
      get_two_crossover_from_line_to_rect(a, b, c, expand_rect, line_crossover);
      if (2 != line_crossover.size())
      {
          cout<<"this should not happen,crosspoints not 2!"<<endl;
          valid_cakpark = false;

          return valid_cakpark;
      }

      /*catch perpendicular line based on two crossovers*/
      std::vector<cv::Point> rect_crossover;
      for (unsigned int idx = 0; idx < line_crossover.size(); idx++)
      {
          cv::Point crossover_point = line_crossover[idx];
          float p_a, p_b, p_c; //p_a*x+p_b*y+p_c=0
          p_a = -1 / a;
          p_b = 1.0;
          p_c = -(crossover_point.y) - p_a * (crossover_point.x);
  
          std::vector<cv::Point> next_crossover;
          next_crossover.push_back(crossover_point);
          get_two_crossover_from_line_to_rect(p_a, p_b, p_c, expand_rect, next_crossover);
          if (2 != next_crossover.size())
          {
              std::cout<<"Warnning:can not get two crossover vs("<<next_crossover.size()<<" in perpendicular line \n";
              valid_cakpark = false;
              continue;
          }
          for (unsigned int pline_idx = 0; pline_idx < next_crossover.size(); pline_idx++)
          {
              rect_crossover.push_back(next_crossover[pline_idx]);
          }
      }

      /*-- translate 4 crossovers into ParkAnchor*/
      if (4 == rect_crossover.size())
      {
          get_park_anchor(rect_crossover, draw_anchor, 1);
      }
      else
      {
          valid_cakpark = false;
      }
         
      adjust_coordination(expand_rect,draw_anchor);

      ROS_INFO("process_carpark3 end");
      
      return valid_cakpark;
}

//对应3种车位类型,正常倒库的长方形,路边侧方位倒进去的长方形,斜着的车位(平行四边形那种)
//只知道line是长边,但是并不知道是哪一条长边
bool VpsDetector::process_carpark(const cv::Vec4i& line,const cv::Rect& expand_rect,ParkAnchor& draw_anchor)
{
    ROS_INFO("%s begin",__FUNCTION__);
    
    bool valid = true;

    cv::Point begin(line[0], line[1]);
    cv::Point end(line[2], line[3]); // ax+by+c =0;

    ROS_INFO("long line:begin(%d,%d),end(%d,%d)",line[0],line[1],line[2],line[3]);
    
    if (begin.x == end.x) //vectical line
    {
        process_carpark1(begin,end,expand_rect,draw_anchor);
    }
    else if (begin.y == end.y) // horizonal line
    {
        process_carpark2(begin,end,expand_rect,draw_anchor);
    }
    else
    {
       valid = process_carpark3(begin,end,expand_rect,draw_anchor);
    }

    ROS_INFO("%s end",__FUNCTION__);
    return valid;
}

void VpsDetector::save_img(const string& img_name,const cv::Mat& img)
{  
    cv::imwrite(m_save_dir + img_name,img);
}

void VpsDetector::process_bbox(const std::vector<BBoxInfo>& free_park_boxes)
{
    cv::Rect expand_rect;
    int input_frame_h = m_frame_input.size().height;
    int input_frame_w = m_frame_input.size().width;
    //ROS_INFO("input_frame_h=%d,input_frame_w=%d",input_frame_h,input_frame_w);

    int box_idx = 0;
    for(BBoxInfo box_info : free_park_boxes)
    {
          //拿到boundingbox,并不意味着拿到了park的坐标.下面要做的就是从boundingbox范围内的图内把park识别出来.
          BBox b = box_info.box;
          /*--enlarge bndingbox with o.1*width pixels--*/
          expand_bndbox(cv::Rect(b.x1, b.y1, (b.x2 - b.x1), (b.y2 - b.y1)), expand_rect,m_bbox_expand_ratio, input_frame_w, input_frame_h);

          cv::Mat roi_img = m_frame_input(expand_rect);  //抠出当前的detected objet

          //just for test
          //cv::imshow("fuck",);
          //cv::waitKey(0);

          std::ostringstream img_name;
          img_name << "expandbox" << box_idx<<".jpg";
          
          save_img(img_name.str(), m_yolo_helper.get_marked_image(0)(expand_rect));
                    
          /*--two stage park line detection--*/
          std::vector<cv::Vec4i> lines;
          get_lines(roi_img,lines,box_idx);
          box_idx += 1;
          if (lines.size() < 1)
          {
              std::cout<<"warnning can not detect lines\n";
              continue;
          }   

          cout<<"lines size:"<<lines.size()<<endl;
          
          //cv::imshow("roi_img_with_line", roi_img); 
          //cv::waitKey(0);
          
          //找出车位的长边
          int line_idx = 0;
          int max_line = 0;
          for (unsigned int j = 0; j < lines.size(); j++)
          {
              cv::Vec4i l = lines[j];
              int line_len = (l[2] - l[0]) * (l[2] - l[0]) + (l[3] - l[1]) * (l[3] - l[1]);
              if (line_len > max_line)
              {
                  max_line = line_len;
                  line_idx = j;
              }
          }

          cv::Vec4i long_line = lines[line_idx];
          ParkAnchor draw_anchor;
          bool is_valid = process_carpark(long_line,expand_rect,draw_anchor);
          //ROS_INFO("park_anchor=%s",draw_anchor.to_string().c_str());

          if (!is_valid)
          {
                std::cout<<"waring unvalid draw_anchor"<<std::endl;
                continue;
          }
          cv::Point center(expand_rect.width / 2, expand_rect.height / 2);
          if (!park_anchor_filter(draw_anchor, center, m_thresold, m_max_lw_ratio, m_min_lw_ratio))
          {
                std::cout<<"unormal park anchor"<<std::endl;
                continue;
          }

          process_curr_park(draw_anchor,box_info);  

          //break;//for test
    }
}

//将车位坐标告诉下游程序
void VpsDetector::pub_parkobj_msg()
{
    std::vector<const ParkInfo *> park_info;
    m_p_carpark_mgr->get_effective_parks(park_info);
    size_t effective_nums = park_info.size();
    if(effective_nums > 0)
    {
        m_parkobj_msg.header.frame_id = "park obj";
        m_parkobj_msg.type = "park_obj";
        m_parkobj_msg.obj.clear();
        for(auto p : park_info)
        {
            autoreg_msgs::park_anchor park_obj;

            park_obj.x0 = p->points_in_car[0];
            park_obj.y0 = p->points_in_car[1];
            park_obj.x1 = p->points_in_car[2];
            park_obj.y1 = p->points_in_car[3];
            park_obj.x2 = p->points_in_car[4];
            park_obj.y2 = p->points_in_car[5];
            park_obj.x3 = p->points_in_car[6];
            park_obj.y3 = p->points_in_car[7];
            
            park_obj.id = p->id;
            
            cout<<"park as below"<<endl;
            cout<<park_obj.x0<<","<<park_obj.y0<<endl;
            cout<<park_obj.x1<<","<<park_obj.y1<<endl;
            cout<<park_obj.x2<<","<<park_obj.y2<<endl;
            cout<<park_obj.x3<<","<<park_obj.y3<<endl;
            m_parkobj_msg.obj.push_back(park_obj);
        }

        m_pub_image_obj.publish(m_parkobj_msg);
    }
}

//
void VpsDetector::process_curr_park(const ParkAnchor& draw_anchor,const BBoxInfo& box_info)
{
    ROS_INFO("process_curr_park begin");

    m_p_park->points_in_img[0] = draw_anchor.ltop.x;
    m_p_park->points_in_img[1] = draw_anchor.ltop.y;
    m_p_park->points_in_img[2] = draw_anchor.rtop.x;
    m_p_park->points_in_img[3] = draw_anchor.rtop.y;
    m_p_park->points_in_img[4] = draw_anchor.rbot.x;
    m_p_park->points_in_img[5] = draw_anchor.rbot.y;
    m_p_park->points_in_img[6] = draw_anchor.lbot.x;
    m_p_park->points_in_img[7] = draw_anchor.lbot.y;
    
    //m_p_park->prob = box_info.prob;
    //m_p_park->id = 0;
    //m_p_park->cls_id = box_info.classId;

    //计算地球坐标系下的坐标
    m_p_atc_mapper->convert_to_vecmap(m_p_park);
        
    m_p_carpark_mgr->add_carpark(m_p_park);

    ROS_INFO("process_curr_park end");
}

//更新车身姿态信息,模型输入图片尺寸信息,以便后续不同坐标系内的坐标转换
void VpsDetector::update_carpose()
{
    //only part of the img contains park.because we handle img from 810*1080-->1080*1080-->300*300
    //810*1080-->1080*1080 we prefill (127,127,127)
    int img_w = m_frame_input.cols;
    int img_h = m_frame_input.rows;

    float effective_w = img_w * (810./1080.);
    
    float dx = m_delta_x * (1080.0/300.0);
    float dy = m_delta_y * (1080.0/300.0);

    m_p_atc_mapper->update(dx,dy,effective_w,img_h,m_pose.pose.position,m_pose.pose.orientation);
}

void VpsDetector::process_frame()
{  
    ROS_INFO("process_frame begin!!!!!!!!!!!!!!!!!!!!!!!!!");
    
    //调整图像尺寸
    img_decode(m_frame, m_frame_input, m_delta_x, m_delta_y);
 
    //adjust img size feed to model
    //cv::imshow("before",m_frame_input);
    //save_img("fuck", m_frame_input);
    imrotate(m_frame_input,m_frame_input,0);

    //trick
    cv::resize(m_frame_input, m_frame_input, cv::Size(300, 300));    
    //cv::imshow("after",m_frame_input);
    //cv::waitKey(0);

    //just for test
    //m_frame_input = cv::imread("/home/nano/suchang/frame1203.jpeg", CV_LOAD_IMAGE_COLOR);
    //cv::imshow("input",m_frame_input);
    //cv::waitKey(100);
    
    //更新车身姿态信息,模型输入图片尺寸信息,以便后续不同坐标系内的坐标转换
    update_carpose();

    //imrotate(src_frame, frame_input,0);

    //推理
    //{"park0"}
    std::vector<BBoxInfo> free_park_boxes;
    std::vector<BBoxInfo> non_free_park_boxes;
    std::vector<BBoxInfo> boxes_info = m_yolo_helper.do_inference(m_frame_input);

    //
    get_nonfree_parks(boxes_info,non_free_park_boxes);

    //
    get_free_park_boxes(boxes_info,non_free_park_boxes,free_park_boxes);
    process_bbox(free_park_boxes);

    m_p_carpark_mgr->record_parkinfo_in_this_frame();//记得处理完每一帧图像保存信息

    pub_parkobj_msg();
    
    cv::Mat frame_show = m_frame_input.clone();
    bool eff = draw_park_on_img(frame_show);

    if(eff)
    {
        pub_img(frame_show);
        save_img("sucess.jpg",m_frame_input);
        save_img("sucess_detected.jpg",frame_show);
    }
    else
    {
        save_img("fail.jpg",m_frame_input);
    }
    
    //cv::imshow("vps_show", frame_show);
    //cv::waitKey(100); //https://stackoverflow.com/questions/5217519/what-does-opencvs-cvwaitkey-function-do

    ROS_INFO("process_frame end!!!!!!!!!!!!!!!!!!!!!!!!!");
}

void VpsDetector::on_recv_frame(const sensor_msgs::Image &image_source)
{
      ROS_INFO("VpsDetector:on_recv_frame begin!!!!!!!!!!!!!!!!!!!!!!!!!");
      
      //cout<<"fuck!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"<<endl;
      m_frame_counts += 1;
      if(m_frame_counts / m_frame_counts_divide > 100000)
      {
            m_frame_counts = 0;
      }
      if(m_frame_counts % m_frame_counts_divide != 0 && m_frame_counts != 1)
      {
            //cout<<m_frame_counts<<','<<m_frame_counts_divide<<endl;

            //ROS_INFO("return,m_frame_counts=%ld,m_frame_counts_divide=%d",m_frame_counts,m_frame_counts_divide);
            return;
      }
 
      m_parkobj_msg.header = image_source.header;
      
      cv_bridge::CvImagePtr cv_image = cv_bridge::toCvCopy(image_source, "bgr8"); 
      m_frame = cv_image->image;

      process_frame();
      
      ROS_INFO("VpsDetector:on_recv_frame end!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
}


void VpsDetector::test()
{
    cout<<"test begin"<<endl;
    
    m_pose.pose.position.x = 1.0;
    m_pose.pose.position.y = 1.0;
    m_pose.pose.position.z = 1.0;
    m_pose.pose.orientation.x = 2.0;
    m_pose.pose.orientation.y = 3.0;
    m_pose.pose.orientation.z = 4.0;
    m_pose.pose.orientation.w = 1.0;
     
    //for(int i =0;i<4;i++)
    {
        std::ostringstream img_name;
        //img_name << "/home/nano/Downloads/test_img/park" << 0<<".jpg";
        img_name << "/home/nano/suchang/fuck_raw.jpg";
        
        cv::Mat test_img = cv::imread(img_name.str(), CV_LOAD_IMAGE_COLOR);
        if(test_img.empty())
        {
            return;
        }
         
        m_frame = test_img;

        process_frame(); 
    }

    cout<<"test end"<<endl;
}

void VpsDetector::get_nonfree_parks(const std::vector<BBoxInfo>& boxes_info,std::vector<BBoxInfo>& non_free_park_boxes)
{
      int box_idx = 0;
      for (BBoxInfo b : boxes_info)
      {          
          cout<<"box_idx:"<< box_idx++ << endl;
          cout<<"boundingbox:"<<b.box.x1<<","<<b.box.y1<<","<<b.box.x2<<","<<b.box.y2<<endl;
          cout<<"label:"<< b.label<< endl;
          cout<<"classId:"<< b.classId <<endl;
          cout<<"prob:"<< b.prob <<endl;
          cout<<"class_name:"<<m_yolo_helper.m_inferNet->getClassName(b.label)<<endl; 
        
          string class_name =  m_yolo_helper.m_inferNet->getClassName(b.label);
          
          //模型暂时不支持检测forbid_park/incar_park  逻辑先保留
          if(class_name == "forbid_park" || class_name == "incar_park")
          {
              non_free_park_boxes.push_back(b);
          }
      }

      return;
}

//初步获取有效的freepark的boundingbox
void VpsDetector::get_free_park_boxes(const std::vector<BBoxInfo>& boxes_info,const std::vector<BBoxInfo>& non_free_park_boxes,std::vector<BBoxInfo>& free_park_boxes)
{
      for (BBoxInfo b : boxes_info)
      {
          string class_name =  m_yolo_helper.m_inferNet->getClassName(b.label);
          if(class_name == "park0")
          {
              if(is_effective_park(non_free_park_boxes, b.box, 0.6))
              {
                  ROS_INFO("%s-%d-find free parks",__FUNCTION__,__LINE__);
                  free_park_boxes.push_back(b);
              }
          }
      }
}

//在图片上绘制出车位框 
bool VpsDetector::draw_park_on_img(cv::Mat &img)
{
    bool effctive = false;
    
    vector<const ParkInfo* > park_info;
    m_p_carpark_mgr->get_effective_parks(park_info);

    for(auto p : park_info)
    {
        //ROS_INFO("find effetive park!");

        effctive = true;
        
        cv::Point point1(p->points_in_img[0],p->points_in_img[1]);
        cv::Point point2(p->points_in_img[2],p->points_in_img[3]);
        cv::Point point3(p->points_in_img[4],p->points_in_img[5]);
        cv::Point point4(p->points_in_img[6],p->points_in_img[7]);

        cv::line(img,point1,point2,cv::Scalar(0,255,0),2,CV_AA);
        cv::line(img,point2,point3,cv::Scalar(0,255,0),2,CV_AA);
        cv::line(img,point3,point4,cv::Scalar(0,255,0),2,CV_AA);
        cv::line(img,point4,point1,cv::Scalar(0,255,0),2,CV_AA);

        std::string label = std::to_string(p->id);
        int base_line=1;        
        cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &base_line);
        cv::rectangle(img, cv::Rect(cv::Point(p->points_in_img[0],p->points_in_img[1]- label_size.height),cv::Size(label_size.width, label_size.height + base_line)),cv::Scalar(255,255,0),CV_FILLED);
        cv::putText(img,label,point1,cv::FONT_HERSHEY_SIMPLEX,0.8,cv::Scalar(0,0,0));
    }

    return effctive;
}


void VpsDetector::pub_img(const cv::Mat& detect_show)
{
      LOG_FUNC_BEGIN

      std::ostringstream img_name;
      img_name << "frame" << m_seq<<".jpg";
      //save_img(img_name.str(), detect_show);
      
      //cout<<"pub_img:"<<m_seq<<endl;
      cv_bridge::CvImage  cv_img(std_msgs::Header(), "bgr8", detect_show);
      sensor_msgs::ImagePtr roi_msg = cv_img.toImageMsg();
      roi_msg->header.seq = m_seq++;
      roi_msg->header.frame_id = "vps image";
      roi_msg->header.stamp = ros::Time::now();
      m_pub_image_raw.publish(roi_msg);
      
      
      LOG_FUNC_END
}

//get iou ratio between two bndingbox
float VpsDetector::get_box_overlap_ratio(const BBox& bbox1,const BBox& bbox2)
{
    float x_min,y_min,x_max,y_max;
    float area_bbox1,area_bbox2,area_intersect;
    float iou=0.0;
    if (bbox2.x1 > bbox1.x2 || bbox2.x2 < bbox1.x1 || bbox2.y1 > bbox1.y2 || bbox2.y2 < bbox1.y1) 
        return iou;

    x_min = std::max(bbox1.x1, bbox2.x1);
    y_min = std::max(bbox1.y1, bbox2.y1);
    x_max = std::min(bbox1.x2, bbox2.x2);   
    y_max = std::min(bbox1.y2, bbox2.y2);
    
    //
    area_bbox1 =(bbox1.x2 - bbox1.x1)*(bbox1.y2 - bbox1.y1);
    area_bbox2 =(bbox2.x2 - bbox2.x1)*(bbox2.y2 - bbox2.y1);
    area_intersect = (x_max - x_min)*(y_max -y_min);
    iou = area_intersect/(area_bbox1 + area_bbox2 - area_intersect);

    return iou;
}

//通过iou判断车位是否有效 与任一无效车位iou超过阈值,则认为是无效车位
bool VpsDetector::is_effective_park(const std::vector<BBoxInfo>& notfreebox_list,const BBox& freepark_box,float iou_thresold)
{
    bool effective = true;
    if(iou_thresold<0|| iou_thresold >1)
        iou_thresold = 0.5;//default iou

    float iou;
    for(unsigned int idx =0;idx<notfreebox_list.size();++idx)
    {
        BBoxInfo box_info = notfreebox_list[idx];

        string class_name =  m_yolo_helper.m_inferNet->getClassName(box_info.label);
        
        if(class_name != "park0") // forbidden_park or incar_park
        {
            iou =  get_box_overlap_ratio(box_info.box,freepark_box);
            if(iou>iou_thresold)
            {
                effective = false;
                break;
            }
        }
    }

    return effective; 
}

//图像裁剪
void VpsDetector::img_decode(cv::Mat raw_data, cv::Mat &dst_img,float &dst_delta_x,float &dst_delta_y)
{
    int raw_w = raw_data.cols;
    int raw_h = raw_data.rows;
    std::cout<<"raw_w:"<<raw_w<<"raw_h:"<<raw_h<<endl;
    cout<<"raw_w:"<<raw_w<<"raw_h:"<<raw_h<<endl;
    cout<<"m_yulan_w"<<m_yulan_w<<"m_yulan_h"<<endl;
    save_img("fuck_raw.jpg", raw_data);
    // get raw_data from yulan device
    if((m_yulan_w<=raw_w)&&(m_yulan_h<=raw_h))
    {
        cout<<"cut img"<<endl;
        dst_img=raw_data(cv::Rect(0,0,m_yulan_w,m_yulan_h));
        dst_delta_x = m_delta_x;
        dst_delta_y = m_delta_y;
    }

    save_img("fuck_cut.jpg",dst_img);
    cout<<"dst_w"<<dst_img.cols<<"dst_h"<<dst_img.rows<<endl;
    
    //cv::imshow("raw_img",dst_img);
    //cv::waitKey(0);
}

 //把图像事先调整成x*x(比如1080*960--->1080*1080)会提高性能?
 void VpsDetector::imrotate(cv::Mat& img,cv::Mat& newIm, double angle)
 {
     //better performance 
     std::size_t rot_size = std::max(img.rows,img.cols);

     //指定旋转中心
     cv::Point2f pt(img.cols/2.,img.rows/2.);

     //获取旋转矩阵（2x3矩阵）
     cv::Mat r = cv::getRotationMatrix2D(pt,angle,1.0);

     //根据旋转矩阵进行仿射变换
     warpAffine(img,newIm,r,cv::Size(rot_size,rot_size),cv::INTER_LINEAR,cv::BORDER_CONSTANT,cv::Scalar(127,127,127));
 }

// get two line's crosspoint
cv::Point2f VpsDetector::getCrossPoint(cv::Vec4i LineA, cv::Vec4i LineB)
{
    double ka, kb;
    ka = (double)(LineA[3] - LineA[1]) / (double)(LineA[2] - LineA[0]); 
    kb = (double)(LineB[3] - LineB[1]) / (double)(LineB[2] - LineB[0]); 

    cv::Point2f crossPoint;
    crossPoint.x = (ka*LineA[0] - LineA[1] - kb*LineB[0] + LineB[1]) / (ka - kb);
    crossPoint.y = (ka*kb*(LineA[0] - LineB[0]) + ka*LineB[1] - kb*LineA[1]) / (ka - kb);
    return crossPoint;
}

// get line' length  获取两点之间的距离
float VpsDetector::get_len(cv::Point begin,cv::Point end)
{
    float len =0;
    len =sqrt((begin.x-end.x)*(begin.x-end.x)+(begin.y-end.y)*(begin.y-end.y));
    return len;
}


//expand_bndbox
void VpsDetector::expand_bndbox(cv::Rect src_rect,cv::Rect &dst_rect,float ratio,int img_w,int img_h)
{   
    if(ratio<0)
        return;

    ROS_INFO("original bndbox (x,y,width,height):(%d,%d,%d,%d)",src_rect.x,src_rect.y,src_rect.width,src_rect.height);
    
    int baseline = src_rect.height<src_rect.width?src_rect.height:src_rect.width;
    int exp_value =std::ceil(baseline*ratio);

    ROS_INFO("exp_value:%d",exp_value);

    dst_rect.x = src_rect.x-exp_value;
    dst_rect.x = dst_rect.x>0?dst_rect.x:0;
    dst_rect.y = src_rect.y - exp_value;
    dst_rect.y = dst_rect.y>0?dst_rect.y:0;

    dst_rect.width = src_rect.width + 2* exp_value;
    dst_rect.width = (dst_rect.x + dst_rect.width)<img_w?dst_rect.width:(img_w-1-dst_rect.x);
    dst_rect.height = src_rect.height + 2*exp_value;
    dst_rect.height = (dst_rect.y+ dst_rect.height)<img_h?dst_rect.height:(img_h-1-dst_rect.y);   

    ROS_INFO("expand_bndbox to (x,y,width,height):(%d,%d,%d,%d)",dst_rect.x,dst_rect.y,dst_rect.width,dst_rect.height);
    
}

//filter abnormal park anchors 
//过滤掉一些检测错误的场景
bool VpsDetector::park_anchor_filter(ParkAnchor src_park,cv::Point bbox_center,float thresold,float max_lw_ratio,float min_lw_ratio/*=1.0*/)
{
    LOG_FUNC_BEGIN
    //ROS_INFO("thresold:%f",thresold);    
    //ROS_INFO("%s",src_park.to_string().c_str());
    
    float l_len,w_len,center_len,lw_ratio;
    cv::Point park_center;
    //if(4!=src_park.num)
    //    return false;
    l_len= get_len(src_park.rtop,src_park.rbot);
    w_len = get_len(src_park.ltop,src_park.rtop);
    lw_ratio  = l_len/w_len;
    //
    if(l_len<w_len)
        lw_ratio = 1/lw_ratio;
    if((lw_ratio<min_lw_ratio)||(lw_ratio>max_lw_ratio))
    {
        cout<<"unvalid h/w ratio:"<<lw_ratio
            <<"should be between ["<<min_lw_ratio<<","<<max_lw_ratio<<"]"<<endl;
        return false;
    }
    
    park_center.x = (src_park.ltop.x + src_park.rbot.x)/2;
    park_center.y = (src_park.ltop.y = src_park.rbot.y)/2;
    
    center_len = get_len(bbox_center, park_center);
    if(center_len>thresold)
    {
        std::cout<<"Warnning: center shift is out of thresold\n";
        cout<<"center_len:"<<center_len<<endl;
        
        return false;
    }

    LOG_FUNC_END
    return true;  
}


// capture nonvertical-horizonl line's crossover within rect
void VpsDetector::get_two_crossover_from_line_to_rect(float a,float b,float c,cv::Rect src_rect,std::vector<cv::Point> &crossover_list)
{   
    float x,y;
    if(b<1e-6)
        return ; 
    //a*x + b*y + c = 0;
    //y =0, crossover in top horizonal line
    y = 0;
    x = -(b*y+c)/a;
    //std::cout<<"src_rect:"<<src_rect.width<<":"<<src_rect.height;
    //std::cout<<"top:"<<x<<":"<<y<<",";
    if((x>=0)&&(x<=(src_rect.width-1)))     
        insert_crossover(crossover_list,cv::Point(floor(x),y));
    
    // crossover in bottom horizonal line
    y = src_rect.height - 1;
    x = -(b*y+c)/a;
    //std::cout<<"bot:"<<x<<":"<<y<<",";
    if((x>=0)&&(x<=(src_rect.width-1)))     
        insert_crossover(crossover_list,cv::Point(floor(x),y));
    
    // crossover in left vectical line
    x =0;
    y = -(a*x+c)/b;
    //std::cout<<"left:"<<x<<":"<<y<<",";
    if((y>=0)&&(y<=(src_rect.height-1)))
         insert_crossover(crossover_list,cv::Point(x,floor(y)));
    
    // crossover in right vertical line
    x = src_rect.width-1;
    y = -(a*x+c)/b;
    //std::cout<<"right:"<<x<<":"<<y<<",";
    if((y>=0)&&(y<=(src_rect.height-1)))
        insert_crossover(crossover_list,cv::Point(x,floor(y)));
    //std::cout<<"finishing "<<std::endl;    
} 

// insert non-repeat crosss over in cross_list
void VpsDetector::insert_crossover(std::vector<cv::Point> &crossover_list,cv::Point src_point)
{
    //std::cout<<"insert crossover--------->";
    int merge_thresold =8;
    bool is_existed = false;
    for(unsigned int i= 0;i<crossover_list.size();i++)
    {   
        cv::Point crossover = crossover_list[i];
        int distance = (src_point.x-crossover.x)*(src_point.x-crossover.x)+(src_point.y-crossover.y)*(src_point.y - crossover.y);
       // std::cout<<"distance is:"<<distance;
        if(distance<merge_thresold) //merge crosspoint
        {
            is_existed = true;
            break;
        }
    }
    
    //std::cout<<"leave insert/n";
    if(!is_existed)
        crossover_list.push_back(src_point);
}


//接收地球坐标系下的车身姿态信息
void VpsDetector::on_pose_msg(const geometry_msgs::PoseStamped &pose_stamp)
{
    m_pose.pose.position = pose_stamp.pose.position;
    m_pose.pose.orientation = pose_stamp.pose.orientation;

    //p_atc_mapper->update(g_delta_x, g_delta_y, src_w, src_h, pos, ort);
}


bool VpsDetector::get_park_anchor(std::vector<cv::Point> anchor_list, ParkAnchor &dst_park, float offset)
{
    cv::Point v10, v32, v_merge;

    if (offset < 0 || offset > 1.0)
        return false;
    if (4 != anchor_list.size())
        return false;
    // line( anchor_list[1],anchor_list[2]) //the maximun distance line
    v10.x = anchor_list[1].x - anchor_list[0].x;
    v10.y = anchor_list[1].y - anchor_list[0].y;
    v32.x = anchor_list[3].x - anchor_list[2].x;
    v32.y = anchor_list[3].y - anchor_list[2].y;

    if ((v10.x * v10.x) < (v32.x * v32.x))
    {
        v_merge.x = ceil(v10.x + offset * (v32.x - v10.x));
        v_merge.y = ceil(v10.y + offset * (v32.y - v10.y));
    }
    else
    {
        v_merge.x = ceil(v32.x + offset * (v10.x - v32.x));
        v_merge.y = ceil(v32.y + offset * (v10.y - v32.y));
    }
    dst_park.ltop = cv::Point(anchor_list[0].x + v_merge.x, anchor_list[0].y + v_merge.y);
    dst_park.rtop = anchor_list[0];
    dst_park.rbot = anchor_list[2];
    dst_park.lbot = cv::Point(anchor_list[2].x + v_merge.x, anchor_list[2].y + v_merge.y);
    return true;
}

