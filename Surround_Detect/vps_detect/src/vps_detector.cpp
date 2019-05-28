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
    //m_p_park = new ATCPark;
} 

void VpsDetector::init_ros(int argc,char **argv)
{
    ros::init(argc,argv,"yolo");
    ros::NodeHandle node;
    ros::NodeHandle private_nh("~");

    private_nh.param<std::string>("image_source_topic", m_sub_topic_image_raw_from_camera, "/usb_cam/image_raw");
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
}

//lines的坐标是在roi_img这张图内的坐标.
void VpsDetector::get_lines(const cv::Mat& roi_img,std::vector<cv::Vec4i>& lines)
{
      cv::Mat tmp_img;
      if(park_edge_detect(roi_img, tmp_img))
      {
         cv::HoughLinesP(tmp_img, lines, 1, CV_PI / 180, 20, 20, 10);
      }
}

// input park patch image,
bool VpsDetector::park_edge_detect(cv::Mat src_img, cv::Mat &dst_img)
{
    cv::Mat mid_img, edge_img, depress_img, mask;
    cv::cvtColor(src_img, mid_img, CV_BGR2GRAY);

    //image enhance
    // cv::equalizeHist(mid_img,mid_img);
    // cv::imshow("gray",mid_img);

    // canny operator
    //std::cout<<"try median filter!"<<std::endl;
    cv::Canny(mid_img, edge_img, 50, 200, 3);
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
    cv::Point center(expand_rect.width / 2, expand_rect.height / 2);
    if (begin.x < center.x)
    {
        draw_anchor.ltop = cv::Point(begin.x, 0);
        draw_anchor.rtop = cv::Point(expand_rect.width - 1, 0);
        draw_anchor.rbot = cv::Point(expand_rect.width - 1, expand_rect.height - 1);
        draw_anchor.lbot = cv::Point(begin.x, expand_rect.height - 1);
    }
    else
    {
        draw_anchor.ltop = cv::Point(0, 0);
        draw_anchor.rtop = cv::Point(begin.x, 0);
        draw_anchor.rbot = cv::Point(begin.x, expand_rect.height - 1);
        draw_anchor.lbot = cv::Point(0, expand_rect.height - 1);
    }

    adjust_coordination(expand_rect,draw_anchor);
}

//长边沿y轴方向
 void VpsDetector::process_carpark2(const cv::Point& begin,
                                           const cv::Point& end,
                                           const cv::Rect& expand_rect,
                                           ParkAnchor& draw_anchor)
 {
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
 }

//长边不沿x轴,y轴方向.
bool VpsDetector::process_carpark3(const cv::Point& begin,
                                        const cv::Point& end,
                               const cv::Rect& expand_rect,ParkAnchor& draw_anchor)
{
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
              //std::cout<<"Warnning:can not get two crossover vs("<<next_crossover.size()<<" in perpendicular line \n";
              valid_cakpark = false;
              continue;
          }
          for (unsigned int pline_idx = 0; pline_idx < next_crossover.size(); pline_idx++)
          {
              rect_crossover.push_back(next_crossover[pline_idx]);
          }
      }

      adjust_coordination(expand_rect,draw_anchor);

      return valid_cakpark;
}

//对应3种车位类型,正常倒库的长方形,路边侧方位倒进去的长方形,斜着的车位(平行四边形那种)
//只知道line是长边,但是并不知道是哪一条长边
bool VpsDetector::process_carpark(const cv::Vec4i& line,const cv::Rect& expand_rect,ParkAnchor& draw_anchor)
{
    bool valid = true;

    cv::Point begin(line[0], line[1]);
    
    cv::Point end(line[2], line[3]); // ax+by+c =0;

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

    return valid;
}

void VpsDetector::process_bbox(const std::vector<BBoxInfo>& free_park_boxes)
{
    cv::Rect expand_rect;
    int raw_h = m_frame_input.size().height;
    int raw_w = m_frame_input.size().width;

    for(BBoxInfo box_info : free_park_boxes)
    {
          //拿到boundingbox,并不意味着拿到了park的坐标.下面要做的就是从boundingbox范围内的图内把park识别出来.
          BBox b = box_info.box;
          /*--enlarge bndingbox with o.1*width pixels--*/
          expand_bndbox(cv::Rect(b.x1, b.y1, (b.x2 - b.x1), (b.y2 - b.y1)), expand_rect, 0.1, raw_w, raw_h);
          cv::Mat roi_img = m_frame_input(expand_rect);  //抠出当前的detected objet
 
          /*--two stage park line detection--*/
          std::vector<cv::Vec4i> lines;
          get_lines(roi_img,lines);
          if (lines.size() < 1)
          {
              std::cout<<"warnning can not detect lines\n";
              continue;
          }       

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
          if (!is_valid)
          {
                std::cout<<"waring unvalid draw_anchor"<<std::endl;
                continue;
          }
          cv::Point center(expand_rect.width / 2, expand_rect.height / 2);
          if (!park_anchor_filter(draw_anchor, center, 20, 4.5, 1.2))
          {
                continue;
          }

          process_curr_park(draw_anchor,box_info);  

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
            park_obj.x0 = p->points_in_world[0];
            park_obj.y0 = p->points_in_world[1];
            park_obj.x1 = p->points_in_world[2];
            park_obj.y1 = p->points_in_world[3];
            park_obj.x2 = p->points_in_world[4];
            park_obj.y2 = p->points_in_world[5];
            park_obj.x3 = p->points_in_world[6];
            park_obj.y3 = p->points_in_world[7];
            park_obj.id = p->id;

            m_parkobj_msg.obj.push_back(park_obj);
        }

        m_pub_image_obj.publish(m_parkobj_msg);
    }
}

//
void VpsDetector::process_curr_park(const ParkAnchor& draw_anchor,const BBoxInfo& box_info)
{
    m_p_park->points_in_img[0] = draw_anchor.ltop.x;
    m_p_park->points_in_img[1] = draw_anchor.ltop.y;
    m_p_park->points_in_img[2] = draw_anchor.rtop.x;
    m_p_park->points_in_img[3] = draw_anchor.rtop.y;
    m_p_park->points_in_img[4] = draw_anchor.rbot.x;
    m_p_park->points_in_img[5] = draw_anchor.rbot.y;
    m_p_park->points_in_img[6] = draw_anchor.lbot.x;
    m_p_park->points_in_img[7] = draw_anchor.lbot.y;
    m_p_park->conf_score = box_info.prob;
    m_p_park->id = 0;
    m_p_park->cls_id = box_info.classId;

    //计算地球坐标系下的坐标
    m_p_atc_mapper->convert_to_vecmap(m_p_park);
    
    ParkInfo info;
    for(int i=0;i<8;i++)
    {
        info.points_in_world[i] = m_p_park->points_in_world[i];
        info.points_in_img[i] = m_p_park->points_in_img[i];
    }
    
    m_p_carpark_mgr->add_carpark(&info);
}

//更新车身姿态信息,模型输入图片尺寸信息,以便后续不同坐标系内的坐标转换
void VpsDetector::update_carpose()
{
    int img_w = m_frame_input.cols;
    int img_h = m_frame_input.rows;
    
    m_p_atc_mapper->update(m_delta_x,m_delta_y,img_w,img_h,m_pose.pose.position,m_pose.pose.orientation);
}

void VpsDetector::on_recv_frame(const sensor_msgs::Image &image_source)
{
      ROS_INFO("VpsDetector:on_recv_frame begin!");

      //
      m_parkobj_msg.header = image_source.header;
      
      cv_bridge::CvImagePtr cv_image = cv_bridge::toCvCopy(image_source, "bgr8"); 
      m_frame = cv_image->image;

      //调整图像尺寸
      img_decode(m_frame, m_frame_input, m_delta_x, m_delta_y);

      //更新车身姿态信息,模型输入图片尺寸信息,以便后续不同坐标系内的坐标转换
      update_carpose();

      //imrotate(src_frame, frame_input,0);

      //推理
      //{"background","free_park", "forbid_park", "incar_park"};
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
      draw_park_on_img(frame_show);

      pub_img(frame_show);

      cv::imshow("vps_show", frame_show);
      cv::waitKey(10); //https://stackoverflow.com/questions/5217519/what-does-opencvs-cvwaitkey-function-do

      ROS_INFO("VpsDetector:on_recv_frame end!");
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
          if(class_name == "free_park")
          {
              if(is_effective_park(non_free_park_boxes, b.box, 0.6))
              {
                  free_park_boxes.push_back(b);
              }
          }
      }
}

//在图片上绘制出车位框 
void VpsDetector::draw_park_on_img(cv::Mat &img)
{
    vector<const ParkInfo* > park_info;
    m_p_carpark_mgr->get_effective_parks(park_info);

    for(auto p : park_info)
    {
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
}


void VpsDetector::pub_img(const cv::Mat& detect_show)
{
      sensor_msgs::ImagePtr roi_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", detect_show).toImageMsg();
      roi_msg->header.seq = m_seq++;
      roi_msg->header.frame_id = "vps image";
      roi_msg->header.stamp = ros::Time::now();
      m_pub_image_raw.publish(roi_msg);
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
        
        if(class_name != "free_park") // forbidden_park or incar_park
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
    //std::cout<<"raw_w:"<<raw_w<<"raw_h:"
    // get raw_data from yulan device
    if((m_yulan_w<=raw_w)&&(m_yulan_h<=raw_h))
    {
        dst_img=raw_data(cv::Rect(0,0,m_yulan_w,m_yulan_h));
        dst_delta_x = m_delta_x;
        dst_delta_y = m_delta_y;
    }else// get raw_data from simulator
    {
        dst_img = raw_data;
        dst_delta_x = 24.0/raw_w;
        dst_delta_y = 30.0/raw_h;
    }
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
    
    int baseline = src_rect.height<src_rect.width?src_rect.height:src_rect.width;
    int exp_value =std::ceil(baseline*ratio);
    dst_rect.x = src_rect.x-exp_value;
    dst_rect.x = dst_rect.x>0?dst_rect.x:0;
    dst_rect.y = src_rect.y - exp_value;
    dst_rect.y = dst_rect.y>0?dst_rect.y:0;

    dst_rect.width = src_rect.width + 2* exp_value;
    dst_rect.width = (dst_rect.x + dst_rect.width)<img_w?dst_rect.width:(img_w-1-dst_rect.x);
    dst_rect.height = src_rect.height + 2*exp_value;
    dst_rect.height = (dst_rect.y+ dst_rect.height)<img_h?dst_rect.height:(img_h-1-dst_rect.y);    
}

//filter abnormal park anchors 
//过滤掉一些检测错误的场景
bool VpsDetector::park_anchor_filter(ParkAnchor src_park,cv::Point bbox_center,float thresold,float max_lw_ratio,float min_lw_ratio/*=1.0*/)
{
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
        return false;
    park_center.x = (src_park.ltop.x + src_park.rbot.x)/2;
    park_center.y = (src_park.ltop.y = src_park.rbot.y)/2;
    
    center_len = get_len(bbox_center, park_center);
    if(center_len>thresold)
    {
        //std::cout<<"Warnning: center shift is out of thresold\n";
        return false;
    }
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
}



