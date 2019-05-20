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

VpsDetector::VpsDetector()
{

}

VpsDetector::~VpsDetector()
{

}


void VpsDetector::set_current_frame(cv::Mat frame)
{
    m_frame = frame;
}

void VpsDetector::init(int argc,char** argv)
{
    //load yolov3 model param
    m_yolo_helper.parse_config_params(argc,argv);
    //
    init_ros(argc,argv);
} 

bool VpsDetector::init_ros(int argc,char** argv)
{
    ros::init(argc,argv,"yolov3");
    ros::NodeHandle node;

    bool ret = load_parameters();
    
    sub_image_raw = node.subscribe(m_image_source_topic, 1, &VpsDetector::on_recv_frame,this);
    //sub_cam_cmd = node.subscribe(m_cam_cmd_topic,1,cmd_callback);

    pub_status_code = node.advertise<std_msgs::UInt8>(m_status_code_topic,1);
    pub_traffic_status = node.advertise<dashboard_msgs::Cmd>(m_traffic_status_topic,1);
    pub_traffic_active = node.advertise<dashboard_msgs::Proc>(m_traffic_active_topic,1);  
    pub_image_raw = node.advertise<sensor_msgs::Image>(m_image_raw_topic,1);

    return ret;
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

//
void VpsDetector::process_carpark1(const cv::Point& begin,
                                          const cv::Point& end,
                                 const cv::Rect& expand_rect,ParkAnchor& draw_anchor)
{
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

//
 void VpsDetector::process_carpark2(const cv::Point& begin,
                                           const cv::Point& end,
                                  const cv::Rect& expand_rect,ParkAnchor& draw_anchor)
 {
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

//
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
              is_valid = false;
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
    cv::Point center(expand_rect.width / 2, expand_rect.height / 2);
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



void VpsDetector::process_bbox(const std::vector<Box>& free_park_boxes)
{
    cv::Rect expand_rect;
    int raw_h = m_frame_input.size().height;
    int raw_w = m_frame_input.size().width;

    for(Box b : free_park_boxes)
    {
          /*--enlarge bndingbox with o.1*width pixels--*/
          expand_bndbox(cv::Rect(b.x0, b.y0, (b.x1 - b.x0), (b.y1 - b.y0)), expand_rect, 0.1, raw_w, raw_h);
          cv::Mat roi_img = m_frame_input(expand_rect);  //抠出当前的detected objet

          /*--two stage park line detection--*/
          std::vector<cv::Vec4i>& lines;
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

          /*--drop unvalid park detection--*/
          if (!is_valid)
          {
                //std::cout<<"waring unvalid draw_anchor\n";
                continue;
          }
          /*--drop center-shift,abnormal shape parks--*/
          cv::Point center(expand_rect.width / 2, expand_rect.height / 2);
          if (!park_anchor_filter(draw_anchor, center, 20, 4.5, 1.2))
          {
                continue;
          }

              
        ATCPark *p_new_park = new ATCPark;
        p_new_park->points_in_img[0] = draw_anchor.ltop.x;
        p_new_park->points_in_img[1] = draw_anchor.ltop.y;
        p_new_park->points_in_img[2] = draw_anchor.rtop.x;
        p_new_park->points_in_img[3] = draw_anchor.rtop.y;
        p_new_park->points_in_img[4] = draw_anchor.rbot.x;
        p_new_park->points_in_img[5] = draw_anchor.rbot.y;
        p_new_park->points_in_img[6] = draw_anchor.lbot.x;
        p_new_park->points_in_img[7] = draw_anchor.lbot.y;
        p_new_park->conf_score = b.score;
        p_new_park->id = 0;
        p_new_park->cls_id = b.class_idx;

        //mapping park from image axes to vector map  axes
        p_atc_mapper->convert_to_vecmap(p_new_park);

        //p_atc_mapper->convert
        bool res = p_atc_tracker->add_tracker(p_new_park);
        if (!res)
        {
            ROS_WARN("add new tracker failed.");
        }
    }
}

void VpsDetector::f()
{
    
}


void VpsDetector::on_recv_frame(const sensor_msgs::Image &image_source)
{
      ROS_INFO("VpsDetector:on_recv_frame begin!");

      cv_bridge::CvImagePtr cv_image = cv_bridge::toCvCopy(image_source, "bgr8"); 
      m_frame = cv_image->image;

      cv::Mat m_frame_tmp;
      img_decode(m_frame, m_frame_tmp, m_delta_x, m_delta_y);
    
      //{"background","free_park", "forbid_park", "incar_park"};
      std::vector<Box> free_park_boxes;
      std::vector<Box> non_free_park_boxes;
      std::vector<BBoxInfo> boxes = m_yolo_helper.do_inference(m_frame_tmp);
      int box_idx = 0;
      for (BBoxInfo b : boxes)
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

          if(class_name == "free_park")
          {
              if(is_effective_park(non_free_park_boxes, b, 0.6))
              {
                  free_park_boxes.push_back(b);
              }
          }
      }

      process_bbox(free_park_boxes);
      
      ROS_INFO("VpsDetector:on_recv_frame end!");
}


//get iou ratio between two bndingbox
float VpsDetector::get_box_overlap_ratio(Box bbox1,Box bbox2)
{
    float x_min,y_min,x_max,y_max;
    float area_bbox1,area_bbox2,area_intersect;
    float iou=0.0;
    if (bbox2.x0 > bbox1.x1 || bbox2.x1 < bbox1.x0 || bbox2.y0 > bbox1.y1 || bbox2.y1 < bbox1.y0) 
        return iou;
    x_min = std::max(bbox1.x0, bbox2.x0);
    y_min = std::max(bbox1.y0, bbox2.y0);
    x_max = std::min(bbox1.x1, bbox2.x1);   
    y_max =std::min(bbox1.y1, bbox2.y1);
    
    //
    area_bbox1 =(bbox1.x1 - bbox1.x0)*(bbox1.y1 - bbox1.y0);
    area_bbox2 =(bbox2.x1 - bbox2.x0)*(bbox2.y1 - bbox2.y0);
    area_intersect = (x_max - x_min)*(y_max -y_min);
    iou = area_intersect/(area_bbox1 + area_bbox2 - area_intersect);

    return iou;
}

//通过iou判断车位是否有效 与任一无效车位iou超过阈值,则认为是无效车位
bool VpsDetector::is_effective_park(std::vector<Box> notfreebox_list,Box freepark_box,float iou_thresold=0.5)
{
    bool effective = true;
    if(iou_thresold<0|| iou_thresold >1)
        iou_thresold = 0.5;//default iou

    float iou;
    for(unsigned int idx =0;idx<notfreebox_list.size();++idx)
    {
        Box box = notfreebox_list[idx];
        if(1!=box.class_idx) // forbidden_park or incar_park
        {
            iou =  get_box_overlap_ratio(box,freepark_box);
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

// image angle expanding  旋转图像
 void VpsDetector::imrotate(cv::Mat& img,cv::Mat& newIm, double angle)
 {
     //better performance 
     std::size_t rot_size=0;
     if(img.rows>img.cols)
         rot_size = img.rows;
     else
         rot_size = img.cols;
     
     cv::Point2f pt(img.cols/2.,img.rows/2.);
     cv::Mat r = cv::getRotationMatrix2D(pt,angle,1.0);
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


// park tracker vis 
void VpsDetector::tracker_vis(cv::Mat &img,std::vector<ATCVisPark> vis_trks)
{
    //const char* cls_names[]={"background:","free","forbidden","incar"};
    for(unsigned int i=0;i<vis_trks.size();i++)
    {
        ATCVisPark *vis_park = &vis_trks[i];
        cv::line(img,cv::Point(vis_park->grid_data[0],vis_park->grid_data[1]),cv::Point(vis_park->grid_data[2],vis_park->grid_data[3]),cv::Scalar(0,255,0),2,CV_AA);
        cv::line(img,cv::Point(vis_park->grid_data[2],vis_park->grid_data[3]),cv::Point(vis_park->grid_data[4],vis_park->grid_data[5]),cv::Scalar(0,255,0),2,CV_AA);
        cv::line(img,cv::Point(vis_park->grid_data[4],vis_park->grid_data[5]),cv::Point(vis_park->grid_data[6],vis_park->grid_data[7]),cv::Scalar(0,255,0),2,CV_AA);
        cv::line(img,cv::Point(vis_park->grid_data[6],vis_park->grid_data[7]),cv::Point(vis_park->grid_data[0],vis_park->grid_data[1]),cv::Scalar(0,255,0),2,CV_AA);
    
        //std::cout<<"--------------------draw bndbox ---------------------"<<vis_trks.size()<<std::endl;
        std::ostringstream score_str;
        std::ostringstream id_str;
        score_str << vis_park->conf_score;
        id_str<<vis_park->id;
        //std::cout<<"cls_id:"<<vis_park[i].cls_id<<":score:"<<score_str.str()<<":id:"<<id_str.str()<<std::endl;
        //std::string label = std::string(cls_names[vis_park->cls_id])+"::"+id_str.str()+"_"+score_str.str();
        std::string label = id_str.str();
        int base_line=1;        
        cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &base_line);
        cv::rectangle(img, cv::Rect(cv::Point(vis_park->grid_data[0],vis_park->grid_data[1]- label_size.height),cv::Size(label_size.width, label_size.height + base_line)),cv::Scalar(255,255,0),CV_FILLED);
        cv::putText(img,label,cv::Point(vis_park->grid_data[0],vis_park->grid_data[1]),cv::FONT_HERSHEY_SIMPLEX,0.8,cv::Scalar(0,0,0));
    }
    return ;
}


// filter abnormal park anchors 
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

