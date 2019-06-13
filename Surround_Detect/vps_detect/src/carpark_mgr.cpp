#include "carpark_mgr.h"
#include <algorithm>
#include <ros/console.h>
#include <ros/ros.h>

CarParkMgr::CarParkMgr()
{
    for(int i = 0;i<m_max_park_nums;i++)
    {
        m_carparks.push_back(new ParkInfo);
    }

    read_cfg();
}

CarParkMgr::~CarParkMgr()
{
    for(int i = 0;i<m_max_park_nums;i++)
    {
        if(nullptr != m_carparks[i])
        {
            delete m_carparks[i];
            m_carparks[i] = nullptr;
        } 
    }
}

void CarParkMgr::add_carpark(ParkInfo* p_park)
{
    ROS_INFO("%s begin",__FUNCTION__);

    //ROS_INFO("add_carpark\n%s",p_park->points_in_img_to_string().c_str());
    
    //ROS_INFO("%s",p_park->points_in_world_to_string().c_str());
    
    bool found_same_park = false;

    for(ParkInfo* curr_park:m_carparks)
    {
        if(curr_park->id != -1)
        {
            ROS_INFO("compare park with park%d",curr_park->id);
            float iou = get_iou(p_park,curr_park);
            ROS_INFO("iou_area:%f",iou);
            if(iou > m_max_iou) //找到同一个车位
            {
                ROS_INFO("update park:%d",curr_park->id);
                found_same_park = true;
                update_park(curr_park,p_park);
                check_consecutive_occurrences(curr_park->id);

                m_carpark_id_in_curr_frame.push_back(curr_park->id);

                break;
            }
        }
    }

    if(!found_same_park)
    {
        int id = get_available_carpark_id(); //当前的可用车位编号
        if(id != -1)
        {
            ROS_INFO("add new park:%d",id);
            update_park(m_carparks[id],p_park);
            m_carparks[id]->counts = 1;

            m_carpark_id_in_curr_frame.push_back(id);
        }
    } 

    ROS_INFO("%s end",__FUNCTION__);
}

//在处理完一帧图片最后调用
void CarParkMgr::record_parkinfo_in_this_frame()
{
    for(auto parkid_lastframe : m_carpark_id_in_last_frame)
    {       
        ROS_ERROR("check last frame park %d",parkid_lastframe);
        vector<int>::iterator it = find(m_carpark_id_in_curr_frame.begin(), m_carpark_id_in_curr_frame.end(), parkid_lastframe);
        if(it == m_carpark_id_in_curr_frame.end())
        {
            m_carparks[parkid_lastframe]->counts = 1; 
            ROS_ERROR("RESET %d",parkid_lastframe);
        }
    }

    m_carpark_id_in_last_frame = m_carpark_id_in_curr_frame;
  
    m_carpark_id_in_curr_frame.clear();
}

//检测这个车位在连续几帧图像里出现.
void CarParkMgr::check_consecutive_occurrences(int id)
{
    vector<int>::iterator it = find(m_carpark_id_in_last_frame.begin(), m_carpark_id_in_last_frame.end(), id);
    if( it != m_carpark_id_in_last_frame.end() )
    {
        m_carparks[id]->counts += 1;
        ROS_INFO("add park%d counts to %d",id,m_carparks[id]->counts);
    }
    else
    {
        m_carparks[id]->counts = 1;  
        ROS_INFO("reset park%d counts to %d",id,m_carparks[id]->counts);
    }
}

//
int CarParkMgr::get_available_carpark_id()
{
    int id = -1;

    for(int idx = 0; idx < (int)m_carparks.size();idx++)
    {
        if(-1 == m_carparks[idx]->id)  //尚未分配的车位
        {
            //ROS_INFO("send parks:%d",nums_pub);
            m_carparks[idx]->id = idx; //分配编号
            id = idx;
            break;
        }
    }

    if(id == -1)
    {
        ROS_WARN("no empty carpark in m_carparks");
    }

    return id;
}

void CarParkMgr::update_park(ParkInfo *park1, ParkInfo *park2)
{
    for(int i=0;i<8;i++)
    {
        park1->points_in_world[i] =  park2->points_in_world[i];
        park1->points_in_img[i] =  park2->points_in_img[i];
        park1->points_in_car[i] =  park2->points_in_car[i];    
    }

    //park1->counts += 1;
}

//检测park1和park2的iou
float CarParkMgr::get_iou(ParkInfo *park_src, ParkInfo *park_dst)
{
    ROS_INFO("park_src********%s",park_src->parkinfo_to_string().c_str());
    ROS_INFO("park_dst********%s",park_dst->parkinfo_to_string().c_str());

    int x_0 = (int)(1000*park_src->points_in_world[0]);//convert m to mm
    int y_0 = (int)(1000*park_src->points_in_world[1]);

    int x_1 = (int)(1000*park_src->points_in_world[2]);
    int y_1 = (int)(1000*park_src->points_in_world[3]);

    int x_2 = (int)(1000*park_src->points_in_world[4]);
    int y_2 = (int)(1000*park_src->points_in_world[5]);

    int x_3 = (int)(1000*park_src->points_in_world[6]);
    int y_3 = (int)(1000*park_src->points_in_world[7]);

    vector<int> x{x_0,x_1,x_2,x_3};
    vector<int> y{y_0,y_1,y_2,y_3};

    sort(x.begin(),x.end());
    int x_min = x[0];
    int x_max = x[3];

    sort(y.begin(),y.end());
    int y_min = y[0];
    int y_max = y[3];

    //ROS_INFO("x_min,y_min,x_max,y_max=(%d,%d,%d,%d)",x_min,y_min,x_max,y_max);
    
    int iou_point = 0;
    int x_step = 50;//50mm
    int y_step = 50;//50mm

    //这边拿到的坐标的单位应该是米,是否应该转换为毫米,否则可能死循环啊!待验证!
    for(int i = x_min; i < x_max;)
    {
        for(int j = y_min; j < y_max;)
        {
            Point curr_point(i,j);
            if(point_in_park(park_src,curr_point) && point_in_park(park_dst,curr_point))
            {
                //cout<<"add iou point"<<endl;
                iou_point += 1;
            }

            j += 50;
        }

        i += 50; //以50mm * 50mm为最小单元
    }
    
    float iou_area = float(iou_point * x_step * y_step) / (1000 * 1000); //转换为平方米
    ROS_INFO("iou_point=%d,iou_area=%f",iou_point,iou_area);

    float area1 = cal_area(park_src);
    float area2 = cal_area(park_dst);

    float iou = iou_area/(area1 + area2 - iou_area);

    ROS_INFO("area1=%f,area2=%f,iou=%f",area1,area2,iou);
    
    return iou;
}

//calculate the correct area for all parallelograms
float CarParkMgr::cal_area(ParkInfo *park)
{  
    float p0_x = park->points_in_world[0];
    float p0_y = park->points_in_world[1];

    float p1_x = park->points_in_world[2];
    float p1_y = park->points_in_world[3];

    //float p2_x = park->points_in_world[4];
    //float p2_y = park->points_in_world[5];

    float p3_x = park->points_in_world[6];
    float p3_y = park->points_in_world[7];

    double dx1 = p3_x - p0_x;
    double dy1 = p3_y - p0_y;
    double dx2 = p1_x - p0_x;
    double dy2 = p1_y - p0_y;
    double area = abs(dx1*dy2 - dy1*dx2);

    ROS_INFO("park%d area=%f",park->id,area);

    return area;
}


void CarParkMgr::get_effective_parks(vector<const ParkInfo* >& park_info)
{
    for(auto park:m_carparks)
    {  
        if(park->id != -1)
        {
            //ROS_ERROR("found effective park %s",park->statics().c_str());
            if(park->counts >= m_min_frame)
            {
                ROS_ERROR("found effective park %s",park->statics().c_str());
                park_info.push_back(park);
            }
        }
    }
}

bool CarParkMgr::point_in_park(ParkInfo *park,Point pt)
{
    Point a(1000*park->points_in_world[0],1000*park->points_in_world[1]);
    Point b(1000*park->points_in_world[2],1000*park->points_in_world[3]);
    Point c(1000*park->points_in_world[4],1000*park->points_in_world[5]);
    Point d(1000*park->points_in_world[6],1000*park->points_in_world[7]);

    float area_pt = get_triangle_area(a, b, pt) + get_triangle_area(b, c, pt) + get_triangle_area(c, d, pt) + get_triangle_area(d, a, pt);
    float area_park = get_triangle_area(a, b, c) + get_triangle_area(c, d, a);
    
    return (abs(area_pt - area_park) < 0.003);
}

//计算三角形面积  单位需要转换为毫米
float CarParkMgr::get_triangle_area(Point a, Point b, Point c)
{
    float result = abs((a.x * b.y + b.x * c.y + c.x * a.y - b.x * a.y - c.x * b.y - a.x * c.y) / 2.0);
    return result;
}

//
void CarParkMgr::read_cfg()
{
    ros::NodeHandle private_nh("~");

    private_nh.param<int>("min_frame", m_min_frame, 5);
    ROS_INFO("Setting m_min_frame to %d", m_min_frame); 

    private_nh.param<float>("max_iou", m_max_iou, 0.5);
    ROS_INFO("Setting m_max_iou to %f", m_max_iou);
}

