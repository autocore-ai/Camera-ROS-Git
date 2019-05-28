#include "carpark_mgr.h"
#include <algorithm>
#include <ros/console.h>

CarParkMgr::CarParkMgr()
{
    for(int i = 0;i<m_max_park_nums;i++)
    {
        m_carparks.push_back(new ParkInfo);
    }
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
    bool found_same_park = false;

    for(ParkInfo* curr_park:m_carparks)
    {
        if(curr_park->id != -1)
        {
            float iou = get_iou(p_park,curr_park);
            if(iou > m_max_iou) //找到同一个车位
            {
                found_same_park = true;
                update_park(curr_park,p_park);
                check_consecutive_occurrences(curr_park->id);
                break;
            }
        }
    }

    if(!found_same_park)
    {
        int id = get_available_carpark_id(); //当前的可用车位编号
        if(id != -1)
        {
            update_park(m_carparks[id],p_park);
            m_carparks[id]->counts = 1;
            carpark_id_in_curr_frame.push_back(id);
        }
    } 
}

//在处理完一帧图片最后调用
void CarParkMgr::record_parkinfo_in_this_frame()
{
    carpark_id_in_last_frame.swap(carpark_id_in_curr_frame);
    carpark_id_in_curr_frame.clear();
}

//检测这个车位在连续几帧图像里出现.
void CarParkMgr::check_consecutive_occurrences(int id)
{
    vector<int>::iterator it = find(carpark_id_in_last_frame.begin(), carpark_id_in_last_frame.end(), id);
    if( it != carpark_id_in_last_frame.end() )
    {
        m_carparks[id]->counts += 1;
    }
    else
    {
        m_carparks[id]->counts = 1;  
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
    }
}

//检测park1和park2的iou
float CarParkMgr::get_iou(ParkInfo *park1, ParkInfo *park2)
{
    int x_0 = (int)(1000*park1->points_in_world[0]);
    int y_0 = (int)(1000*park1->points_in_world[1]);

    int x_1 = (int)(1000*park1->points_in_world[2]);
    int y_1 = (int)(1000*park1->points_in_world[3]);

    int x_2 = (int)(1000*park1->points_in_world[4]);
    int y_2 = (int)(1000*park1->points_in_world[5]);

    int x_3 = (int)(1000*park1->points_in_world[6]);
    int y_3 = (int)(1000*park1->points_in_world[7]);

    vector<int> x{x_0,x_1,x_2,x_3};
    vector<int> y{y_0,y_1,y_2,y_3};

    sort(x.begin(),x.end());
    int x_min = x[0];
    int x_max = x[3];

    sort(y.begin(),y.end());
    int y_min = x[0];
    int y_max = x[3];

    int iou_point = 0;
    //这边拿到的坐标的单位应该是米,是否应该转换为毫米,否则可能死循环啊!待验证!
    for(int i = x_min; i < x_max;)
    {
        for(int j = y_min; j < y_max;)
        {
            Point curr_point(i,j);
            if(point_in_park(park1,curr_point) && point_in_park(park2,curr_point))
            {
                iou_point += 1;
            }

            j += m_delta_y;
        }

        i += m_delta_x; //这里粒度不能太粗了.  比如以1m*1m为最小粒度单元的话,交叉面积在1平方米以内的可能被认为iou是一样的.
    }
    ROS_INFO("iou_point=%d",iou_point);


    float iou = 0.5;
    return iou;
}
