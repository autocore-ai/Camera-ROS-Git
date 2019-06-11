#ifndef __CARPARK_MGR__
#define __CARPARK_MGR__

#include <algorithm>
#include <vector>
#include <string>
#include <iostream>
#include <sstream> 
using namespace std;

struct ParkInfo
{
    float points_in_img[8];  //车位图片坐标
    float points_in_world[8];//车位现实世界坐标
    float points_in_car[8];//相对车子的坐标
    
    int id = -1;//车位编号 有效编号从0开始  

    int counts = 0; //连续在几帧图片中出现. 超过一定次数才告知下游程序

    //int disappearance = 0;//todo:连续几帧图像检测不到当前park,就该把对应的m_carparks[id]放给新park用.

    string parkinfo_to_string()
    {
        return statics()+ points_in_img_to_string() +  points_in_car_to_string() + points_in_world_to_string();
    }

    string points_in_car_to_string()
    {
        std::ostringstream stringStream;
        stringStream << "points_in_car";
        for(int i = 0;i<8;i++)
        {
            stringStream << points_in_car[i] ;
            if(i != 7)
            {
                stringStream << ",";
            }
            else
            {
                stringStream << "\n";
            }
        }
        
        return stringStream.str();  
    }
    
    string points_in_img_to_string()
    {
        std::ostringstream stringStream;
        stringStream << "points_in_img";
        for(int i = 0;i<8;i++)
        {
            stringStream << points_in_img[i] ;
            if(i != 7)
            {
                stringStream << ",";
            }
            else
            {
                stringStream << "\n";
            }
        }
        
        return stringStream.str();  
    }

    string points_in_world_to_string()
    {
        std::ostringstream stringStream;
        stringStream << "points_in_world";
        for(int i = 0;i<8;i++)
        {
            stringStream << points_in_world[i] ;
            if(i != 7)
            {
                stringStream << ",";
            }
            else
            {
                stringStream << "\n";
            }
        }
        
        return stringStream.str();  
    }
    
    string statics()
    {
        std::ostringstream stringStream;
        stringStream <<"park id:"<<id<<","
                     <<"counts:"<<counts
                     <<endl;

        return stringStream.str();
    }
};


struct Point
{
    Point()
    {

    }
    Point(int x_,int y_):x(x_),y(y_)
    {

    }

    int x;
    int y;
};

class CarParkMgr
{
public:   
    CarParkMgr();

    ~CarParkMgr();

    //从m_carparks中分配车位,填入车位id及坐标
    void add_carpark(ParkInfo* p_park);

    //在处理完一帧图片最后调用
    void record_parkinfo_in_this_frame();

    //获取有效车位
    void get_effective_parks(vector<const ParkInfo* >& park_info);
private:
    //检测这个车位在连续几帧图像里出现.
    void check_consecutive_occurrences(int id);

    //
    int get_available_carpark_id();

    //
    void update_park(ParkInfo *park1, ParkInfo *park2);

    //
    float get_iou(ParkInfo *park_src, ParkInfo *park_dst);

    //求一个平行四边形内的最小粒度单元个数.
    //int get_points_num_in_park()


    //判断点是否在平行四边形内.  单位转换为毫米处理
    bool point_in_park(ParkInfo *park,Point pt);

    //计算三角形面积  单位需要转换为毫米
    float get_triangle_area(Point a, Point b, Point c);

    //
    void read_cfg();

    //calculate the correct area for all parallelograms
    float cal_area(ParkInfo *park);
private:
    vector<ParkInfo*> m_carparks;         //存放车位,id=-1表示尚无对应车位数据.
    vector<int> m_carpark_id_in_last_frame; //上一帧出现的车位id. 在一帧图像处理完毕后更新    
    vector<int> m_carpark_id_in_curr_frame; //当前帧出现的车位id

    int   m_max_park_nums = 100; //保存车位信息数量上限
    float m_max_iou = 0.5;   //重叠面积比例超过这个被视为同一车位
    int   m_min_frame = 5;   //连续x帧图像都出现的park才是有效park
};

#endif