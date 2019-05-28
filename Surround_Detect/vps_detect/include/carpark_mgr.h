#ifndef __CARPARK_MGR__
#define __CARPARK_MGR__

#include <algorithm>
#include <vector>
using namespace std;

struct ParkInfo
{
    float points_in_img[8];  //车位图片坐标
    float points_in_world[8];//车位现实世界坐标
    
    int id = -1;//车位编号 有效编号从0开始  

    int counts = 0; //连续在几帧图片中出现. 超过一定次数才告知下游程序

    int disappearance = 0;//todo:连续几帧图像检测不到当前park,就该把对应的m_carparks[id]放给新park用.
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
    void get_effective_parks(vector<const ParkInfo* >& park_info)
    {
        for(auto park:m_carparks)
        {
            if(park->counts >m_max_frame)
            {
                park_info.push_back(park);
            }
        }
    }
private:
    //检测这个车位在连续几帧图像里出现.
    void check_consecutive_occurrences(int id);

    //
    int get_available_carpark_id();

    //
    void update_park(ParkInfo *park1, ParkInfo *park2);

    //
    float get_iou(ParkInfo *park1, ParkInfo *park2);

    //求一个平行四边形内的最小粒度单元个数.
    //int get_points_num_in_park()


    //判断点是否在平行四边形内.  单位转换为毫米处理
    bool point_in_park(ParkInfo *park,Point pt)
    {
        Point a(park->points_in_world[0],park->points_in_world[1]);
        Point b(park->points_in_world[2],park->points_in_world[3]);
        Point c(park->points_in_world[4],park->points_in_world[5]);
        Point d(park->points_in_world[6],park->points_in_world[7]);

        float area_pt = get_triangle_area(a, b, pt) + get_triangle_area(b, c, pt) + get_triangle_area(c, d, pt) + get_triangle_area(d, a, pt);
        float area_park = get_triangle_area(a, b, c) + get_triangle_area(c, d, a);
        
        return (abs(area_pt - area_park) < 0.003);
    }

    //计算三角形面积  单位需要转换为毫米
    float get_triangle_area(Point a, Point b, Point c)
    {
        float result = abs((a.x * b.y + b.x * c.y + c.x * a.y - b.x * a.y - c.x * b.y - a.x * c.y) / 2.0);
        return result;
    }
private:
    vector<ParkInfo*> m_carparks;         //存放车位,id=-1表示尚无对应车位数据.
    vector<int> carpark_id_in_last_frame; //上一帧出现的车位id. 在一帧图像处理完毕后更新    
    vector<int> carpark_id_in_curr_frame; //当前帧出现的车位id

    int   m_max_park_nums = 100; //保存车位信息数量上限
    float m_max_iou = 0.5;   //超过此iou将被认为是相同车位
    int   m_max_frame = 5;   //连续x帧图像都出现的park才是有效park

    //在 x 方向上每个像素代表 21.9mm， y 方向每个像素代表 20.1mm
    float  m_delta_x = 0.0219;// yulan: 0.0219 simu:0.0285  
    float  m_delta_y = 0.0201; //yulan: 0.0201 simu 0.0287
};

#endif