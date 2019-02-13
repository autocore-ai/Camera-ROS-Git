#ifndef AUTO_TRACKER_H_H_H
#define AUTO_TRACKER_H_H_H
#include"auto_park.h"
#include<string.h>
#include<vector>
/*
struct ATCPark
{
    
    float points_in_world[8];//(x0,y0,x1,y1,x2,y2,x3,y3)
    float points_in_car[8];//(x0,y0,x1,y1,x2,y2,x3,y3)
    float points_in_img[8];//(x0,y0,x1,y1,x2,y2,x3,y4)
    float conf_score=0;
    unsigned int id = 0;
    unsigned int clock =0;
    unsigned int count =0;
    unsigned int cls_id = 0;
};

struct ATCVisPark
{
    int grid_data[8];// x0,y0,x1,y1,x2,y2,x3,y3
    float conf_score;
    unsigned int id;
    //unsigned int clock;    
    unsigned int cls_id;
};

struct ATCPubPark
{
    float grid_data[8];//x0,y0,x1,y1,x2,y2,x3,y3
    float conf_score;
    unsigned int id;
    //unsigned int clock;    
    unsigned int cls_id;
};
*/
struct ATCPoint
{
    int x;
    int y;
};

struct ATCBox{
    ATCPoint point1;
    ATCPoint point2;
};


class ATCParkTracker
{

public:
    ATCParkTracker( unsigned int init_clock, float delta_x,float delta_y,unsigned int clock_thresh, float center_thresh,float iou_thresh, unsigned int iou_level,unsigned int send_epoch)
    {
        if(init_clock <1)
	    init_clock = 1;
	m_init_clock = init_clock;
	m_delta_x = delta_x;
	m_delta_y = delta_y;
	m_clock_thresh = clock_thresh;
	m_center_thresh = center_thresh;
	m_iou_thresh = iou_thresh;
	m_iou_level = iou_level;
        m_send_epoch = send_epoch;
        m_curr_epoch =0;
        m_max_num = 100;
        m_max_id =0;
	
	if(m_iou_level<1)
	    m_iou_level = 1;
       // m_ptrackers_id = new bool[m_max_num];
        for(int i =0; i< int(m_max_num);i++)
            m_ptrackers_id[i] =false;
    }
    ~ATCParkTracker()
    {
       // std::cout<<"destruct ATCParkTracker.\n";
        this->clear();
    }
    void clear();
    //int init()
    bool add_tracker(ATCPark * p_new_park);
    unsigned int get_pub_trackers(std::vector<ATCPubPark> &pub_trks);
    unsigned int get_vis_trackers(std::vector<ATCVisPark> &vis_trks);
    //void vis(cv::Mat &vis_img);
    unsigned int update()
    {
        return update_clock();
    }
    void test_compute_iou();    
                 
private:
    //vector<AutoPark*> m_ptrackers;
    std::vector<ATCPark*>::iterator pop(std::vector<ATCPark*>::iterator it);       
    bool allocate_tracker_id(unsigned int &tracker_id);
    bool callback_tracker_id(unsigned int tracker_id);
    //unsigned int update_trackers();
    bool compare(ATCPark* park1,ATCPark* park2,float &distance);
    float get_iou(ATCPark* park1,ATCPark*park2);
    bool  point_in_vis_park(ATCVisPark *vis_park,ATCPoint pt);
    float get_triangle_area(ATCPoint pt1,ATCPoint pt2,ATCPoint pt3);
    void  get_box_from_vis_park(ATCVisPark *vis_park,ATCBox &bnd_box);
    void  update_park_area(ATCPark* p_new_park);
    unsigned int update_clock();
private:    
    std::vector<ATCPark*> m_ptrackers;
    unsigned int m_max_num;
    unsigned int m_max_id;
    //unsigned int get_stable_trackers(int trks_idx[3],unsigned int clock_thresh);
    int m_init_clock;
    unsigned int m_clock_thresh;
    unsigned int m_send_epoch;
    unsigned int m_curr_epoch;
    float m_center_thresh;
    float m_iou_thresh ;
    unsigned int m_iou_level;
    bool m_ptrackers_id[100];
    float m_delta_x;
    float m_delta_y;
        
};
#endif
