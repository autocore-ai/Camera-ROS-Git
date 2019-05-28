#include "auto_tracker_sdk.h"
#include <vector>
#include <iostream>
#include <math.h>

//add a detected park and update trackers
bool ATCParkTracker::add_tracker(ATCPark *p_new_park)
{
    bool non_new = false;
    // park clock --
    //update_clock();i
    float max_iou_score = 0;
    float max_iou_id = 0;
    update_park_area(p_new_park);
    for (unsigned int i = 0; i < m_ptrackers.size(); i++)
    {
        ATCPark *p_iter = m_ptrackers[i];
        if ((p_iter->clock) > 0)
        {
            float distance;
            bool ret = compare(p_new_park, p_iter, distance);

            // < center thresh
            if (ret)
            {
                float iou_score = get_iou(p_iter, p_new_park);
                if (iou_score > max_iou_score)
                {
                    max_iou_score = iou_score;
                    max_iou_id = i;
                }
            }
        }
    }
    if (max_iou_score > m_iou_thresh)
        non_new = true;
    // park track succeed,update tracker status,
    if (non_new)
    {
        ATCPark *p_iter = m_ptrackers[max_iou_id];
        for (unsigned int j = 0; j < 8; j++)
        {
            p_iter->points_in_world[j] = p_new_park->points_in_world[j];
            p_iter->points_in_img[j] = p_new_park->points_in_img[j];
            p_iter->points_in_car[j] = p_new_park->points_in_car[j];
        }
        p_iter->conf_score = p_new_park->conf_score;
        p_iter->cls_id = p_new_park->cls_id;
        p_iter->clock = m_init_clock;
        p_iter->area = p_new_park->area;
        //std::cout<<"--------tracking:("<<p_iter->id<<" , "<<max_iou_score<<")\n";

        // count upper:100
        if ((p_iter->count) < 100)
        {
            p_iter->count += 1;
        }
        delete p_new_park;
    }
    else
    {
        unsigned int new_id = m_max_num;
        bool ret = allocate_tracker_id(new_id);
        if (!ret)
        {
            delete p_new_park;
            return false;
        }
        // allocate new tracker id
        p_new_park->clock = m_init_clock;
        p_new_park->id = new_id;
        p_new_park->count = 1;
        m_ptrackers.push_back(p_new_park);
        std::cout << "--------new:(" << new_id << " , " << max_iou_score << ")\n";
    }
    return true;
}

//get current tracker objects
unsigned int ATCParkTracker::get_pub_trackers(std::vector<ATCPubPark> &pub_trks)
{
    unsigned int num_trks = 0;
    pub_trks.clear();

    //send trackes every m_send_epochs
    if (m_curr_epoch == m_send_epoch)
    {
        m_curr_epoch = 0;
        for (unsigned int i = 0; i < m_ptrackers.size(); i++)
        {
            ATCPark *p_iter = m_ptrackers[i];

            //tracker succed
            if ((p_iter->count) > 1)
            {
                // get stable tracker
                if ((p_iter->clock) > m_clock_thresh)
                {
                    ATCPubPark new_pub_park;
                    new_pub_park.conf_score = p_iter->conf_score;
                    new_pub_park.id = p_iter->id;
                    new_pub_park.cls_id = p_iter->cls_id;

                    for (int j = 0; j < 8; j++)
                    {
                        new_pub_park.grid_data[j] = p_iter->points_in_car[j];
                    }
                    pub_trks.push_back(new_pub_park);
                    num_trks += 1;
                }
            }
        }
    }
    return num_trks;
}

// get vis parks from
unsigned int ATCParkTracker::get_vis_trackers(std::vector<ATCVisPark> &vis_trks)
{
    unsigned int num_trks = 0;
    std::vector<ATCPark *>::iterator it;
    ATCPark *p_iter;
    for (it = m_ptrackers.begin(); it != m_ptrackers.end();)
    {
        p_iter = *it;
        if ((p_iter->count) > 1)
        {
            // get stable trackers
            if ((p_iter->clock) > m_clock_thresh)
            {
                ATCVisPark new_vis_park;
                new_vis_park.conf_score = p_iter->conf_score;
                new_vis_park.id = p_iter->id;
                new_vis_park.cls_id = p_iter->cls_id;
                //
                for (int i = 0; i < 8; i++)
                {
                    new_vis_park.grid_data[i] = int(p_iter->points_in_img[i]);
                }
                vis_trks.push_back(new_vis_park);
                num_trks += 1;
            }
        }
        ++it;
    }
    return num_trks;
}

// pop tracker from trackers list and return next iter
std::vector<ATCPark *>::iterator ATCParkTracker::pop(std::vector<ATCPark *>::iterator it)
{
    ATCPark *p_iter = *it;
    std::cout << "------pop:(" << p_iter->id << " , " << p_iter->count << ")" << std::endl;
    it = m_ptrackers.erase(it);
    delete p_iter;
    return it;
}

// update tracker's clock and pop tracker when life clock is end
unsigned int ATCParkTracker::update_clock()
{
    std::vector<ATCPark *>::iterator it;
    ATCPark *p_iter;
    unsigned int num_trackers = 0;
    m_curr_epoch += 1;
    for (it = m_ptrackers.begin(); it != m_ptrackers.end();)
    {
        p_iter = *it;
        p_iter->clock -= 1;

        // clock is end,pop tracker
        if ((p_iter->clock) < 1)
        {
            unsigned int tracker_id = p_iter->id;
            it = pop(it);
            callback_tracker_id(tracker_id);
        }
        else
        {
            ++it;
            num_trackers += 1;
        }
    }
    return num_trackers;
}

bool ATCParkTracker::compare(ATCPark *park1, ATCPark *park2, float &distance)
{
    bool ret = false;
    float center1_x = (park1->points_in_world[0] + park1->points_in_world[4]) * 0.5;
    float center1_y = (park1->points_in_world[1] + park1->points_in_world[5]) * 0.5;
    float center2_x = (park2->points_in_world[0] + park2->points_in_world[4]) * 0.5;
    float center2_y = (park2->points_in_world[1] + park2->points_in_world[5]) * 0.5;
    distance = sqrt((center1_x - center2_x) * (center1_x - center2_x) + (center1_y - center2_y) * (center1_y - center2_y));
    /*float distance_x = (center2_x-center1_x)*(center2_x-center1_x);
    float distance_y = (center2_y-center1_y)*(center2_y-center1_y);
    float distance;
    if(distance_x<distance_y)
    {
        distance = sqrt(distance_x);
    }
    else
    {
	distance = sqrt(distance_y);
    }*/
    // std::cout<<"distance is:"<<distance<<std::endl;
    if (distance < m_center_thresh)
        ret = true;
    return ret;
}

/*unsigned int ATCParkTracker::update_trackers()
{
    std::vector(ATCPark
    for(unsigned int i=0;i<m_ptrackers.size(); )
}
*/
//
bool ATCParkTracker::allocate_tracker_id(unsigned int &tracker_id)
{
    unsigned int begin = m_max_id;
    for (unsigned int i = 0; i < m_max_num; i++)
    {
        begin = ((m_max_id + i) % m_max_num);
        //have free id
        if (!m_ptrackers_id[begin])
        {
            m_ptrackers_id[begin] = true;
            tracker_id = begin;
            m_max_id = (begin + 1) % m_max_num;
            return true;
        }
    }
    return false;
}

//
bool ATCParkTracker::callback_tracker_id(unsigned int tracker_id)
{
    //std::vector<ATCPARKTracker>
    //for(unsigned int i)
    if (tracker_id >= m_max_num)
        return false;
    std::vector<ATCPark *>::iterator it;
    ATCPark* p_park = nullptr;
    for (it = m_ptrackers.begin(); it != m_ptrackers.end();)
    {
        p_park = *it;
        if (tracker_id == (p_park->id))
        { //std::cout<<"obj(id:"<<tracker_id<<")is still in trackers.cannot call back obj id"<<std::endl;
            return false;
        }
        ++it;
    }
    m_ptrackers_id[tracker_id] = false;
    return true;
}
// draw park in image
/*void ATCTracker::vis(cv::Mat &vis_img)
{
    std::vector<ATCPark*>::iterator it;
    ATCPark *p_iter;
    for(it= this->m_trackers.begin();it!=m_trackers.end();)
    {
        p_iter = *it;
	if(p_iter->clock>=3)
	{
	    // draw parks in vis_img
	}	
}
*/

/*
unsigned int ATCTracker::get_stable_trackers(int trks_idx[3], unsigned int clock_thresh)
{
    for(int i =0;i<3;i++)
        trk_idx[i]=m_ptraclers.size();
}*/

float ATCParkTracker::get_iou(ATCPark *park1, ATCPark *park2)
{
    ATCVisPark vis_park1;
    ATCVisPark vis_park2;
    ATCBox bnd_box;
    ATCPoint pt;
    bool in_park;
    bool in_iou;
    for (int i = 0; i < 8; i++)
    {
        vis_park1.grid_data[i] = int(park1->points_in_world[i]);
        vis_park2.grid_data[i] = int(park2->points_in_world[i]);
    }
    get_box_from_vis_park(&vis_park1, bnd_box);

    // walk the bnd_box and find points in park1 and in IOU
    float num_in_iou = 0;
    for (int i = bnd_box.point1.x; i <= bnd_box.point2.x;)
    {
        for (int j = bnd_box.point1.y; j <= bnd_box.point2.y;)
        {
            pt.x = i;
            pt.y = j;
            in_park = point_in_vis_park(&vis_park1, pt);
            if (in_park)
            {
                in_iou = point_in_vis_park(&vis_park2, pt);
                if (in_iou)
                {
                    num_in_iou += 1;
                }
            }
            j += m_iou_level * m_delta_y;
        }
        i += m_iou_level * m_delta_x;
    }

    return (num_in_iou / (park1->area + park2->area + num_in_iou + 0.0001));
}

//
bool ATCParkTracker::point_in_vis_park(ATCVisPark *vis_park, ATCPoint pt)
{
    ATCPoint a, b, c, d;
    a.x = vis_park->grid_data[0];
    a.y = vis_park->grid_data[1];
    b.x = vis_park->grid_data[2];
    b.y = vis_park->grid_data[3];
    c.x = vis_park->grid_data[4];
    c.y = vis_park->grid_data[5];
    d.x = vis_park->grid_data[6];
    d.y = vis_park->grid_data[7];
    float area_pt = get_triangle_area(a, b, pt) + get_triangle_area(b, c, pt) + get_triangle_area(c, d, pt) + get_triangle_area(d, a, pt);
    float area_park = get_triangle_area(a, b, c) + get_triangle_area(c, d, a);
    return (abs(area_pt - area_park) < 0.003);
    //return (area_pt == area_park);
}

//#
float ATCParkTracker::get_triangle_area(ATCPoint a, ATCPoint b, ATCPoint c)
{
    float result = abs((a.x * b.y + b.x * c.y + c.x * a.y - b.x * a.y - c.x * b.y - a.x * c.y) / 2.0);
    return result;
}

//
void ATCParkTracker::get_box_from_vis_park(ATCVisPark *vis_park, ATCBox &bnd_box)
{
    int x0, x1, y0, y1;
    x0 = vis_park->grid_data[0];
    x1 = vis_park->grid_data[0];
    y0 = vis_park->grid_data[1];
    y1 = vis_park->grid_data[1];

    for (int i = 1; i < 4; i++)
    {
        if ((vis_park->grid_data[2 * i]) < x0)
            x0 = vis_park->grid_data[2 * i];
        if ((vis_park->grid_data[2 * i]) > x1)
            x1 = vis_park->grid_data[2 * i];
        if ((vis_park->grid_data[2 * i + 1]) < y0)
            y0 = vis_park->grid_data[2 * i + 1];
        if ((vis_park->grid_data[2 * i + 1]) > y1)
            y1 = vis_park->grid_data[2 * i + 1];
    }
    bnd_box.point1.x = x0;
    bnd_box.point1.y = y0;
    bnd_box.point2.x = x1;
    bnd_box.point2.y = y1;
    return;
}

void ATCParkTracker::test_compute_iou()
{
    ATCPark *park1 = new ATCPark;
    ATCPark *park2 = new ATCPark;
    park1->points_in_img[0] = 0;
    park1->points_in_img[1] = 0;
    park1->points_in_img[2] = 100;
    park1->points_in_img[3] = 0;
    park1->points_in_img[4] = 100;
    park1->points_in_img[5] = 50;
    park1->points_in_img[6] = 0;
    park1->points_in_img[7] = 50;

    park2->points_in_img[0] = 50;
    park2->points_in_img[1] = 0;
    park2->points_in_img[2] = 150;
    park2->points_in_img[3] = 0;
    park2->points_in_img[4] = 150;
    park2->points_in_img[5] = 50;
    park2->points_in_img[6] = 50;
    park2->points_in_img[7] = 50;
    //float res = get_iou(park1,park2);
    for (int x = 0; x < 120;)
    {
        park2->points_in_img[0] = 70;
        park2->points_in_img[1] = x;
        park2->points_in_img[2] = 150;
        park2->points_in_img[3] = x;
        park2->points_in_img[4] = 150;
        park2->points_in_img[5] = 50 + x;
        park2->points_in_img[6] = 70;
        park2->points_in_img[7] = 50 + x;
        float res = get_iou(park1, park2);
        std::cout << "iou test (x:" << x << ")" << res << std::endl;
        x += 10;
    }
    delete park1;
    delete park2;
}
// g
void ATCParkTracker::update_park_area(ATCPark *p_new_park)
{
    ATCVisPark vis_park;
    ATCBox bnd_box;
    ATCPoint pt;
    for (int i = 0; i < 8; i++)
    {
        vis_park.grid_data[i] = p_new_park->points_in_world[i];
    }
    get_box_from_vis_park(&vis_park, bnd_box);
    float area = 0;
    for (int i = bnd_box.point1.x; i <= bnd_box.point2.x;)
    {
        for (int j = bnd_box.point1.y; j <= bnd_box.point2.y;)
        {
            pt.x = i;
            pt.y = j;
            bool res = point_in_vis_park(&vis_park, pt);
            if (res)
            {
                area += 1;
            }
            j += m_iou_level * m_delta_x;
        }
        i += m_iou_level * m_delta_y;
    }
    p_new_park->area = area;
    return;
}

// clear m_ptrackers
void ATCParkTracker::clear()
{
    ATCPark *p_iter;
    std::vector<ATCPark *>::iterator it;
    for (it = m_ptrackers.begin(); it != m_ptrackers.end();)
    {
        p_iter = *it;
        unsigned int tracker_id = p_iter->id;
        it = pop(it);
        callback_tracker_id(tracker_id);
    }
    this->m_max_id = 0;
    std::cout << "clear trackers" << std::endl;
}


void ATCParkTracker::draw_park_img(cv::Mat &img)
{
    std::vector<ATCVisPark> vis_trks;
    get_vis_trackers(vis_trks);
    
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
    
    return;
}

