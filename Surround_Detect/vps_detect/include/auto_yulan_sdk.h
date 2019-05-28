#ifndef AUTO_YULAN_SDK_H_H_H
#define AUTO_YULAN_SDK_H_H_H

#include "auto_park.h"
#include "auto_transform.h"

// dx,dy,w,h=21.9,20.1,1080,810
/*struct CarPose
{
    double x=0;
    double y =0;
    double z=0;
    float angle1;
    float angle2;
    float angle3;
};
*/
class ATCMapper
{
public:
    ATCMapper()
    {
        m_delta_x = 0.0219;
        m_delta_y = 0.0201;
        m_width = 810;
        m_height = 1080;
        init_yulan_sdk();
	    init_tf_sdk();
    }
    
    ATCMapper(float dx,float dy,float img_width,float img_height)
    {
        m_delta_x = dx;
        m_delta_y = dy;
        m_width = img_width;
        m_height = img_height;	
        init_yulan_sdk();
	    init_tf_sdk();
    }

    void update(float dx,float dy,float img_width,float img_height,geometry_msgs::Point position,geometry_msgs::Quaternion orientation);

    void convert_to_vecmap(ATCPark *p_new_park);
private:
    void convert_to_carw(ATCPark* p_new_park);
    
    inline float get_width()
    {
        return m_width;
    }

    inline float get_height()
    {
        return m_height;
    }    
private:
    void init_yulan_sdk();
    void init_tf_sdk();
    float convert_location(float src_loc,float center,float delta);
    //float convert_simu_location();
    
private:
    float m_delta_x;
    float m_delta_y;
    float m_width;
    float m_height;
    float m_center_x;
    float m_center_y;
    tf::Transform m_tf;
    geometry_msgs::PoseStamped m_pose_stamp;
    geometry_msgs::Point m_pos;
    geometry_msgs::Quaternion m_ort;
    //ATCPark *m_atc_park =NULL;
};
#endif
