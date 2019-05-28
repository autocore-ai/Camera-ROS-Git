#include"auto_yulan_sdk.h"

void ATCMapper::init_yulan_sdk()
{
    m_center_x = m_width*0.5 -0.5;
    m_center_y = m_height*0.5 -0.5;
    m_pos.x = 0;
    m_pos.y = 0;
    m_pos.z = 0;
    m_ort.x = 0;
    m_ort.y = 0;
    m_ort.z = 0;
    m_ort.w = 0;
}

//
float ATCMapper::convert_location(float src_loc,float center,float delta)
{
    float dst_loc=0;
    dst_loc = (src_loc - center)*delta;
    return dst_loc;
}

//图片坐标系到车身坐标系
void ATCMapper::convert_to_carw(ATCPark* p_new_park)
{
   // this->m_atc_park= p_new_park;
    for(int i =0;i<4;i++)
    {
    	int x_idx = 2*i;
    	int y_idx = 2*i + 1;
        float x = p_new_park->points_in_img[x_idx];
    	float y = p_new_park->points_in_img[y_idx];
    	p_new_park->points_in_car[x_idx] = convert_location(x,this->m_center_x,this->m_delta_x);
        p_new_park->points_in_car[y_idx] = convert_location(y,this->m_center_y,this->m_delta_y);
    }
}

void ATCMapper::update(float dx,float dy,float img_width,float img_height,geometry_msgs::Point position,geometry_msgs::Quaternion orientation)
{
    m_delta_x = dx;
    m_delta_y = dy;
    m_width = img_width;
    m_height = img_height;
    init_yulan_sdk();
    m_pos = position;
    m_ort = orientation;
    init_tf_sdk();    
}

void ATCMapper::convert_to_vecmap(ATCPark* p_new_park)
{	
    convert_to_carw(p_new_park);
    float x_car,y_car,x,y,z;
    for(int i =0;i<4;i++ )
    {
    	x_car = p_new_park->points_in_car[2*i];
    	y_car = p_new_park->points_in_car[2*i+1];
	
    	// convet from img-cooridate to car-coordiate
    	x = -y_car;
        y = -x_car;
    	z = 0;
        
    	tf::Vector3 vec_in_car(x,y,z);
    	vec_in_car = m_tf*vec_in_car;	
    	p_new_park->points_in_world[2*i] = float(vec_in_car.x());
    	p_new_park->points_in_world[2*i+1] =float(vec_in_car.y());
    }	    
}

void ATCMapper::init_tf_sdk()
{
    tf::Vector3 vec_ori(m_pos.x,m_pos.y,m_pos.z);
    tf::Quaternion tfqt(m_ort.x,m_ort.y,m_ort.z,m_ort.w);
    m_tf.setOrigin(vec_ori);
    m_tf.setRotation(tfqt);
}
