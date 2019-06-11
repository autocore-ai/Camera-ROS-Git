#ifndef AUTO_PARK_H_H_H
#define AUTO_PARK_H_H_H
struct ATCPark
{
    
    float points_in_world[8];//(x0,y0,x1,y1,x2,y2,x3,y3)
    
    float points_in_img[8];//(x0,y0,x1,y1,x2,y2,x3,y4)
    float conf_score=0;
    unsigned int id = 0;
    unsigned int clock =0;
    unsigned int count =0;
    unsigned int cls_id = 0;
    float area;
};

/*
struct ATCVisPark
{
    int grid_data[8];// x0,y0,x1,y1,x2,y2,x3,y3
    float conf_score=0;
    unsigned int id=0;
    //unsigned int clock;    
    unsigned int cls_id=0;
};

struct ATCPubPark
{
    float grid_data[8];//x0,y0,x1,y1,x2,y2,x3,y3
    float conf_score =0;
    unsigned int id=0;
    //unsigned int clock;    
    unsigned int cls_id=0;
};
*/
#endif
