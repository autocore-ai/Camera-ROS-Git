/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * License); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * AS IS BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*
 * Copyright (c) 2018, Open AI Lab
 * Author: chunyinglv@openailab.com
 */

#include <unistd.h>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <sstream>
#include <fstream>
#include <iomanip>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "tengine_c_api.h"
#include <sys/time.h>
#include "common.hpp"

#define DEF_PROTO "models/MobileNetSSD_deploy.prototxt"
#define DEF_MODEL "models/MobileNetSSD_deploy.caffemodel"
#define DEF_IMAGE "tests/images/ssd_dog.jpg"

using namespace std;
using namespace cv;

struct Box
{
    float x0;
    float y0;
    float x1;
    float y1;
    int class_idx;
    float score;
};
struct ParkAnchor
{
    //std::vector<cv::Point> list;// 0:ltop,1:rtop;2,rbot,3:lbot;
    //int num =0;//
    cv::Point ltop;
    cv::Point rtop;
    cv::Point rbot;
    cv::Point lbot;
};
float get_box_overlap_ratio(Box bbox1,Box bbox2);
bool overlap_park_supress(std::vector<Box> box_list,Box src_box,float iou_thresold =0.5); 
bool park_anchor_filter( ParkAnchor src_park,cv::Point center,float thresold,float max_lw_ratio,float min_lw_ratio=1.0);

//get iou ratio between two bndingbox
float get_box_overlap_ratio(Box bbox1,Box bbox2)
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

// supress detected free_park which is overlap with forbidden_park or incar_park over iou thresold
bool overlap_park_supperss(std::vector<Box> box_list,Box src_box,float iou_thresold/* =0.5*/)
{
    bool suppressed = false;
    if(iou_thresold<0|| iou_thresold >1)
        iou_thresold = 0.5;//default iou
    float iou;
    for(int idx =0;idx<box_list.size();++idx)
    {
        Box box = box_list[idx];
        if(1!=box.class_idx) // forbidden_park or incar_park
        {
            iou =  get_box_overlap_ratio(box,src_box);
            if(iou>iou_thresold)
            {
                suppressed = true;
                break;
            }
        }
    }
    
    //
    return suppressed;
  
}

// image angle expanding
void imrotate(cv::Mat& img,cv::Mat& newIm, double angle)
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

// get line' length  
float get_len(cv::Point begin,cv::Point end)
{
    float len =0;
    len =sqrt((begin.x-end.x)*(begin.x-end.x)+(begin.y-end.y)*(begin.y-end.y));
    return len;
}

// filter abnormal park anchors
bool park_anchor_filter(ParkAnchor src_park,cv::Point bbox_center,float thresold,float max_lw_ratio,float min_lw_ratio/*=1.0*/)
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
        std::cout<<"Warnning: center shift is out of thresold\n";
        return false;
    }
    return true;  
}

// get inputdata
void get_input_data_ssd(Mat& image_org, float* input_data, int img_h,  int img_w)
{
    cv::Mat image_input = image_org.clone();
    cv::resize(image_input, image_input, cv::Size(img_h, img_w));
    image_input.convertTo(image_input, CV_32FC3);
    float *img_data = (float *)image_input.data;
    int hw = img_h * img_w;

    float mean[3]={127.5,127.5,127.5};
    for (int h = 0; h < img_h; h++)
    {
        for (int w = 0; w < img_w; w++)
        {
            for (int c = 0; c < 3; c++)
            {
                input_data[c * hw + h * img_w + w] = 0.007843* (*img_data - mean[c]);
                img_data++;
            }
        }
    }
    std::cout<<"[GET IMAGE]:\t"<< "\n";
}

// get two line's crosspoint
cv::Point2f getCrossPoint(cv::Vec4i LineA, cv::Vec4i LineB)
{
    double ka, kb;
    ka = (double)(LineA[3] - LineA[1]) / (double)(LineA[2] - LineA[0]); 
    kb = (double)(LineB[3] - LineB[1]) / (double)(LineB[2] - LineB[0]); 

    cv::Point2f crossPoint;
    crossPoint.x = (ka*LineA[0] - LineA[1] - kb*LineB[0] + LineB[1]) / (ka - kb);
    crossPoint.y = (ka*kb*(LineA[0] - LineB[0]) + ka*LineB[1] - kb*LineA[1]) / (ka - kb);
    return crossPoint;
}
//
bool get_park_anchor(std::vector<cv::Point> anchor_list, ParkAnchor &dst_park,float offset)
{
    cv::Point v10,v32,v_merge;
    
    if(offset<0||offset>1.0)
        return false;
    if(4!=anchor_list.size())
        return false;
    // line( anchor_list[1],anchor_list[2]) //the maximun distance line
    v10.x = anchor_list[1].x-anchor_list[0].x;
    v10.y = anchor_list[1].y - anchor_list[0].y;
    v32.x = anchor_list[3].x -anchor_list[2].x;
    v32.y = anchor_list[3].y-anchor_list[2].y;
    
    if ((v10.x*v10.x)<(v32.x*v32.x))
    {   
        v_merge.x = ceil(v10.x+offset*(v32.x-v10.x));
        v_merge.y = ceil(v10.y+offset*(v32.y-v10.y));
    }
    else
    {
        v_merge.x = ceil(v32.x+offset*(v10.x-v32.x));
        v_merge.y = ceil(v32.y+offset*(v10.y-v32.y));
    }
    dst_park.ltop =cv::Point(anchor_list[0].x + v_merge.x,anchor_list[0].y+v_merge.y);
    dst_park.rtop =anchor_list[0];
    dst_park.rbot = anchor_list[2];
    dst_park.lbot =cv::Point(anchor_list[2].x + v_merge.x,anchor_list[2].y+v_merge.y);
    return true;
    
}

// remove image border edges
void depress_fringe_grad(cv::Mat src_img,cv::Mat &dst_img,int shrink)
{
    cv::Mat mid_img,mask;
    int dst_w,dst_h;

    //
    mask = cv::Mat::zeros(src_img.size(),src_img.type());
    dst_w = src_img.cols -2*shrink;
    dst_h = src_img.rows -2*shrink;
    if(shrink<0)
    {
        dst_img = mask;
        return;
    }
    if(dst_w<1||dst_h<1)// bad shrink
    {
        std::cout<<"Warnning: bad image shrink,please decrease shrink offset\n";
        dst_img = mask;
        return;
    }
    
    mask(cv::Rect(shrink-1,shrink-1,dst_w,dst_h)).setTo(255); 
    //imshow("mask",mask);
    
    src_img.copyTo(dst_img,mask);
    return;
}

// input park patch image,
void park_edge_detect(cv::Mat src_img,cv::Mat &dst_img)
{   
    cv::Mat mid_img,edge_img,depress_img,mask;
    cv::cvtColor(src_img,mid_img,CV_BGR2GRAY);
    
    //image enhance
   // cv::equalizeHist(mid_img,mid_img);
   // cv::imshow("gray",mid_img); 
    
    // canny operator
    //std::cout<<"try median filter!"<<std::endl;
    cv::Canny(mid_img, edge_img, 50, 200, 3);
    depress_fringe_grad(edge_img,depress_img,5);
    
    //get binary mask
   cv::equalizeHist(mid_img,mid_img);
    cv::threshold(mid_img,mask,180,255,CV_THRESH_BINARY);
    cv::Mat element = cv::getStructuringElement(MORPH_RECT, cv::Size(3, 3)); 
    // cv::erode(mask, mask, element);
    cv::dilate(mask,mask,element);
    cv::threshold(mask,mask,180,255,CV_THRESH_BINARY);

    //cv::imshow("bin_img",mask);    
    depress_img.copyTo(dst_img,mask);
    //cv::imshow("raw_edge",edge_img);
    //cv::imshow("dst_edge",dst_img);
    //cv::waitKey(100);


}
//expand_bndbox
void expand_bndbox(cv::Rect src_rect,cv::Rect &dst_rect,float ratio,int img_w,int img_h)
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

// insert non-repeat crosss over in cross_list
void insert_crossover(std::vector<cv::Point> &crossover_list,cv::Point src_point)
{
    //std::cout<<"insert crossover--------->";
    int merge_thresold =8;
    bool is_existed = false;
    for(int i= 0;i<crossover_list.size();i++)
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

// capture nonvertical-horizonl line's crossover within rect
void get_two_crossover_from_line_to_rect(float a,float b,float c,cv::Rect src_rect,std::vector<cv::Point> &crossover_list)
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

int idx_save =0;
void post_process_ssd(cv::Mat& image_input,float threshold,float* outdata,int num, cv::Mat& img)
{
    const char* class_names[] = {"background",
                            "free_park", "forbid_park", "in_park", "boat",
                            "bottle", "bus", "car", "cat", "chair",
                            "cow", "diningtable", "dog", "horse",
                            "motorbike", "person", "pottedplant",
                            "sheep", "sofa", "train", "tvmonitor"};

    img = image_input.clone();
    int raw_h = img.size().height;
    int raw_w = img.size().width;
    std::vector<Box> boxes;
    std::vector<Box> non_free_park_boxes;
    int line_width=raw_w*0.005;
    printf("detect ruesult num: %d \n",num);
    
    //add forbidden_park bbox and incar_park box 
    for (int i=0;i<num;i++)
    {
        if(outdata[1]>=threshold)
        {
            Box box;
            box.class_idx=outdata[0];
            box.score=outdata[1];
            box.x0=outdata[2]*raw_w;
            box.y0=outdata[3]*raw_h;
            box.x1=outdata[4]*raw_w;
            box.y1=outdata[5]*raw_h;
            //only keep park bndbox
            if((2==outdata[0])||(3==outdata[0]))
            {
                non_free_park_boxes.push_back(box);
            }
        }
        outdata+=6;
    }

    // back to outdata[0]
    outdata = outdata -6*num;
    //free park supress
    for (int i=0;i<num;i++)
    {
        if(outdata[1]>=threshold)
        {
            Box box;
            box.class_idx=outdata[0];
            box.score=outdata[1];
            box.x0=outdata[2]*raw_w;
            box.y0=outdata[3]*raw_h;
            box.x1=outdata[4]*raw_w;
            box.y1=outdata[5]*raw_h;
            
            if(1==outdata[0])
            {   
                bool ret = overlap_park_supperss(non_free_park_boxes,box,0.6);
                if(!ret) //non supressed
                {    boxes.push_back(box);
                    printf("%s\t:%.0f%%\n", class_names[box.class_idx], box.score * 100);
                    printf("BOX:( %g , %g ),( %g , %g )\n",box.x0,box.y0,box.x1,box.y1);
                }
            }
            else if((2==outdata[0])||(3==outdata[0]))
            {
                boxes.push_back(box);
                printf("%s\t:%.0f%%\n", class_names[box.class_idx], box.score * 100);
                printf("BOX:( %g , %g ),( %g , %g )\n",box.x0,box.y0,box.x1,box.y1);
            }
            else
            {
                ;//skip
            }
        }
        outdata +=6;
    }
    
    //finetune and map park anchor 
   // cv::namedWindow("demo_show",CV_WINDOW_NORMAL);
    for(int i =0;i<(int)boxes.size();i++)
    {
        cv::Mat roi_img, mid_img;
        cv::Rect expand_rect;
        Box box=boxes[i];
        bool is_valid =true;
        expand_bndbox(cv::Rect(box.x0,box.y0,(box.x1 - box.x0),(box.y1 - box.y0)),expand_rect,0.1,raw_w,raw_h);
        //draw park bnding box;
        cv::rectangle(img, expand_rect,cv::Scalar(255, 0, 0),line_width);

        // --- twice line detect -----
       // std::cout<<"get roi image:"<<expand_rect.x<<":"<<expand_rect.y<<":"<<":"<<(expand_rect.x+expand_rect.width)<<":"<<(expand_rect.y+expand_rect.height)<<raw_w<<":"<<raw_h<<std::endl;
        roi_img=img(expand_rect);
        //string str_idx=to_string(idx_save);
        //imwrite("temp/"+str_idx+".jpg",roi_img);
        //cv::rectangle(img, cv::Rect(box.x0, box.y0,(box.x1-box.x0),(box.y1-box.y0)),cv::Scalar(255, 0, 0),line_width);

        //cv::imwrite();
       // idx_save +=1;
        //cv::Canny(roi_img, mid_img, 50, 200, 3);
        park_edge_detect(roi_img,mid_img);
        std::vector<cv::Vec4i> lines;
        cv::HoughLinesP(mid_img, lines, 1, CV_PI/180, 20, 20, 10 );
        if(1>lines.size())
        {
            std::cout<<"warnning can not detect lines\n";
            continue;
        }

        int line_idx = 0;
        int max_line =0;
        for(int j =0;j<lines.size();j++)
        {
            cv::Vec4i l = lines[j];
            int line_len =(l[2]-l[0])*(l[2]-l[0])+(l[3]-l[1])*(l[3]-l[1]);
            //find the maximun length line
            if(line_len>max_line)
            {
                max_line = line_len;
                line_idx = j;
            }
            //std::cout<<j<<std::endl;
        }

        cv::Vec4i result=lines[line_idx]; 
        cv::Point begin(result[0],result[1]);
        cv::Point center(expand_rect.width/2,expand_rect.height/2); 
        cv::Point end(result[2],result[3]);// ax+by+c =0;
        ParkAnchor draw_anchor;
       // cv::line( roi_img, begin, end,cv::Scalar(0,255,255), 5, CV_AA);

        float a,b,c;

        if(begin.x == end.x)//vectical line
        {
            if(begin.x<center.x)// |*
            {
                draw_anchor.ltop=cv::Point(begin.x,0);
                draw_anchor.rtop=cv::Point(expand_rect.width-1,0);
                draw_anchor.rbot=cv::Point(expand_rect.width-1,expand_rect.height-1);
                draw_anchor.lbot=cv::Point(begin.x,expand_rect.height-1);
            }
            else // *|
            {
                draw_anchor.ltop=cv::Point(0,0);
                draw_anchor.rtop=cv::Point(begin.x,0);
                draw_anchor.rbot=cv::Point(begin.x,expand_rect.height-1);
                draw_anchor.lbot=cv::Point(0,expand_rect.height-1);
            }         
        }
        else if(begin.y == end.y)// horizonal line
        {
            if(begin.y<center.y)//V
            {
                draw_anchor.ltop=cv::Point(0,begin.y);
                draw_anchor.rtop=cv::Point(expand_rect.width-1,begin.y);
                draw_anchor.rbot=cv::Point(expand_rect.width-1,expand_rect.height -1);
                draw_anchor.lbot=cv::Point(0,expand_rect.height-1);
            }else//^
            {   
                //std::cout<<"please fix bug in horizonal"<<std::endl;
                draw_anchor.ltop = cv::Point(0,0);
                draw_anchor.rtop = cv::Point(expand_rect.width-1,0);
                draw_anchor.rbot = cv::Point(expand_rect.width -1,begin.y);
                draw_anchor.lbot = cv::Point(0,begin.y);
            }
        }
        else //non-vertical and non-horizonal line
        {   
            // ax+by+c =0
            b =1.0;
            a = -(end.y-begin.y)*1.0/(end.x-begin.x);
            c = -a*begin.x - begin.y;
            std::vector<cv::Point> line_crossover;
            std::cout<<"long line:"<<begin.x<<":"<<begin.y<<","<<end.x<<":"<<end.y<<"---->"<<a<<","<<b<<","<<c<<std::endl;
            //
            // get two crossover between maximun detected line with dst rect
            get_two_crossover_from_line_to_rect(a,b, c,expand_rect,line_crossover);
            if(2!=line_crossover.size())
            {
                std::cout<<"Warnning:can not get two crossover vs("<<line_crossover.size()<<"between maximun detected line with ROI rect \n";
                is_valid = false;
                continue;
            }
            
            //get perpendicular line two crossover
            std::vector<cv::Point> rect_crossover;
            for(int idx=0;idx<line_crossover.size();idx++)
            {
                cv::Point crossover_point = line_crossover[idx];
                float p_a,p_b,p_c;//p_a*x+p_b*y+p_c=0
                p_a =-1/a;
                p_b = 1.0;
                p_c = -(crossover_point.y)-p_a*(crossover_point.x);
                                
                std::vector<cv::Point> next_crossover;
                next_crossover.push_back(crossover_point);
                get_two_crossover_from_line_to_rect(p_a,p_b,p_c,expand_rect,next_crossover);
                if(2!=next_crossover.size())
                {
                    std::cout<<"Warnning:can not get two crossover vs("<<next_crossover.size()<<" in perpendicular line \n";
                    is_valid =false;
                    continue;   
                }
                for(int pline_idx=0;pline_idx<next_crossover.size();pline_idx++)
                {
                    rect_crossover.push_back(next_crossover[pline_idx]);
                }
            }

            // transform 4 crossover into ParkAnchor
            if(4==rect_crossover.size())
            {
                get_park_anchor( rect_crossover,draw_anchor,1);
                
            }
            else
            {
                is_valid =false;
            }
           
        }
        //cv::line( roi_img,test1,test2,cv::Scalar(255,0,0),1,CV_AA);
        // drop unvalid anchor detecti 
        if(!is_valid)
        {
            //std::cout<<"waring unvalid draw_anchor\n";
            continue;
        }

        if(!park_anchor_filter(draw_anchor, center,20,4.5,1.2))
        {
            std::cout<<"bad detected park,skip......\n";
            //draw red_box
            draw_anchor.ltop.x += expand_rect.x;
            draw_anchor.ltop.y += expand_rect.y;
            draw_anchor.rtop.x += expand_rect.x;
            draw_anchor.rtop.y += expand_rect.y;
            draw_anchor.rbot.x += expand_rect.x;
            draw_anchor.rbot.y += expand_rect.y;
            draw_anchor.lbot.x += expand_rect.x;
            draw_anchor.lbot.y += expand_rect.y;
            
            //bad lw-ratio or center-shifted park,drop this detected park
            
            begin.x +=expand_rect.x;
            begin.y +=expand_rect.y;
            end.x +=expand_rect.x;
            end.y +=expand_rect.y;

            /*cv::line( img, begin, end,cv::Scalar(0,255,255),5, CV_AA);
            cv::line( img, draw_anchor.ltop, draw_anchor.rtop,cv::Scalar(0,0,255), 2, CV_AA);
            cv::line( img, draw_anchor.rtop, draw_anchor.rbot,cv::Scalar(0,0,255),2, CV_AA);
            cv::line( img, draw_anchor.rbot, draw_anchor.lbot,cv::Scalar(0,0,255), 2, CV_AA);
            cv::line( img, draw_anchor.lbot, draw_anchor.ltop,cv::Scalar(0,0,255), 2, CV_AA);
            */
            continue;
        }

        draw_anchor.ltop.x += expand_rect.x;
        draw_anchor.ltop.y += expand_rect.y;
        draw_anchor.rtop.x += expand_rect.x;
        draw_anchor.rtop.y += expand_rect.y;
        draw_anchor.rbot.x += expand_rect.x;
        draw_anchor.rbot.y += expand_rect.y;
        draw_anchor.lbot.x += expand_rect.x;
        draw_anchor.lbot.y += expand_rect.y;
        
        //bad lw-ratio or center-shifted park,drop this detected park

        begin.x +=expand_rect.x;
        begin.y +=expand_rect.y;
        end.x +=expand_rect.x;
        end.y +=expand_rect.y;
        //cv::line( img, begin, end,cv::Scalar(0,255,255),5, CV_AA);

        //std::cout<<"ltop"<<draw_anchor.ltop.x<<":"<<draw_anchor.ltop.y<<","<<draw_anchor.rtop.x<<":"<<draw_anchor.rtop.y<<","<<draw_anchor.rbot.x<<":"<<draw_anchor.rbot.y<<","<<draw_anchor.lbot.x<<":"<<draw_anchor.lbot.y<<std::endl;
        cv::line( img, draw_anchor.ltop, draw_anchor.rtop,cv::Scalar(0,255,0), 2, CV_AA);
        cv::line( img, draw_anchor.rtop, draw_anchor.rbot,cv::Scalar(0,255,0),2, CV_AA);
        cv::line( img, draw_anchor.rbot, draw_anchor.lbot,cv::Scalar(0,255,0), 2, CV_AA);
        cv::line( img, draw_anchor.lbot, draw_anchor.ltop,cv::Scalar(0,255,0), 2, CV_AA);

        //------------
        std::ostringstream score_str;
        score_str<<box.score;
        std::string label = std::string(class_names[box.class_idx]) + ": " + score_str.str();
        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        cv::rectangle(img, cv::Rect(cv::Point(box.x0,box.y0- label_size.height),cv::Size(label_size.width, label_size.height + baseLine)),cv::Scalar(255, 255, 0), CV_FILLED);
        cv::putText(img, label, cv::Point(box.x0, box.y0),cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));     
        //imwrite("temp/"+str_idx+".jpg",img);
        //idx_save +=1;

    } 
      // imwrite("temp/"+str_idx+".jpg",img);
       //idx_save +=1;
       //: cv::imshow("demo_show",roi_img);

   
    //std::cout<<"======================================\n";
    //std::cout<<"[DETECTED IMAGE SAVED]:\t"<< save_name<<"\n";
    //std::cout<<"======================================\n";
}

int main(int argc, char *argv[])
{
    const std::string root_path = get_root_path();
    std::string proto_file;
    std::string model_file;
    std::string image_file;
    std::string video_source;
    std::string save_name="save.jpg";
    //std::string txt_source;
    char c;

    int res;
    while( ( res=getopt(argc,argv,"p:m:i:h"))!= -1)
    {
        switch(res)
        {
            case 'p':
                proto_file=optarg;
                break;
            case 'm':
                model_file=optarg;
                break;
            case 'i':
                image_file=optarg;
                break;
            case 'v':
                video_source=optarg;
            case 'h':
                std::cout << "[Usage]: " << argv[0] << " [-h]\n"
                          << "   [-p proto_file] [-m model_file] [-i image_file] [-v video_source] \n";
                return 0;
            default:
                break;
        }
    }



    const char *model_name = "mssd_300";
    if(proto_file.empty())
    {
        proto_file = root_path + DEF_PROTO;
        std::cout<< "proto file not specified,using "<<proto_file<< " by default\n";

    }
    if(model_file.empty())
    {
        model_file = root_path + DEF_MODEL;
        std::cout<< "model file not specified,using "<<model_file<< " by default\n";
    }
    if(video_source.empty())
    {
        video_source = "/dev/video1";
        //txt_source = "/home/yang/data/park_test/park_test.txt";
        std::cout<< "image file not specified,using "<<image_file<< " by default\n";
    }

    // init tengine
    init_tengine_library();
    if (request_tengine_version("0.1") < 0)
        return 1;
    if (load_model(model_name, "caffe", proto_file.c_str(), model_file.c_str()) < 0)
        return 1;
    std::cout << "load model done!\n";
   
    // create graph
    graph_t graph = create_runtime_graph("graph", model_name, NULL);
    if (!check_graph_valid(graph))
    {
        std::cout << "create graph0 failed\n";
        return 1;
    }

    // init video
    VideoCapture captRefrnc(video_source);
    if (!captRefrnc.isOpened())
    {
        cout  << "Could not open reference " << video_source << endl;
        return -1;
    }
    Size refS = Size((int) captRefrnc.get(CV_CAP_PROP_FRAME_WIDTH),
                     (int) captRefrnc.get(CV_CAP_PROP_FRAME_HEIGHT));
    
    // init windows
    const char* WIN_UT = "SSD-MobileNet";
    namedWindow(WIN_UT, CV_WINDOW_AUTOSIZE);
    cvMoveWindow(WIN_UT, refS.width, 0);

    // cv frame
    Mat frame_input, frame_show;

    // input
    int img_h = 300;
    int img_w = 300;
    int img_size = img_h * img_w * 3;
    float *input_data = (float *)malloc(sizeof(float) * img_size);

    int node_idx=0;
    int tensor_idx=0;
    tensor_t input_tensor = get_graph_input_tensor(graph, node_idx, tensor_idx);
    if(!check_tensor_valid(input_tensor))
    {
        printf("Get input node failed : node_idx: %d, tensor_idx: %d\n",node_idx,tensor_idx);
        return 1;
    }

    int dims[] = {1, 3, img_h, img_w};
    set_tensor_shape(input_tensor, dims, 4);
    prerun_graph(graph);

    struct timeval t0, t1;
    float total_time = 0.f;
    
    cv::Mat src_img;
    for(;;)
    {
    	//get camera frame
        captRefrnc >> src_img;
	    if (src_img.empty())
        {
            cout << " < < <  Game over!  > > > ";
            break;
        }
        imrotate(src_img, frame_input,0);
        get_input_data_ssd(frame_input, input_data, img_h,  img_w);
  
        //gettimeofday(&t0, NULL);
        set_tensor_buffer(input_tensor, input_data, img_size * 4);
        run_graph(graph, 1);

        //gettimeofday(&t1, NULL);
        //float mytime = (float)((t1.tv_sec * 1000000 + t1.tv_usec) - (t0.tv_sec * 1000000 + t0.tv_usec)) / 1000;
        //total_time += mytime;

    	//std::cout << "repeat " << repeat_count << " times, avg time per run is " << total_time << " ms\n";
    
    	tensor_t out_tensor = get_graph_output_tensor(graph, 0,0);//"detection_out");
    	int out_dim[4];
    	get_tensor_shape( out_tensor, out_dim, 4);
    	float *outdata = (float *)get_tensor_buffer(out_tensor);
    	int num=out_dim[1];
    	float show_threshold=0.3;
    
    	post_process_ssd(frame_input, show_threshold, outdata, num, frame_show);
	    //string save_img_path = dst_path + img_name;
        //std::cout<<"image save path:"<<save_img_path<<std::endl;
        //cv::imwrite(save_img_path,frame_show);
    	cv::imshow(WIN_UT, frame_show);
        c = (char)cvWaitKey(15);
	if (c == 27) break;
    }

    postrun_graph(graph);
    free(input_data);
    destroy_runtime_graph(graph);
    remove_model(model_name);

    return 0;
}
