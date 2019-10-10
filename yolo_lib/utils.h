#include <algorithm>
#include <iomanip>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <math.h>
#include <chrono>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;

#define CONF 0.85
#define NMS_THRESHOLD 0.05f

typedef struct {
    int w;
    int h;
    int c;
    float *data;
} image;

image load_image_cv(const cv::Mat& img);
image letterbox_image(image im, int w, int h);
void free_image(image m);
float sigmoid(float p);
float overlap(float x1, float w1, float x2, float w2);
float cal_iou(vector<float> box, vector<float>truth);
vector<vector<float>> applyNMS(vector<vector<float>>& boxes,int classes, const float thres);
void get_output(int8_t* dpuOut, int sizeOut, float scale, int oc, int oh, int ow, vector<float>& result);
float get_pixel(image m, int x, int y, int c);
void set_pixel(image m, int x, int y, int c, float val);
void add_pixel(image m, int x, int y, int c, float val);
image make_empty_image(int w, int h, int c);
void free_image(image m);
image make_image(int w, int h, int c);
void fill_image(image m, float s);
void embed_image(image source, image dest, int dx, int dy);
void ipl_into_image(IplImage* src, image im);
image ipl_to_image(IplImage* src);
void rgbgr_image(image im);
image resize_image(image im, int w, int h);
image load_image_cv(const cv::Mat& img);
image letterbox_image(image im, int w, int h);
void leftTrim(std::string& s);
void rightTrim(std::string& s);
string trim(std::string s);
std::string type2str(int type);
bool file_exists(const std::string fileName, bool verbose=true);



