#include "trafficlights_detector.h"
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include "std_msgs/UInt8.h"
#include <std_msgs/Int32.h>
#include <std_msgs/Int32MultiArray.h>
#include <chrono>

TrafficLightsDetector::TrafficLightsDetector()
{
}

TrafficLightsDetector::~TrafficLightsDetector()
{
}

void TrafficLightsDetector::set_current_frame(cv::Mat frame)
{
    frame_ = frame;
}

void TrafficLightsDetector::init(int argc, char **argv)
{
    init_ros(argc, argv);

    mv1_.init(model_path_);
}

bool TrafficLightsDetector::init_ros(int argc, char **argv)
{
    ros::init(argc, argv, "detect");
    ros::NodeHandle node;

    bool ret = load_parameters();

    image_raw_suber_ = node.subscribe(image_source_topic_, 1, &TrafficLightsDetector::on_recv_frame, this);
    roi_signal_suber_ = node.subscribe("/roi_signal", 1, &TrafficLightsDetector::on_recv_signal_roi, this);

    signal_state_puber_ = node.advertise<autoware_msgs::TrafficLight>("light_color", 1);
    image_detected_puber_ = node.advertise<sensor_msgs::Image>(image_detected_topic_, 1);

    return ret;
}

void TrafficLightsDetector::on_recv_frame(const sensor_msgs::Image &image_source)
{
    ROS_INFO("traffic_detect:image_callback!!!!!!!");
    cv_bridge::CvImagePtr cv_image = cv_bridge::toCvCopy(image_source, "bgr8");
    cv::Mat frame = cv_image->image;
    set_current_frame(frame);

    frame_header_ = image_source.header;

    //process_frame();
}

void TrafficLightsDetector::on_recv_signal_roi(const autoware_msgs::Signals::ConstPtr &extracted_pos)
{
    static ros::Time previous_timestamp;
    // If frame has not been prepared, abort this callback
    if (frame_.empty() ||
        frame_header_.stamp == previous_timestamp)
    {
        std::cout << "No Image" << std::endl;
        return;
    }

    // Acquire signal posotion on the image
    Context::SetContexts(contexts_, extracted_pos, frame_.rows, frame_.cols);

    // Recognize the color of the traffic light
    for (Context &context : contexts_)
    {
        // for (unsigned int i = 0; i < contexts_.size(); i++) {
        //   Context& context = contexts_.at(i);
        if (context.topLeft.x > context.botRight.x)
        {
            continue;
        }

        //std::cout << "roi inside: " << cv::Rect(context.topLeft, context.botRight) << std::endl;
        // extract region of interest from input image
        cv::Mat roi = frame_(cv::Rect(context.topLeft, context.botRight)).clone();

        //cv::imshow("ssd_tlr", roi);
        //  cv::waitKey(200);

        preprocess_frame(roi);
        std::vector<uint8_t> input;
        int width = mv1_.get_width();
        int height = mv1_.get_height();
        if (frame_model_input_.isContinuous())
        {
            input.insert(input.end(), frame_model_input_.data, frame_model_input_.data + width * height * 3);
        }
        int cls_id = mv1_.inference(input);
        lights_color_ = static_cast<LightColor>(cls_id);

        /*
        // Get current state of traffic light from current frame
        LightState current_state = recognizer.RecognizeLightState(roi);

        // Determine the final state by referring previous state
        context.lightState = DetermineState(context.lightState, // previous state
                                            current_state,      // current state
                                            &(context.stateJudgeCount)); // counter to record how many times does state recognized
*/
    }

    // Save timestamp of this frame so that same frame has never been process again
    previous_timestamp = frame_header_.stamp;
}

bool TrafficLightsDetector::load_parameters()
{
    ros::NodeHandle private_nh("~");

    private_nh.param<std::string>("image_source_topic", image_source_topic_, "/image_raw");
    ROS_INFO("Setting image_source_topic to %s", image_source_topic_.c_str());

    private_nh.param<std::string>("status_code_topic", signal_state_topic_, "");
    ROS_INFO("Setting signal_state_topic to %s", signal_state_topic_.c_str());

    private_nh.param<std::string>("image_detected_topic", image_detected_topic_, "");
    ROS_INFO("Setting image_detected_topic to %s", image_detected_topic_.c_str());

    private_nh.param<std::string>("model_path", model_path_, "");
    ROS_INFO("Setting model_path to %s", model_path_.c_str());

    return true;
}

unsigned char TrafficLightsDetector::status_encode()
{
    unsigned char code = 4;

    if (lights_color_ == LightColor::red)
    {
        code = 0;
    }
    else if (lights_color_ == LightColor::yellow || lights_color_ == LightColor::green)
    {
        code = 6;
    }

    cout << "code=" << (int)code << endl;

    return code;
}

void TrafficLightsDetector::preprocess_frame(const cv::Mat &frame)
{
    int width = mv1_.get_width();
    int height = mv1_.get_height();
    //cv::cvtColor(frame_, frame_, CV_RGB2BGR);
    cv::resize(frame, frame_model_input_, cv::Size(width, height));
    cv::cvtColor(frame_model_input_, frame_model_input_, CV_RGB2BGR);

    ROS_INFO("resize img to %d x %d", width, height);
}

//处理收到的待检测帧
void TrafficLightsDetector::process_frame()
{
    auto begin = std::chrono::system_clock::now();

    preprocess_frame(frame_);
    std::vector<uint8_t> input;
    int width = mv1_.get_width();
    int height = mv1_.get_height();
    if (frame_model_input_.isContinuous())
    {
        input.insert(input.end(), frame_model_input_.data, frame_model_input_.data + width * height * 3);
    }
    int cls_id = mv1_.inference(input);
    lights_color_ = static_cast<LightColor>(cls_id);

    auto end = std::chrono::system_clock::now();
    auto elsp = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
    std::cout << "process_frame:" << elsp.count() << std::endl;
}

void TrafficLightsDetector::publish_traffic_light(std::vector<Context> contexts)
{
    autoware_msgs::TrafficLight topic;
    static int32_t previous_state = kTrafficLightUnknown;
    topic.traffic_light = kTrafficLightUnknown;
    for (const auto ctx : contexts)
    {
        switch (ctx.lightState)
        {
        case GREEN:
            topic.traffic_light = kTrafficLightGreen;
            break;
        case YELLOW:
        case RED:
            topic.traffic_light = kTrafficLightRed;
            break;
        case UNDEFINED:
            topic.traffic_light = kTrafficLightUnknown;
            break;
        }

        // Publish the first state in contexts,
        // which has largest estimated radius of signal.
        // This program assume that the signal which has the largest estimated radius
        // equal the nearest one from camera.
        if (topic.traffic_light != kTrafficLightUnknown)
        {
            break;
        }
    }

    // If state changes from previous one, publish it
    if (topic.traffic_light != previous_state)
    {
        signal_state_puber_.publish(topic);
        previous_state = topic.traffic_light;
    }
}
