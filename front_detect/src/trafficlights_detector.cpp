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

void TrafficLightsDetector::init()
{
    init_ros();

    mv1_.init(model_path_);
}

bool TrafficLightsDetector::init_ros()
{
    //ros::init(argc, argv, "detect");
    ros::NodeHandle node;

    bool ret = load_parameters();

    image_raw_suber_ = node.subscribe(image_source_topic_, 1, &TrafficLightsDetector::on_recv_frame, this);
    roi_signal_suber_ = node.subscribe("/roi_signal", 1, &TrafficLightsDetector::on_recv_signal_roi, this);

    signal_state_puber_ = node.advertise<autoware_msgs::TrafficLight>("light_color", 1);
    image_detected_puber_ = node.advertise<sensor_msgs::Image>(image_detected_topic_, 1);

    return ret;
}

bool TrafficLightsDetector::load_parameters()
{
    ros::NodeHandle private_nh("~");

    ROS_INFO("****TrafficLightsDetector params****");

    private_nh.param<std::string>("tl_image_source_topic", image_source_topic_, "/image_raw");
    ROS_INFO("Setting image_source_topic to %s", image_source_topic_.c_str());

    private_nh.param<std::string>("tl_image_detected_topic", image_detected_topic_, "");
    ROS_INFO("Setting image_detected_topic to %s", image_detected_topic_.c_str());

    private_nh.param<std::string>("tl_model_path", model_path_, "");
    ROS_INFO("Setting model_path to %s", model_path_.c_str());

    private_nh.param<int>("change_state_threshold", change_state_threshold_, 5);
	ROS_INFO("change_state_threshold: %d", change_state_threshold_);

    return true;
}

void TrafficLightsDetector::on_recv_frame(const sensor_msgs::Image &image_source)
{
    //ROS_INFO("traffic_detect:image_callback!!!!!!!");
    cv_bridge::CvImagePtr cv_image = cv_bridge::toCvCopy(image_source, "bgr8");
    cv::Mat frame = cv_image->image;
    set_current_frame(frame);

    frame_header_ = image_source.header;
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

        preprocess_frame(roi);
        std::vector<uint8_t> input;
        int width = mv1_.get_width();
        int height = mv1_.get_height();
        if (frame_model_input_.isContinuous())
        {
            input.insert(input.end(), frame_model_input_.data, frame_model_input_.data + width * height * 3);
        }
        int cls_id = mv1_.inference(input);
        
        LightState current_state = static_cast<LightState>(cls_id);

        determin_state(current_state, context);
    }

    publish_traffic_light(contexts_);
    publish_image(contexts_);

    // Save timestamp of this frame so that same frame has never been process again
    previous_timestamp = frame_header_.stamp;
}

void TrafficLightsDetector::preprocess_frame(const cv::Mat &frame)
{
    int width = mv1_.get_width();
    int height = mv1_.get_height();
    cv::resize(frame, frame_model_input_, cv::Size(width, height));
    cv::cvtColor(frame_model_input_, frame_model_input_, CV_RGB2BGR);

    ROS_INFO("resize img to %d x %d", width, height);
}

void TrafficLightsDetector::determin_state(LightState in_current_state,
                                           Context& in_out_signal_context)
{
	//if reported state by classifier is different than the previously stored
	if (in_current_state != in_out_signal_context.lightState)
	{
		//and also different from the previous difference
		if (in_current_state != in_out_signal_context.newCandidateLightState)
		{
			//set classifier result as a candidate
			in_out_signal_context.newCandidateLightState = in_current_state;
			in_out_signal_context.stateJudgeCount = 0;
		}
		else
		{
			//if classifier returned the same result previously increase its confidence
			in_out_signal_context.stateJudgeCount++;
		}
	}
	//if new candidate has been found enough times, change state to the new candidate
	if (in_out_signal_context.stateJudgeCount >= change_state_threshold_)
	{
		in_out_signal_context.lightState = in_current_state;
	}

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

void TrafficLightsDetector::publish_image(std::vector<Context> contexts)
{
    // Copy the frame image for output
	cv::Mat result_image = frame_.clone();

	// Define information for written label
	std::string label;
	const int kFontFace = cv::FONT_HERSHEY_COMPLEX_SMALL;
	const double kFontScale = 0.8;
	int font_baseline = 0;
	CvScalar label_color;

    for (const auto ctx: contexts_)
	{
		// Draw superimpose result on image
		circle(result_image, ctx.redCenter, ctx.lampRadius, CV_RGB(255, 0, 0), 1, 0);
		circle(result_image, ctx.yellowCenter, ctx.lampRadius, CV_RGB(255, 255, 0), 1, 0);
		circle(result_image, ctx.greenCenter, ctx.lampRadius, CV_RGB(0, 255, 0), 1, 0);

		// Draw recognition result on image
		switch (ctx.lightState)
		{
			case GREEN:
				label = "GREEN";
				label_color = CV_RGB(0, 255, 0);
				break;
			case YELLOW:
				label = "YELLOW";
				label_color = CV_RGB(255, 255, 0);
				break;
			case RED:
				label = "RED";
				label_color = CV_RGB(255, 0, 0);
				break;
			case UNDEFINED:
				label = "UNKNOWN";
				label_color = CV_RGB(0, 0, 0);
		}

		if (ctx.leftTurnSignal)
		{
			label += " LEFT";
		}
		if (ctx.rightTurnSignal)
		{
			label += " RIGHT";
		}
		//add lane # text
		label += " " + std::to_string(ctx.closestLaneId);

		cv::Point label_origin = cv::Point(ctx.topLeft.x, ctx.botRight.y + font_baseline);

		cv::putText(result_image, label, label_origin, kFontFace, kFontScale, label_color);
	}

    // Publish superimpose result image
    cv_bridge::CvImage converter;
    converter.header = frame_header_;
    converter.encoding = sensor_msgs::image_encodings::BGR8;
    converter.image = result_image;
    image_detected_puber_.publish(converter.toImageMsg());
}
