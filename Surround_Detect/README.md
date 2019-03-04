# Camera-ROS-Git
1.mobilenet_ssd base on tengine lib which in <Detect_Lib/tengine_lib>

2.parm in vps_detect/launch/vps_detect.launch
`
proto_file: CNN architecture based on Caffe.
model_file:  CNN weights.
cam_cmd_topic: set dashboard camera cmd subcriber.
image_source_topic: set Surround-view image topic subscriber.
image_pose_topic: set current vechile pose subscriber.
image_object_topic: set detected park publisher.
image_raw_topic: set visualized image publisher.
vps_status_topic: set dashboard camera callback publisher.
vps_active_topic: set process heartbeat publisher.

`
