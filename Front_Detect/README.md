# Camera-ROS-Git
1.mobilenet_ssd base on tengine lib which in <Detect_Lib/tengine_lib>

2.parm in traffic_detect/launch/traffic_detect.launch
`
proto_file: CNN architecture based on Caffe.
model_file:  CNN weights.
roi_region:  set traffic detect ROI(x,y,width,height).
refresh_epoch: set status-machine update clock,low value,traffic light status update more frequently.
cam_cmd_topic: set dashboard camera cmd subscriber.
image_source_topic: set Front-view image topic subscriber.
status_code_topic: set traffic signals code publisher.
image_raw_topic: set visualized image publisher.
traffic_status_topic: set dashboard camera callback publisher.
traffic_active_topic: set process heartbeat publisher.

`
