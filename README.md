# Camera-ROS-Git
2018.09.04
-------
1.Front Detect is base on ROS

$ mkdir -p ~/catkin_ws/src
$ cd ~/catkin_ws/src
$ catkin_init_workspace
$ cd ..
$ catkin_make

--------
2.start traffic light detect

$ source devel/setup.bash
$ roslaunch traffic_detect traffic_detect.launch


--------
4.start vechile parks detect

$ source devel/setup.bash
$ roslaunch vps_detect vps_detect.launch

--------
5.start cameara driver

$ source devel/setup.bash
$ roslaunch opencv_ros opencv_ros.launch

