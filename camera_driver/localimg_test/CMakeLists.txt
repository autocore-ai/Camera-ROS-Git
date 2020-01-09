cmake_minimum_required(VERSION 2.8.3)
project(localimg_test)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  image_transport
  roscpp
  sensor_msgs
  std_msgs
)
find_package(OpenCV REQUIRED)

###########
## Build ##
###########
add_executable(localimg_test src/localimg_test.cpp)
target_link_libraries(localimg_test
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
)

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)
