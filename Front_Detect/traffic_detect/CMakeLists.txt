cmake_minimum_required(VERSION 2.8.3)
project(traffic_detect)

set( TENGINE_LIBS_DIR ${PROJECT_SOURCE_DIR}/../../Tengine_Lib)
set( TENGINE_LIBS tengine)
message("***************************************************")
message("${TENGINE_LIBS_DIR}")
message("***************************************************")
## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)
add_definitions(-Wno-unused-command-line-argument)
add_definitions(-Wall)
add_definitions(-Werror)
add_definitions(-fPIC)
add_definitions(-g)
add_definitions(-O3)
add_definitions(-funroll-loops)
add_definitions(-Wno-overloaded-virtual)
add_definitions(-Wno-deprecated-register)
add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-std=c++11>)

catkin_package(
        CATKIN_DEPENDS cv_bridge
	image_transport
        roscpp
        std_msgs
	autoreg_msgs
	dashboard_msgs
)


find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  cv_bridge
  image_transport
  autoreg_msgs
  dashboard_msgs
)
find_package(OpenCV REQUIRED)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

link_directories( ${TENGINE_LIBS_DIR})

add_executable(traffic_detect src/traffic_detect.cpp src/status_machine.cpp)

target_link_libraries(traffic_detect ${TENGINE_LIBS} ${OpenCV_LIBS} ${catkin_LIBRARIES})

add_dependencies(traffic_detect ${catkin_EXPORTED_TARGETS})


