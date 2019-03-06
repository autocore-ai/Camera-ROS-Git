cmake_minimum_required(VERSION 2.8.3)
project(vps_detect)

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
        roscpp
        std_msgs
        autoreg_msgs
)


find_package(catkin REQUIRED COMPONENTS
  autoreg_msgs
  roscpp
  std_msgs
  cv_bridge
  tf
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

add_executable(vps_detect src/vps_detect.cpp src/auto_yulan_sdk.cpp src/auto_tracker_sdk.cpp)

target_link_libraries(vps_detect ${TENGINE_LIBS} ${OpenCV_LIBS} ${catkin_LIBRARIES})

add_dependencies(vps_detect ${catkin_EXPORTED_TARGETS})


