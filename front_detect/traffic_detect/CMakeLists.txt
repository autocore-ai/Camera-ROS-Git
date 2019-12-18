cmake_minimum_required(VERSION 2.8.3)
project(traffic_detect)

message("***************************************************")
MESSAGE("PROJECT_SOURCE_DIR: " ${PROJECT_SOURCE_DIR})
message("***************************************************")

include("../../cmake.dir")

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
add_definitions(-Wno-unused-function)
add_definitions(-Wno-unused-variable)

catkin_package(
        CATKIN_DEPENDS cv_bridge
	image_transport
        roscpp
        std_msgs
        autoware_msgs
)


find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  cv_bridge
  image_transport
  autoware_msgs
)
find_package(OpenCV REQUIRED)


## Specify additional locations of header files
## Your package locations should be listed before other locations
MESSAGE("catkin_INCLUDE_DIRS: " ${catkin_INCLUDE_DIRS})
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)


#
set (MODEL_HELPER_DIR ${PROJECT_SOURCE_DIR}/../../model_helper)
MESSAGE("MODEL_HELPER_DIR: " ${MODEL_HELPER_DIR})
include_directories(${MODEL_HELPER_DIR})
link_directories( ${MODEL_HELPER_DIR})


add_executable(traffic_detect src/main.cpp src/Context.cpp  src/trafficlights_detector.cpp )
target_link_libraries(traffic_detect  model_helper tensorflow-lite ${OpenCV_LIBS} ${catkin_LIBRARIES} )
