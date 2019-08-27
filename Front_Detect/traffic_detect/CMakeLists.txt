cmake_minimum_required(VERSION 2.8.3)
project(traffic_detect)


set( CUDA_LIBS_DIR /usr/local/cuda/lib64)
link_directories( ${CUDA_LIBS_DIR})

#link_directories(/home/nano/suchang/devel/lib/)

message("***************************************************")
MESSAGE("PROJECT_SOURCE_DIR: " ${PROJECT_SOURCE_DIR})
message("***************************************************")

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
add_definitions(-Wno-unused-function)
add_definitions(-Wno-unused-variable)

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
MESSAGE("catkin_INCLUDE_DIRS: " ${catkin_INCLUDE_DIRS})
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)


#link_directories( ${TENGINE_LIBS_DIR})

#add_executable(traffic_detect src/traffic_detect.cpp src/status_machine.cpp)

#target_link_libraries(traffic_detect ${TENGINE_LIBS} ${OpenCV_LIBS} ${catkin_LIBRARIES})

#add_dependencies(traffic_detect ${catkin_EXPORTED_TARGETS})


####### for jetson
#find_package(jetson-utils)
#find_package(jetson-inference)

# CUDA and Qt4 are required
#find_package(CUDA)
find_package(Qt4)

# setup Qt4 for build
include(${QT_USE_FILE})
add_definitions(${QT_DEFINITIONS})

# Add yolo lib as subdir
#add_subdirectory(../../yolo_lib yolo_lib.out)

# yolo lib
set (YOLO_DIR ${PROJECT_SOURCE_DIR}/../../yolo_lib)
MESSAGE("YOLO_DIR: " ${YOLO_DIR})
include_directories(${YOLO_DIR})
link_directories( ${YOLO_DIR})

#compile
add_executable(traffic_detect src/traffic_detect_jetson.cpp src/status_machine.cpp  src/trafficlights_detector.cpp)

#link
target_link_libraries(traffic_detect  yolo-lib ${OpenCV_LIBS} ${catkin_LIBRARIES} )

#
#add_dependencies(traffic_detect ${catkin_EXPORTED_TARGETS})


