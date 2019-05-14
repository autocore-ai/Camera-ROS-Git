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
include_directories(
  include
  yolo_lib
  ${catkin_INCLUDE_DIRS}
)

#link_directories( ${TENGINE_LIBS_DIR})

#add_executable(traffic_detect src/traffic_detect.cpp src/status_machine.cpp)

#target_link_libraries(traffic_detect ${TENGINE_LIBS} ${OpenCV_LIBS} ${catkin_LIBRARIES})

#add_dependencies(traffic_detect ${catkin_EXPORTED_TARGETS})


####### for jetson
find_package(jetson-utils)
find_package(jetson-inference)

# CUDA and Qt4 are required
message("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!find cuda!!!!!!!!!!!!!!!!!!!!!!!!")
find_package(CUDA)
message("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!find_cuda!!!!!!!!!!!!!!!!!!!!!!!!")
find_package(Qt4)

message("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!***************************************************")
#MESSAGE("opencv_libs: " ${OpenCV_LIBS})
MESSAGE("cuda_libs: " ${CUDA_LIBS})
message("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!***************************************************")

# setup Qt4 for build
include(${QT_USE_FILE})
add_definitions(${QT_DEFINITIONS})

# Find TensorRT
find_path(TRT_INCLUDE_DIR NvInfer.h HINTS ${TRT_SDK_ROOT} PATH_SUFFIXES include)
if(${TRT_INCLUDE_DIR} MATCHES "TRT_INCLUDE_DIR-NOTFOUND")
  MESSAGE(FATAL_ERROR "-- Unable to find TensorRT headers. Please set path using -DTRT_SDK_ROOT")
else()
  MESSAGE(STATUS "Found TensorRT headers at ${TRT_INCLUDE_DIR}")
endif()

find_library(TRT_LIBRARY_INFER nvinfer HINTS ${TRT_SDK_ROOT} PATH_SUFFIXES lib lib64 lib/x64)
find_library(TRT_LIBRARY_INFER_PLUGIN nvinfer_plugin HINTS ${TRT_SDK_ROOT} PATH_SUFFIXES lib lib64 lib/x64)
if((${TRT_LIBRARY_INFER} MATCHES "TRT_LIBRARY_INFER-NOTFOUND") OR (${TRT_LIBRARY_INFER_PLUGIN} MATCHES "TRT_LIBRARY_INFER_PLUGIN-NOTFOUND"))
  MESSAGE(FATAL_ERROR "-- Unable to find TensorRT libs. Please set path using -DTRT_SDK_ROOT")
else()
  set(TRT_LIBRARY ${TRT_LIBRARY_INFER} ${TRT_LIBRARY_INFER_PLUGIN})
  MESSAGE(STATUS "Found TensorRT libs at ${TRT_LIBRARY}")
endif()


# Add yolo lib as subdir
add_subdirectory(./yolo_lib)

#compile
cuda_add_executable(traffic_detect_jetson src/traffic_detect_jetson.cpp src/status_machine.cpp  src/yolo_helper.cpp src/trafficlights_detector.cpp)

#link
target_link_libraries(traffic_detect_jetson  yolo-lib ${OpenCV_LIBS} ${catkin_LIBRARIES} jetson-inference )

#
#add_dependencies(traffic_detect ${catkin_EXPORTED_TARGETS})


