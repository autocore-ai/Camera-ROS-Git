cmake_minimum_required(VERSION 2.8.3)
project(vps_detect)

message("***************************************************")
MESSAGE("PROJECT_SOURCE_DIR: " ${PROJECT_SOURCE_DIR})
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
	dashboard_msgs
)


find_package(catkin REQUIRED COMPONENTS
  autoreg_msgs
  dashboard_msgs
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
  #yolo_lib
  ${catkin_INCLUDE_DIRS}
)

####### for jetson
find_package(jetson-utils)
find_package(jetson-inference)

# CUDA and Qt4 are required
#message("cuda")
find_package(CUDA)
find_package(Qt4)

# setup Qt4 for build
include(${QT_USE_FILE})
add_definitions(${QT_DEFINITIONS})

# yolo lib
set (YOLO_DIR ${PROJECT_SOURCE_DIR}/../../yolo_lib)
MESSAGE("YOLO_DIR: " ${YOLO_DIR})
include_directories(${YOLO_DIR})
link_directories( ${YOLO_DIR})

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


set( CUDA_LIBS_DIR /usr/local/cuda/lib64)
link_directories( ${CUDA_LIBS_DIR})


# add yolo_lib directory
# add_subdirectory(../../yolo_lib yolo_lib.out)

#compile
cuda_add_executable(vps_detect src/main.cpp src/vps_detector.cpp src/auto_yulan_sdk.cpp src/carpark_mgr.cpp)

#link
target_link_libraries(vps_detect  yolo-lib ${OpenCV_LIBS} ${catkin_LIBRARIES} jetson-inference )

#
#add_dependencies(vps_detect ${catkin_EXPORTED_TARGETS})

