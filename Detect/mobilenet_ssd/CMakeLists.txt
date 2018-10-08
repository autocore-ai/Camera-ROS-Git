cmake_minimum_required(VERSION 2.8.3)
project(mobilenet_ssd)

set( TENGINE_LIBS_DIR ${PROJECT_SOURCE_DIR}/../Detect_lib/tengine_lib)
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
        autoware_msgs
)


find_package(catkin REQUIRED COMPONENTS
  autoware_msgs
  roscpp
  std_msgs
  cv_bridge
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

add_executable(mobilenet_ssd src/mssd.cpp)

target_link_libraries(mobilenet_ssd ${TENGINE_LIBS} ${OpenCV_LIBS} ${catkin_LIBRARIES})

add_dependencies(mobilenet_ssd ${catkin_EXPORTED_TARGETS})


