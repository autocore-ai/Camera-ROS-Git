cmake_minimum_required(VERSION 2.8.3)
project(traffic_detect)

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
set(CURR_DIR $(shell pwd))
#set(ARCH $(shell uname -m | sed -e s/arm.*/armv71/ \-e s/aarch64.*/aarch64/ ) )
#ifeq ($(ARCH),aarch64)
    add_definitions(-mcpu=cortex-a53)
#endif

set(PROJECT_EMBED_OBJ_FILES "")
set(PROJECT_EMBED_FILES "model/dpu_yolo.elf")
foreach(FILENAME ${PROJECT_EMBED_FILES})
    get_filename_component(FILENAME_ONLY ${FILENAME} NAME)
    message(STATUS "FILENAME_ONLY is " ${FILENAME_ONLY})
    get_filename_component(FILEPATH_ONLY ${FILENAME} PATH)
    message(STATUS "FILEPATH_ONLY is " ${FILEPATH_ONLY})
    message(STATUS "CMAKE_CURRENT_BINARY_DIR is " ${CMAKE_CURRENT_BINARY_DIR})
    message(STATUS "PROJECT_SOURCE_DIR is " ${PROJECT_SOURCE_DIR})
    message(STATUS "WORKING_DIRECTORY is " ${PROJECT_SOURCE_DIR}/${FILEPATH_ONLY})
    message(STATUS "EFF.O is " ${CMAKE_CURRENT_BINARY_DIR}/${FILENAME_ONLY}.o)

    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${FILENAME_ONLY}.o 
                       WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/${FILEPATH_ONLY} 
                       COMMAND ${CMAKE_OBJCOPY} 
                       ARGS ${FILENAME_ONLY} ${CMAKE_CURRENT_BINARY_DIR}/${FILENAME_ONLY}.o )
    list(APPEND PROJECT_EMBED_OBJ_FILES ${CMAKE_CURRENT_BINARY_DIR}/${FILENAME_ONLY}.o)
    message(STATUS "PROJECT_EMBED_OBJ_FILES is " ${PROJECT_EMBED_OBJ_FILES})
    message(STATUS "Objcopy'ing ${FILENAME}")
endforeach(FILENAME)

add_executable(traffic_detect src/main.cpp src/status_machine.cpp  src/trafficlights_detector.cpp ${PROJECT_EMBED_OBJ_FILES})
#add_executable(traffic_detect src/main.cpp src/status_machine.cpp  src/trafficlights_detector.cpp ${PROJECT_EMBED_FILES})


#link
target_link_libraries(traffic_detect  yolo-lib ${OpenCV_LIBS} ${catkin_LIBRARIES} )

#
#add_dependencies(traffic_detect ${catkin_EXPORTED_TARGETS})


