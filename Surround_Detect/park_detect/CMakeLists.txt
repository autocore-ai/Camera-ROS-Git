
cmake_minimum_required (VERSION 2.8)
project(SOURROUND_DETECT)


set( TENGINE_LIBS tengine)

if( ANDROID AND ("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "armv7-a"))
    set( PROTOBUF_LIB ${PROTOBUF_DIR}/arm32_lib/libprotobuf.so)
endif()
if( ANDROID AND ("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "aarch64"))
    set( PROTOBUF_LIB ${PROTOBUF_DIR}/arm64_lib/libprotobuf.so)
endif()

if( BLAS_DIR)
    if( ANDROID AND ("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "armv7-a"))
         set( BLAS_LIB ${BLAS_DIR}/arm32/lib/libopenblas.so)
    endif()
    if( ANDROID AND ("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "aarch64"))
         set( BLAS_LIB ${BLAS_DIR}/arm64/lib/libopenblas.so)
    endif()
endif()


set( CODE_SRCS src/park_detect.cpp src/common.cpp)

#opencv
find_package(OpenCV REQUIRED)

#include
include_directories(include)

#lib
link_directories( ../../Tengine_Lib)

#exe
add_executable(PARK_DETECT ${CODE_SRCS})
if( ANDROID)
   target_link_libraries(PARK_DETECT ${TENGINE_LIBS} ${PROTOBUF_LIB} ${OpenCV_LIBS} ${BLAS_LIB})
else()
   target_link_libraries(PARK_DETECT ${TENGINE_LIBS} ${OpenCV_LIBS})
endif()


