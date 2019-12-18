# /**
# MIT License

# Copyright (c) 2018 NVIDIA CORPORATION. All rights reserved.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# *
# */

cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

include("../cmake.dir")

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  cv_bridge
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

add_definitions(-Wno-sign-compare -Wno-narrowing -std=c++11 -g)

file(GLOB CXX_SRCS model_helper.cpp)

find_package(PkgConfig)
pkg_check_modules(GLIB REQUIRED glib-2.0)

include_directories(${OpenCV_INCLUDE_DIRS}  ${GLIB_INCLUDE_DIRS} ${GLFAGS_INCLUDE_DIRS} )

set(TFLIBS "edgetpu")
set(COMMON_LDFLAGS "pthread")
add_library(model_helper SHARED ${CXX_SRCS})
target_link_libraries(model_helper ${OpenCV_LIBRARIES} ${TFLIBS} ${COMMON_LDFLAGS})
