cmake_minimum_required(VERSION 3.6)
project(sgm_project)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package(OpenCV REQUIRED)

#set(SOURCE_FILES simple_tree.cpp)
#add_executable(stereo_vision ${SOURCE_FILES})

set(SOURCE_FILES main)
add_executable(sgm_project ${SOURCE_FILES})

#set(SOURCE_FILES test.cpp)
#add_executable(stereo_vision ${SOURCE_FILES})

include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(sgm_project ${OpenCV_LIBS})