cmake_minimum_required(VERSION 2.8.3)
project(auv_ekf_slam)

# check c++14 / c++0x
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++14" COMPILER_SUPPORTS_CXX14)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
if(COMPILER_SUPPORTS_CXX14)
    set(CMAKE_CXX_FLAGS "-std=c++14")
     add_compile_options(-std=c++14 -Wall -Wextra)
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "-std=c++11")
     add_compile_options(-std=c++11 -Wall -Wextra)
else()
    message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++14 or C++11 support. Please use a different C++ compiler.")
endif()

# Set debug flags
# Better to compile catkin_make -DCMAKE_BUILD_TYPE=Debug
#set(CMAKE_CXX_FLAGS_DEBUG ${CMAKE_CXX_FLAGS_DEBUG} -Wall)
#set(ROS_BUILD_TYPE Debug)

find_package(catkin REQUIRED COMPONENTS
    roscpp
    rospy
    std_msgs
    tf
    tf_conversions
    message_generation
    geometry_msgs
    sensor_msgs
    message_filters
    pcl_ros
    auv_ekf_localization
    smarc_lm_visualizer
 )

find_package(PCL REQUIRED)
if(PCL_FOUND)
    include_directories(${PCL_INCLUDE_DIRS})
    link_directories(${PCL_LIBRARY_DIRS})
    add_definitions(${PCL_DEFINITIONS})
endif()

#set(Boost_USE_STATIC_LIBS        ON)
find_package(Boost REQUIRED COMPONENTS system signals)
if(Boost_FOUND)
  include_directories(${Boost_INCLUDE_DIRS})
endif()

find_package(Munkres)
if(MUNKRES_FOUND)
  include_directories(${MUNKRES_INCLUDE_DIRS})
endif()

# setup.py for map_provider_node.py
#catkin_python_setup()

#generate_messages(
#    DEPENDENCIES
#    geometry_msgs
#    std_msgs
#    )

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES
  CATKIN_DEPENDS roscpp rospy std_msgs tf message_runtime message_filters smarc_lm_visualizer
#  DEPENDS system_lib
)

include_directories(
 include
 ${PROJECT_SOURCE_DIR}/utils
 ${catkin_INCLUDE_DIRS}
)

add_library(ekf_core
    src/ekf_slam_core.cpp
    )

add_library(ekf_utils
    utils/ekf_utils.cpp
    )

add_library(corresp_obj
    src/correspondence_obj_mbes.cpp
    src/correspondence_obj_fls.cpp
    include/correspondence_class/correspondence_class.hpp
    )

add_library (munkres
    ${PROJECT_SOURCE_DIR}/utils/munkres/munkres.cpp
)

add_executable(ekf_slam_node
    src/ekf_slam.cpp
    src/ekf_slam_node.cpp
    )

add_dependencies(ekf_slam_node
    ${${PROJECT_NAME}_EXPORTED_TARGETS}
    ${catkin_EXPORTED_TARGETS}
    )

target_link_libraries(ekf_slam_node
    noise_kf
    ekf_core
    ekf_utils
    corresp_obj
    munkres
    ${catkin_LIBRARIES}
    ${Boost_LIBRARIES}
    )










