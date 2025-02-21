cmake_minimum_required(VERSION 3.0.2)
project(joint_capsule)

find_package(catkin REQUIRED COMPONENTS
  message_generation
  roscpp
  rospy
  std_msgs
  tf
  urdf
)

add_message_files(
  FILES
  Int32MultiArrayWithHeader.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS roscpp rospy std_msgs message_runtime
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

add_executable(camera_link_frame_copier src/camera_link_frame_copier.cpp)
target_link_libraries(camera_link_frame_copier ${catkin_LIBRARIES})

add_executable(transform_from_link2_to_link1 src/transform_from_link2_to_link1.cpp)
target_link_libraries(transform_from_link2_to_link1 ${catkin_LIBRARIES})

add_executable(test_message_filter src/test_message_filter.cpp)
target_link_libraries(test_message_filter ${catkin_LIBRARIES})

add_executable(message_filter_and_dump_to_csv src/message_filter_and_dump_to_csv.cpp)
target_link_libraries(message_filter_and_dump_to_csv ${catkin_LIBRARIES})
