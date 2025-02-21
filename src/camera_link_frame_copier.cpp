#include <ros/ros.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>

int main(int argc, char** argv) {
  ros::init(argc, argv, "camera_link_frame_copier");
  ros::NodeHandle nh;

  tf::TransformListener tf_listener;
  tf::TransformBroadcaster tf_broadcaster;

  std::string source_frame = "camera_link";
  std::string copy_frame = "camera_link_dummy";

  ros::Rate rate(10.0);
  while (nh.ok()) {
    tf::StampedTransform transform;
    try {
      // source frame -> copy_frame
      tf_listener.lookupTransform("/link1", source_frame, ros::Time(0),
                                  transform);

      // broadcast frame to copy_frame
      tf_broadcaster.sendTransform(tf::StampedTransform(
          transform, ros::Time::now(), "link1", copy_frame));
    } catch (tf::TransformException& e) {
      ROS_WARN("%s", e.what());
      ros::Duration(1.0).sleep();
      continue;
    }
    rate.sleep();
  }
  return 0;
}
