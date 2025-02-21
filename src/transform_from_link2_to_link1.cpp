#include <geometry_msgs/QuaternionStamped.h>
#include <geometry_msgs/Vector3Stamped.h>
#include <ros/ros.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>

int main(int argc, char** argv) {
  ros::init(argc, argv, "tf_from_link1_to_link2_publisher");
  ros::NodeHandle nh;

  tf::TransformListener tf_listener;
  tf::TransformBroadcaster tf_broadcaster;

  std::string source_frame = "link1";
  std::string target_frame = "link2";

  ros::Publisher translation_pub =
      nh.advertise<geometry_msgs::Vector3Stamped>("translation", 10);
  ros::Publisher quaternion_pub =
      nh.advertise<geometry_msgs::QuaternionStamped>("quaternion", 10);
  ros::Publisher rpy_rad_pub =
      nh.advertise<geometry_msgs::Vector3Stamped>("rpy_rad", 10);
  ros::Publisher rpy_deg_pub =
      nh.advertise<geometry_msgs::Vector3Stamped>("rpy_deg", 10);

  ros::Rate rate(10.0);
  while (nh.ok()) {
    tf::StampedTransform transform_camera_link_to_link1;
    tf::StampedTransform transform_link2_to_camera_link_dummy;

    // The transformation matrix link1-H-camera_link satisfies the
    // transformation:
    // Point_{link1} = H * Point_{camera_link}
    // In other words, it is the camera_link observed from link1,
    //    which is the transform from camera_link to link1.

    // This time, we will transform the coordinates of link2's point
    // coordinates to those observed in the camera_link frame, and then
    // further transform them to the coordinates observed in the link1
    // frame, so the final target_frame will be the link1 frame.
    try {
      // wait for the transforms to become available
      tf_listener.waitForTransform("link1", "camera_link", ros::Time(0),
                                   ros::Duration(1.0));
      tf_listener.lookupTransform("link1", "camera_link", ros::Time(0),
                                  transform_camera_link_to_link1);
      tf_listener.waitForTransform("camera_link_dummy", "link2", ros::Time(0),
                                   ros::Duration(1.0));
      tf_listener.lookupTransform("camera_link_dummy", "link2", ros::Time(0),
                                  transform_link2_to_camera_link_dummy);
    } catch (tf::TransformException& e) {
      ROS_ERROR("%s", e.what());
      ros::Duration(1.0).sleep();
      continue;
    }

    // calculate the transform link2 to link1
    tf::Transform transform_link2_to_link1 =
        transform_camera_link_to_link1 * transform_link2_to_camera_link_dummy;

    try {
      // broadcast frame
      tf_broadcaster.sendTransform(
          tf::StampedTransform(transform_link2_to_link1, ros::Time::now(),
                               source_frame, target_frame));
    } catch (tf::TransformException& e) {
      ROS_WARN("%s", e.what());
      ros::Duration(1.0).sleep();
      continue;
    }

    tf::Vector3 origin = transform_link2_to_link1.getOrigin();
    tf::Quaternion rotation = transform_link2_to_link1.getRotation();

    // Create and publish Translation message
    geometry_msgs::Vector3Stamped translation_msg;
    translation_msg.header.stamp = ros::Time::now();
    translation_msg.vector.x = origin.x();
    translation_msg.vector.y = origin.y();
    translation_msg.vector.z = origin.z();
    translation_pub.publish(translation_msg);

    // Create and publish Quaternion message
    geometry_msgs::QuaternionStamped quaternion_msg;
    quaternion_msg.header.stamp = ros::Time::now();
    quaternion_msg.quaternion.x = rotation.x();
    quaternion_msg.quaternion.y = rotation.y();
    quaternion_msg.quaternion.z = rotation.z();
    quaternion_msg.quaternion.w = rotation.w();
    quaternion_pub.publish(quaternion_msg);

    // Convert quaternion to roll, pitch, and yow
    double roll, pitch, yaw;
    tf::Matrix3x3(rotation).getRPY(roll, pitch, yaw);

    // Convert radian to degree
    double roll_deg = roll * 180 / M_PI;
    double pitch_deg = pitch * 180 / M_PI;
    double yaw_deg = yaw * 180 / M_PI;

    // Create and publish RPY degree message
    geometry_msgs::Vector3Stamped rpy_rad_msg;
    rpy_rad_msg.header.stamp = ros::Time::now();
    rpy_rad_msg.vector.x = roll;
    rpy_rad_msg.vector.y = pitch;
    rpy_rad_msg.vector.z = yaw;
    rpy_rad_pub.publish(rpy_rad_msg);

    // Create and publish RPY degree message
    geometry_msgs::Vector3Stamped rpy_deg_msg;
    rpy_deg_msg.header.stamp = ros::Time::now();
    rpy_deg_msg.vector.x = roll_deg;
    rpy_deg_msg.vector.y = pitch_deg;
    rpy_deg_msg.vector.z = yaw_deg;
    rpy_deg_pub.publish(rpy_deg_msg);

    // for debug
    ROS_INFO("Translation: [%f, %f, %f]", origin.x(), origin.y(), origin.z());
    ROS_INFO("Rotation: [%f, %f, %f, %f]", rotation.x(), rotation.y(),
             rotation.z(), rotation.w());
    ROS_INFO("Roll: [%f], Pitch: [%f], Yaw: [%f]", roll, pitch, yaw);
    ROS_INFO("Roll(deg): [%f], Pitch(deg): [%f], Yaw(deg): [%f]", roll_deg,
             pitch_deg, yaw_deg);

    rate.sleep();
  }
  return 0;
}
