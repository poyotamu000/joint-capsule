#include <geometry_msgs/QuaternionStamped.h>
#include <geometry_msgs/Vector3Stamped.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <ros/ros.h>
#include <std_msgs/Header.h>
#include <std_msgs/Int32MultiArray.h>

#include <fstream>

#include "joint_capsule/Int32MultiArrayWithHeader.h"

void callback(
    const joint_capsule::Int32MultiArrayWithHeaderConstPtr& adc0,
    const geometry_msgs::Vector3StampedConstPtr& translation,
    const geometry_msgs::QuaternionStampedConstPtr& quaternion,
    const geometry_msgs::Vector3StampedConstPtr& rpy_deg) {
  ROS_INFO("Received synchronized messages");
  ROS_INFO("ADC0: [%d, %lf, %lf, %lf]", adc0->data.data.at(0),
           rpy_deg->vector.x, rpy_deg->vector.y, rpy_deg->vector.z);

  // Open CSV file in append mode
  std::ofstream file;
  file.open("data/output.csv", std::ios_base::app);

  if (file.is_open()) {
    // Write header data
    file << adc0->header.stamp << "," << rpy_deg->header.stamp << ","
         << translation->header.stamp << "," << quaternion->header.stamp << ",";

    // Write adc data (16 elements each)
    for (size_t i = 0; i < 16; ++i) {
      file << adc0->data.data[i] << ",";
    }

    // Write rpy data
    file << rpy_deg->vector.x << "," << rpy_deg->vector.y << ","
         << rpy_deg->vector.z << ",";

    // Write translation data
    file << translation->vector.x << "," << translation->vector.y << ","
         << translation->vector.z << ",";

    // Write quaternion data
    file << quaternion->quaternion.x << "," << quaternion->quaternion.y << ","
         << quaternion->quaternion.z << "," << quaternion->quaternion.w
         << std::endl;

    file.close();
  } else {
    ROS_ERROR("Unable to open file");
  }
}

int main(int argc, char** argv) {
  ros::init(argc, argv, "message_filter_example");

  ros::NodeHandle nh;

  message_filters::Subscriber<joint_capsule::Int32MultiArrayWithHeader>
      adc_sub0(nh, "/adc16", 1);
  message_filters::Subscriber<geometry_msgs::Vector3Stamped> translation_sub(
      nh, "/translation", 1);
  message_filters::Subscriber<geometry_msgs::QuaternionStamped> quaternion_sub(
      nh, "/quaternion", 1);
  message_filters::Subscriber<geometry_msgs::Vector3Stamped> rpy_sub(
      nh, "/rpy_deg", 1);

  typedef message_filters::sync_policies::ApproximateTime<
      joint_capsule::Int32MultiArrayWithHeader,
      geometry_msgs::Vector3Stamped, geometry_msgs::QuaternionStamped,
      geometry_msgs::Vector3Stamped>
      MySyncPolicy;

  message_filters::Synchronizer<MySyncPolicy> sync(
      MySyncPolicy(10), adc_sub0, translation_sub, quaternion_sub, rpy_sub);
  sync.registerCallback(boost::bind(&callback, _1, _2, _3, _4));

  // Create or clear the CSV file at the start
  std::ofstream file;
  file.open("data/output.csv");
  if (file.is_open()) {
    file
        << "adc0_stamp,adc1_stamp,rpy_stamp,translation_stamp,quaternion_stamp,"
        << "adc0_data0,adc0_data1,adc0_data2,adc0_data3,adc0_data4,adc0_data5,"
           "adc0_data6,adc0_data7,"
        << "adc0_data8,adc0_data9,adc0_data10,adc0_data11,adc0_data12,adc0_"
           "data13,adc0_data14,adc0_data15,"
        << "rpy_x,rpy_y,rpy_z,"
        << "translation_x,translation_y,translation_z,"
        << "quaternion_x,quaternion_y,quaternion_z,quaternion_w" << std::endl;
    file.close();
  } else {
    ROS_ERROR("Unable to create file");
  }

  ros::spin();

  return 0;
}
