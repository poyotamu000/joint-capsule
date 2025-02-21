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
    const joint_capsule::Int32MultiArrayWithHeaderConstPtr& adc1,
    const joint_capsule::Int32MultiArrayWithHeaderConstPtr& adc2,
    const joint_capsule::Int32MultiArrayWithHeaderConstPtr& adc3,
    const geometry_msgs::Vector3StampedConstPtr& translation,
    const geometry_msgs::QuaternionStampedConstPtr& quaternion,
    const geometry_msgs::Vector3StampedConstPtr& rpy_deg) {
  ROS_INFO("Received synchronized messages: [%d, %d, %d, %d, %lf, %lf, %lf]",
           adc0->data.data.at(0), adc0->data.data.at(0), adc0->data.data.at(0),
           adc0->data.data.at(3), rpy_deg->vector.x, rpy_deg->vector.y,
           rpy_deg->vector.z);
  // Open CSV file in append mode
  std::ofstream file;
  file.open("data/output.csv", std::ios_base::app);

  if (file.is_open()) {
    // Write header data
    file << adc0->header.stamp << "," << adc1->header.stamp << ","
         << adc2->header.stamp << "," << adc3->header.stamp << ","
         << rpy_deg->header.stamp << "," << translation->header.stamp << ","
         << quaternion->header.stamp << ",";

    // Write adc data (16 elements each)
    for (size_t i = 0; i < 16; ++i) {
      file << adc0->data.data[i] << ",";
    }
    for (size_t i = 0; i < 16; ++i) {
      file << adc1->data.data[i] << ",";
    }
    for (size_t i = 0; i < 16; ++i) {
      file << adc2->data.data[i] << ",";
    }
    for (size_t i = 0; i < 16; ++i) {
      file << adc3->data.data[i] << ",";
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
      adc_sub0(nh, "/adc_topic0", 1);
  message_filters::Subscriber<joint_capsule::Int32MultiArrayWithHeader>
      adc_sub1(nh, "/adc_topic1", 1);
  message_filters::Subscriber<joint_capsule::Int32MultiArrayWithHeader>
      adc_sub2(nh, "/adc_topic2", 1);
  message_filters::Subscriber<joint_capsule::Int32MultiArrayWithHeader>
      adc_sub3(nh, "/adc_topic3", 1);
  message_filters::Subscriber<geometry_msgs::Vector3Stamped> translation_sub(
      nh, "/translation", 1);
  message_filters::Subscriber<geometry_msgs::QuaternionStamped> quaternion_sub(
      nh, "/quaternion", 1);
  message_filters::Subscriber<geometry_msgs::Vector3Stamped> rpy_sub(
      nh, "/rpy_deg", 1);

  typedef message_filters::sync_policies::ApproximateTime<
      joint_capsule::Int32MultiArrayWithHeader,
      joint_capsule::Int32MultiArrayWithHeader,
      joint_capsule::Int32MultiArrayWithHeader,
      joint_capsule::Int32MultiArrayWithHeader,
      geometry_msgs::Vector3Stamped, geometry_msgs::QuaternionStamped,
      geometry_msgs::Vector3Stamped>
      MySyncPolicy;

  message_filters::Synchronizer<MySyncPolicy> sync(
      MySyncPolicy(10), adc_sub0, adc_sub1, adc_sub2, adc_sub3, translation_sub,
      quaternion_sub, rpy_sub);
  sync.registerCallback(boost::bind(&callback, _1, _2, _3, _4, _5, _6, _7));

  // Create or clear the CSV file at the start
  std::ofstream file;
  file.open("data/output.csv");
  if (file.is_open()) {
    file << "adc0_stamp,adc1_stamp,adc2_stamp,adc3_stamp,rpy_stamp,translation_"
            "stamp,quaternion_stamp,"
         << "adc0_data0,adc0_data1,adc0_data2,adc0_data3,adc0_data4,adc0_data5,"
            "adc0_data6,adc0_data7,"
         << "adc0_data8,adc0_data9,adc0_data10,adc0_data11,adc0_data12,adc0_"
            "data13,adc0_data14,adc0_data15,"
         << "adc1_data0,adc1_data1,adc1_data2,adc1_data3,adc1_data4,adc1_data5,"
            "adc1_data6,adc1_data7,"
         << "adc1_data8,adc1_data9,adc1_data10,adc1_data11,adc1_data12,adc1_"
            "data13,adc1_data14,adc1_data15,"
         << "adc2_data0,adc2_data1,adc2_data2,adc2_data3,adc2_data4,adc2_data5,"
            "adc2_data6,adc2_data7,"
         << "adc2_data8,adc2_data9,adc2_data10,adc2_data11,adc2_data12,adc2_"
            "data13,adc2_data14,adc2_data15,"
         << "adc3_data0,adc3_data1,adc3_data2,adc3_data3,adc3_data4,adc3_data5,"
            "adc3_data6,adc3_data7,"
         << "adc3_data8,adc3_data9,adc3_data10,adc3_data11,adc3_data12,adc3_"
            "data13,adc3_data14,adc3_data15,"
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
