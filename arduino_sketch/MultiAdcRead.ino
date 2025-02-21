#include <ros.h>
#include <std_msgs/Header.h>
#include <std_msgs/Int32MultiArray.h>
#include <joint_capsule/Int32MultiArrayWithHeader.h>

ros::NodeHandle nh;


joint_capsule::Int32MultiArrayWithHeader adc_msg;
ros::Publisher adc_pub("adc_topic", &adc_msg);

void setup() {
  nh.initNode();
  nh.advertise(adc_pub);

  adc_msg.data.data_length = 16;
  adc_msg.data.data = (int32_t*)malloc(16 * sizeof(int32_t));
  adc_msg.data.layout.dim = (std_msgs::MultiArrayDimension*)malloc(1 * sizeof(std_msgs::MultiArrayDimension));
  adc_msg.data.layout.dim[0].size = 16;
  adc_msg.data.layout.dim[0].stride = 1;
  adc_msg.data.layout.dim[0].label = "adc_values";
}

void loop() {
  adc_msg.header.stamp = nh.now();
  for (int i=0; i<16; i++) {
    adc_msg.data.data[i] = analogRead(i);
  }

  adc_pub.publish(&adc_msg);
  nh.spinOnce();

  delay(100);
}
