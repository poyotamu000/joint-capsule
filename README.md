 # README

## Overview
This research code is based on ROS Noetic.

## Data Collection
Set up Arduino MEGA and the SLAM Camera according to the referenced papers and URDF files. Then, execute the following commands to drive the joints:

```sh
roslaunch multi_adc.launch
roslaunch joint_tf_for_rosbag.launch
rosbag record -a
```

## Data Verification
To visualize the recorded data in RViz, execute the following commands:

```sh
roslaunch joint_tf_for_rosbag.launch
rosbag play $rosbag_filename.bag
```

## Training and Analysis
- The code corresponding to section **2.2**:
  ```sh
  python3 learning_adc64_to_rpy_and_translation.py
  ```
- The code corresponding to section **2.3**:
  ```sh
  python3 learning_with_PI_analysis_reduction_one_trial.py
  ```
- The code corresponding to section **2.4**:
  ```sh
  python3 learning_with_filtering.py
  ```

## Analysis Results
The analysis results are stored in the `learning_results/` directory.


