 # Dataset README

## Overview
This dataset contains data collected from 60 strain gauge sensors and position/orientation data obtained from RealSense T265 while operating a bio-mimetic joint.

## Dataset Files
- **output_whole_all.csv**: Data recorded while moving in all directions.
- **output_angle90_random.csv**: Data recorded when moving up to approximately 90 degrees.
- **filtered_data_pitch_yaw_over_70deg.csv**: Filtered data where the bending angle is 70 degrees or more.
- **filtered_data_roll_over_20deg.csv**: Filtered data where the twisting angle is 20 degrees or more.
- **filtered_data_x_over_abs_0.002_from_push_back.csv**: Extracted data where displacement in the push and pull direction is 2mm or more.

## Sensor Number Mapping
Due to a mismatch between the circuit board and sensor numbers, the `adc_data{i}` index should be replaced as follows during analysis:

- `adc_data6` → `adc_data4`
- `adc_data7` → `adc_data5`
- `adc_data4` → `adc_data6`
- `adc_data5` → `adc_data7`

- `adc_data14` → `adc_data12`
- `adc_data15` → `adc_data13`
- `adc_data12` → `adc_data14`
- `adc_data13` → `adc_data15`


