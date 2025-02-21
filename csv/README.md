 # Dataset README

## Overview
This dataset contains data collected from 60 strain gauge sensors and position/orientation data obtained from RealSense T265 while operating a bio-mimetic joint.

## Dataset Files
- **output_whole_all.csv**: Data recorded while moving in all directions.
- **output_angle90_random.csv**: Data recorded when moving up to approximately 90 degrees.
- **output_push_back.csv**: Data involving push and pull operations.
- **all_merged_data.csv**: A comprehensive dataset that includes the above data along with additional diverse data.
- **py_filtered_data_over_70.csv**: Filtered data where the bending angle is 70 degrees or more.
- **roll_filtered_data_over_20.csv**: Filtered data where the twisting angle is 20 degrees or more.
- **x_filtered_over_0.002_or_under_0.002_from_push_back.csv**: Extracted data where displacement in the push and pull direction is 2mm or more.

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


