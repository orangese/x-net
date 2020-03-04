# x-net

A deep convolutional neural model for X-ray threat detection.

__Reuses some code from and is inspired by https://github.com/qqwweee/keras-yolo3.__

## About

Entry for NJRSF 2020, JSHS 2020, and MIT THINK 2020. 

## Data

Used the [SIXray dataset](https://github.com/MeioJane/SIXray).

## Results

### mAP

Classification Results                                     |  Localization Results                                  
:---------------------------------------------------------:|:-------------------------------------------------------:
![Classification Results](results/plots/classification_map.jpg)  |  ![Localization Results](results/plots/localization_map.jpg)

### Examples

All correctly detected.

![Scissors](results/examples/scissors.png)    |  ![Wrench and Pliers](results/examples/wrench_pliers.png)                                  
:--------------------------------------------:|:-------------------------------------------------------------:
![Knife](results/examples/knife.png)          |  ![Pliers](results/examples/pliers.png)
