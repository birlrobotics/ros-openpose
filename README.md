# openpose-ros
CMU's Openpose for ROS

## Packages

```
$ sudo apt-get install ros-indigo-image-common ros-indigo-vision-opencv ros-indigo-video-stream-opencv
```
## Test with Camera/Video Files

```
$ roslaunch openpose_ros_node videostream.launch video:=0
$ roslaunch openpose_ros_node videostream.launch video:=${filepath}
```
