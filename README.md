# openpose-ros
CMU's Openpose for ROS

## Packages

```
$ sudo apt-get install ros-indigo-image-common ros-indigo-vision-opencv ros-indigo-video-stream-opencv ros-indigo-image-view
```
## Test with Camera/Video Files

```
$ roslaunch openpose_ros_node videostream.launch video:=0 video_visualize:=true
$ roslaunch openpose_ros_node videostream.launch video:=${filepath} video_visualize:=true
```
