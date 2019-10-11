# Camera-ROS-Git xlinx
## requirements
- ros installed on debian buster
- dnndk installed
- opencv

## compile

## run app
- source devel/setup.bash
- roslaunch traffic_detect traffic_detect.launch

## test
modify traffic_detect/src/main.cpp
```
//test_xlinx(dir_path,filename,cfgfile);
to
test_xlinx(dir_path,filename,cfgfile);
return 0
```
/root/sc/data/test目录下有测试图片及label文件.程序将读取图片,推理得到类别,与label文件中真值作比较,得到统计数据.




