# Surround_Detect

**Surround_Detect**,developed by **AUTOCORE** AI,is a lite,high-performance,high-accuracy vechile surround camera object detectors for embedded device.
Surround_Detect is composed of one modules:**park_detect/**.

-[**park_detect**](park_detect) provide a high-performance park detector for surrounding images

**park_detect**
## Build 
```
cd ${SURROUND_DETECT_ROOT}/park_detect
cmake ./
make
```
build as ${SURROUND_DETECT_ROOT}/README.md

## Run
1.run PARK_DETECT by default
    ```
    - models are in `models/MobileNetSSD_deploy.prototxt` and`models/MobileNetSSD_deploy.caffemodel`
    - camera device is `/dev/video1`
    ```
    cd ${SURROUND_DETECT_ROOT}/park_detect/
    ./PARK_DETECT
2. run with your model_path and camera device
    ```
    [Usge]:./park_detect/PARK_DETECT [-h]
    [-p proto_file] [-m model_file] [-v video_source]
    
    ./PARK_DETECT -p deploy.prototxt -m deploy.caffemodel -v /dev/video0
    ```
### version 0.1.0 -2018/09/29
Initial release of single park detector,only support free and forbidden park real-time detect
