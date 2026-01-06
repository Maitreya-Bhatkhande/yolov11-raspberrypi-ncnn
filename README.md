# YOLOv11 Inference on Raspberry Pi using C++, ncnn & OpenCV
## Vulkan + OpenMP + ARM NEON Optimized
High-performance YOLOv11 Nano (YOLO11n) inference pipeline for Raspberry Pi using C++, ncnn, and OpenCV, optimized for edge devices with:
- Vulkan GPU acceleration
- OpenMP multi-core CPU parallelism
- ARM NEON vectorization
- FP16 and INT8 inference support

This repository is designed to be used alongside the full YouTube tutorial, where every step is explained in detail — from building OpenCV and ncnn, to custom YOLO training, INT8 quantization, and final performance comparison on Raspberry Pi.

👉 Watch the full tutorial here:
[ Youtube Tutorial ](https://youtu.be/qTQMqTeQjFQ)

The README focuses on commands, structure, and reproducibility, while the video covers the reasoning, optimizations, and performance insights behind each step.
## Key Features
- YOLOv11 Nano (YOLO11n) custom training
- Pure C++ inference (no Python runtime)
- ncnn inference engine
- Vulkan backend enabled
- OpenMP multi-threading
- ARM NEON optimized execution
- FP16 optimized inference
- INT8 quantization with calibration
- Raspberry Pi 4 (2GB RAM) tested

## Project Structure

```
yoloncnn/
├── thirdparty/        # OpenCV and ncnn (built locally, not tracked)
├── src/               # C++ source code
├── build/             # Build directory
├── data/
│   ├── models/        # FP16 / optimized / INT8 models
│   └── calib_imgs/    # INT8 calibration images
└── CMakeLists.txt
```
**Note:**
The thirdparty directory is intentionally not included in the repository.
OpenCV and ncnn must be built locally using the commands below.
## System Dependencies
```
sudo apt update && sudo apt upgrade
sudo apt install build-essential cmake git libvulkan-dev vulkan-tools protobuf-compiler libprotobuf-dev libomp-dev
```
## OpenCV Build (ARM Optimized)
### Install OpenCV Dependencies
```
sudo apt update
sudo apt install -y cmake gfortran \
    python3-dev python3-numpy \
    libjpeg-dev libtiff-dev \
    libavcodec-dev libavformat-dev libswscale-dev \
    libgtk-3-dev \
    libxvidcore-dev libx264-dev \
    libtbb12 libtbb-dev libdc1394-dev libv4l-dev \
    libopenblas-dev liblapack-dev libblas-dev \
    libjpeg62-turbo-dev \
    libhdf5-dev \
    libprotobuf-dev protobuf-compiler \
    libopenexr-dev
```
### Project Setup
```
mkdir yoloncnn && cd yoloncnn
mkdir thirdparty && cd thirdparty
```
### Download OpenCV
```
wget -O opencv.zip https://github.com/opencv/opencv/archive/4.12.0.zip
wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.12.0.zip

unzip opencv.zip && unzip opencv_contrib.zip
mv opencv-4.12.0 opencv && mv opencv_contrib-4.12.0 opencv_contrib
mkdir build && cd build
```
### Configure OpenCV (⚠️ Do NOT modify flags unnecessarily)
```
cmake ../opencv \
      -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D OPENCV_EXTRA_MODULES_PATH=../opencv_contrib/modules \
      -D ENABLE_NEON=ON \
      -D WITH_OPENMP=ON \
      -D WITH_TBB=ON \
      -D WITH_FFMPEG=ON \
      -D WITH_GSTREAMER=OFF \
      -D WITH_V4L=ON \
      -D WITH_LIBV4L=ON \
      -D WITH_1394=OFF \
      \
      -D BUILD_opencv_python3=ON \
      -D PYTHON3_EXECUTABLE=$(which python3) \
      -D OPENCV_ENABLE_NONFREE=ON \
      -D BUILD_EXAMPLES=OFF \
      -D BUILD_TESTS=OFF \
      -D BUILD_PERF_TESTS=OFF \
      \
      -D BUILD_opencv_apps=OFF \
      -D BUILD_opencv_java=OFF \
      -D BUILD_opencv_js=OFF \
      -D WITH_QT=OFF \
      -D WITH_GTK=OFF \
      -D WITH_OPENGL=OFF \
      -D WITH_IPP=OFF \
      -D OPENCV_GENERATE_PKGCONFIG=ON \
      -D OPENCV_EXTRA_EXE_LINKER_FLAGS=-latomic \
      -D BUILD_DOCS=OFF \
      -D BUILD_opencv_world=OFF
```
### Temporary Swap (Recommended for Pi)
```
sudo fallocate -l 4G /swapfile 
sudo chmod 600 /swapfile 
sudo mkswap /swapfile 
sudo swapon /swapfile
```
Verify swap:
```
free -h
```
#### Build & Install OpenCV
```
make -j4
sudo make install 
sudo ldconfig
```
## NCNN Build (Vulkan + OpenMP + ARM Optimized)
### Download NCNN
```
git clone https://github.com/Tencent/ncnn.git
cd ncnn
git submodule update --init
cd ..
```
### Configure NCNN (⚠️ Do NOT modify flags unnecessarily)
```
mkdir ncnn_build && cd ncnn_build
cmake ../ncnn \
      -DCMAKE_BUILD_TYPE=Release \
      -DNCNN_VULKAN=ON \
      -DNCNN_SYSTEM_GLSLANG=OFF \
      -DNCNN_DISABLE_RTTI=OFF \
      -DNCNN_OPENMP=ON \
      -DNCNN_BUILD_TOOLS=ON \
      -DNCNN_INSTALL_SDK=ON \
      -DNCNN_BUILD_BENCHMARK=OFF \
      -DNCNN_BUILD_TESTS=OFF \
      -DNCNN_BUILD_EXAMPLES=OFF
```
#### Build & Install NCNN
```
make -j4
sudo make install
sudo mkdir -p /usr/local/lib/ncnn 
sudo cp -r install/include/ncnn /usr/local/include/
sudo cp install/lib/libncnn.a /usr/local/lib/ncnn/
sudo ldconfig
```
Verify installation:
```
ls /usr/local/include/ncnn/net.h && ls /usr/local/lib/ncnn/libncnn.a
```
## INT8 Quantization (ncnn)
### Model Optimization
```
~/yoloncnn/thirdparty/ncnn_build/tools/ncnnoptimize   model.ncnn.param   model.ncnn.bin   model-opt.param   model-opt.bin   0
```
### Generate INT8 Calibration Table
```
~/yoloncnn/thirdparty/ncnn_build/tools/quantize/ncnn2table \
model-opt.param \
model-opt.bin \
/home/user/yoloncnn/data/calib_imgs/calib_list.txt \
model.table \
mean=0 norm=0.0039216 shape=480,480,3 pixel=BGR thread=4 method=kl
```
### Convert to INT8
```
~/yoloncnn/thirdparty/ncnn_build/tools/quantize/ncnn2int8 \
model-opt.param \
model-opt.bin \
model-int8.param \
model-int8.bin \
model.table
```
## Running Inference
## FP16 Model:
```
./yoloncnn /home/user/yoloncnn/data/bus.jpg /home/user/yoloncnn/data/models/model.ncnn 0
```
## Optimized FP16 Model:
```
./yoloncnn /home/user/yoloncnn/data/bus.jpg /home/user/yoloncnn/data/models/model-opt 0
```
## INT8 Model:
```
./yoloncnn /home/user/yoloncnn/data/bus.jpg /home/user/yoloncnn/data/models/model-int8 1

```
## Custom YOLO Training (Google Colab)
The repository includes a custom Google Colab notebook for:
- COCO-Person dataset preparation
- YOLOv11n training (480×480)
- ncnn export
- INT8 calibration dataset generation

The notebook is included in the repository and can be uploaded directly to Google Colab.
## Final Notes
- Designed for educational + real edge deployment
- Commands are tested and reproducible
- Avoid modifying build flags unless you know exactly why

