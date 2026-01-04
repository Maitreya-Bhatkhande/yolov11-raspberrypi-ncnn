cmake_minimum_required(VERSION 3.14)
project(yoloncnn LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_BUILD_TYPE MATCHES Debug)
    message("Debug Build")
    add_compile_options(-Wall -g)
else()
    message("Release Build")
    add_compile_options(-O3)
endif()

set(SOURCES
    src/yolo11.cpp          
)

#OpenCV
find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
    message(STATUS "OpenCV_LIBS: ${OpenCV_LIBS}")
    message(STATUS "OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
else()
    message(FATAL_ERROR "OpenCV not found!")
endif()

#OpenMP
find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
    message(STATUS "OpenMP found")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else()
    message(FATAL_ERROR "OpenMP not found!")
endif()

add_executable(yoloncnn ${SOURCES})

target_include_directories(yoloncnn PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    /usr/local/include/ncnn
)

target_link_libraries(yoloncnn
    # ncnn static lib
    /usr/local/lib/ncnn/libncnn.a

    # OpenCV
    ${OpenCV_LIBS}

    # ALL THE GLSLANG / SPIR-V STATIC LIBS NEEDED FOR VULKAN SUPPORT
    ${CMAKE_SOURCE_DIR}/thirdparty/ncnn_build/glslang/glslang/libglslang.a
    ${CMAKE_SOURCE_DIR}/thirdparty/ncnn_build/glslang/glslang/libMachineIndependent.a
    ${CMAKE_SOURCE_DIR}/thirdparty/ncnn_build/glslang/glslang/libGenericCodeGen.a
    ${CMAKE_SOURCE_DIR}/thirdparty/ncnn_build/glslang/glslang/libglslang-default-resource-limits.a
    ${CMAKE_SOURCE_DIR}/thirdparty/ncnn_build/glslang/glslang/OSDependent/Unix/libOSDependent.a
    ${CMAKE_SOURCE_DIR}/thirdparty/ncnn_build/glslang/SPIRV/libSPIRV.a
)
