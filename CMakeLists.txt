cmake_minimum_required(VERSION 3.17 FATAL_ERROR)
project(deepstream_app LANGUAGES CUDA CXX C)

# Setting values not set by user
function(check_variable_set variable_name default_value)
        if (NOT DEFINED ${variable_name})
                set(${variable_name} ${default_value} PARENT_SCOPE)
        endif()
endfunction()

check_variable_set(DS_VERSION 6.0.1)
check_variable_set(NVDS_INSTALL_DIR /opt/nvidia/deepstream/deepstream)

# Find include directories.
find_path(NVDS_INCLUDE_DIRS
  NAMES
    nvds_version.h
  HINTS
    ${NVDS_INSTALL_DIR}/sources/includes
)

set(CMAKE_CXX_FLAGS_DEBUG "-O0 -ggdb -g")

add_definitions("-DDEBUG")

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

set(CMAKE_CXX_STANDARD 14)
set(SRC_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/sources/includes)
set(SRC_FOLDER ${PROJECT_SOURCE_DIR}/sources/apps/sample_apps/deepstream-app)
set(SRC_APP_COMMON_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/sources/apps/apps-common/includes)

find_package(GStreamer REQUIRED)
find_package(CUDA REQUIRED)
find_package(X11 REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(OpenCV REQUIRED)
pkg_search_module(GLIB REQUIRED glib-2.0)
pkg_check_modules(JSON-GLIB REQUIRED json-glib-1.0)

message(STATUS "version: ${OpenCV_VERSION}$\n")
message(STATUS "libraries:${OpenCV_LIBS}$\n")
message(STATUS "INCLUDE PATH:${OpenCV_INCLUDE_DIRS}$\n")

include_directories(
    ${GSTREAMER_INCLUDE_DIRS}
    ${GSTREAMER_VIDEO_INCLUDE_DIRS}
    ${GSTREAMER_RTSPSERVER_INCLUDE_DIRS}
    ${GLIB_INCLUDE_DIRS}
    ${JSON-GLIB_INCLUDE_DIRS}
    ${NVDS_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${X11_INCLUDE_DIR}
    ${PROJECT_SOURCE_DIR}
    ${SRC_FOLDER}
    ${SRC_APP_COMMON_INCLUDE_DIRS}
    ${SRC_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    /usr/include/opencv2
)

link_directories(${NVDS_INSTALL_DIR}/lib)

file(GLOB SRCS2 ${SRC_FOLDER}/*.c ${SRC_FOLDER}/*.cpp ${SRC_FOLDER}/*.h)
file(GLOB SRC_APP_COMMON_SRCS ${SRC_APP_COMMON_INCLUDE_DIRS}/../src/*.c ${SRC_APP_COMMON_INCLUDE_DIRS}/../src/*.cpp ${SRC_APP_COMMON_INCLUDE_DIRS}/../src/*.h)

add_executable(${PROJECT_NAME}
    ${SRC_APP_COMMON_SRCS}
    ${SRCS2}
)

target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
    ${GSTREAMER_LIBRARIES}
    ${GSTREAMER_VIDEO_LIBRARIES}
    ${GSTREAMER_RTSPSERVER_LIBRARIES}
    ${GLIB_LIBRARIES}
    ${JSON-GLIB_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${CMAKE_DL_LIBS}
    ${X11_LIBRARIES}
    nvdsgst_meta
    nvds_meta
    nvdsgst_helper
    nvdsgst_smartrecord
    nvds_utils
    nvds_msgbroker
    nvds_batch_jpegenc
    m
)
