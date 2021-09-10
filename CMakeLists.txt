cmake_minimum_required(VERSION 3.0.0)
project(DepthEstimation VERSION 0.1.0)

include(CTest)
enable_testing()

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS})

set(INCLUDE_DIR ${PROJECT_SOURCE_DIR}/inc)
set(SOURCE_DIR ${PROJECT_SOURCE_DIR}/src)

include_directories( include ${INCLUDE_DIR})
add_library( SRC 
            ${SOURCE_DIR}/StreoVision.cpp)

add_executable(DepthEstimation DepthEstimation.cpp ${test_src})

target_link_libraries( DepthEstimation SRC)
target_link_libraries( DepthEstimation ${OpenCV_LIBS} )

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
