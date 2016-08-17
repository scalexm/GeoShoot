cmake_minimum_required (VERSION 2.6)
project (CLGeoShoot)

#On ubuntu, make sure that the opencl headers have been installed:
# -> "sudo apt-get install opencl-headers"
# -> get appropriate openCL drivers (eg: https://software.intel.com/en-us/articles/opencl-drivers#latest_linux_driver) and install them (eg https://software.intel.com/en-us/forums/opencl/topic/390630)
#get the binaries of  https://github.com/clMathLibraries/clFFT/releases  (under linux -> put them in /usr/lib/ or in ./lib)


set (CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Os")


include_directories ("${PROJECT_SOURCE_DIR}/include")
include_directories ("${PROJECT_SOURCE_DIR}/include/niftilib")
include_directories ("${PROJECT_SOURCE_DIR}/include/znzlib")

add_subdirectory (GeoShoot)

find_package (OpenCL)
include_directories (${OPENCL_INCLUDE_DIRS})
include_directories ("${PROJECT_SOURCE_DIR}/lib")

set (EXTRA_LIBS ${EXTRA_LIBS} CLGS)
set (EXTRA_LIBS ${EXTRA_LIBS} ${OpenCL_LIBRARY})
set (EXTRA_LIBS ${EXTRA_LIBS} ${OPENCL_LIBRARIES})
set (EXTRA_LIBS ${EXTRA_LIBS} clFFT)

add_executable (CLGeoShoot            App/regions.cpp App/main.cpp)


target_link_libraries (CLGeoShoot  ${EXTRA_LIBS})


configure_file("${PROJECT_SOURCE_DIR}/GeoShoot/OpenCL.cl" "${CMAKE_BINARY_DIR}/OpenCL.cl" COPYONLY)
