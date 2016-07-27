cmake_minimum_required (VERSION 2.6)
project (CLGeoShoot)

#On ubuntu, make sure that the opencl headers have been installed: 
# -> "sudo apt-get install opencl-headers"
# -> get appropriate openCL drivers (eg: https://software.intel.com/en-us/articles/opencl-drivers#latest_linux_driver) and install them (eg https://software.intel.com/en-us/forums/opencl/topic/390630)
#get the binaries of  https://github.com/clMathLibraries/clFFT/releases  (under linux -> put them in /usr/lib/)



set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")


include_directories ("${PROJECT_SOURCE_DIR}")
include_directories ("${PROJECT_SOURCE_DIR}/App")
include_directories ("${PROJECT_SOURCE_DIR}/include")
include_directories ("${PROJECT_SOURCE_DIR}/include/niftilib")
include_directories ("${PROJECT_SOURCE_DIR}/include/znzlib")

add_subdirectory (GeoShoot)


set (EXTRA_LIBS ${EXTRA_LIBS} libCLGS)
set (EXTRA_LIBS ${EXTRA_LIBS} OpenCL)
set (EXTRA_LIBS ${EXTRA_LIBS} clFFT)

add_executable (CLGeoShoot            App/main.cpp)


target_link_libraries (CLGeoShoot  ${EXTRA_LIBS})


configure_file("${PROJECT_SOURCE_DIR}/GeoShoot/OpenCL.cl" "${CMAKE_BINARY_DIR}/OpenCL.cl" COPYONLY)
