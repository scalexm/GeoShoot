set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")



add_library (CLGS      Calculus.cpp GPU.cpp  GeoShoot.cpp VectorField.cpp FFTConvolver.cpp Matrix.cpp)
