//
//  main.cpp
//  App
//
//  Created by Alexandre Martin on 21/06/2016.
//  Copyright © 2016 scalexm. All rights reserved.
//

#include <iostream>
#include <chrono>
#include <clFFT/clFFT.h>
#include "../GeoShoot/VectorField.hpp"
#include "../GeoShoot/GPU.hpp"
#include "../GeoShoot/FFTConvolver.hpp"

void Run(int argc, char ** argv) {
    SetDevice(compute::system::default_device());
    std::cout << "OpenCL will use " << GetDevice().name() << std::endl;
}

int main(int argc, char ** argv) {
    clfftSetupData data;
    clfftSetup(&data);
    Run(argc, argv);
    clfftTeardown();
    return 0;
}
