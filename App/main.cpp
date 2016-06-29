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
#include "../GeoShoot/GPU.hpp"
#include "../GeoShoot/GeoShoot.hpp"

void Run(int argc, char ** argv) {
    SetDevice(compute::system::default_device());
    std::cout << "OpenCL will use " << GetDevice().name() << std::endl;
    compute::command_queue queue { GetContext(), GetDevice() };

    auto image = ScalarField::Read({"/Users/alexm/Desktop/image.nii"});
    auto momentum = ScalarField::Read({"/Users/alexm/Desktop/momentum.nii"});
    //GeoShoot gs { std::move(image), std::move(momentum), 10, queue };
    //gs.Shoot();
    queue.finish();
}

int main(int argc, char ** argv) {
    clfftSetupData data;
    clfftSetup(&data);
    Run(argc, argv);
    clfftTeardown();
    return 0;
}
