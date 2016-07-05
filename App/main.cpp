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
    auto target = ScalarField::Read({"/Users/alexm/Desktop/target.nii"});
    auto momentum = ScalarField { image.NX(), image.NY(), image.NZ() };
    momentum.Fill(0.f);
    GeoShoot gs { std::move(image), std::move(target), std::move(momentum), 10, queue };
    gs.Run(20);
    queue.finish();
}

int main(int argc, char ** argv) {
    clfftSetupData data;
    clfftSetup(&data);
    Run(argc, argv);
    clfftTeardown();
    return 0;
}
