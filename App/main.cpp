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
#include "../GeoShoot/GeoShoot.hpp"
#include "../GeoShoot/Matrix.hpp"

void Run(int argc, char ** argv) {
    SetDevice(compute::system::default_device());
    std::cout << "OpenCL will use " << GetDevice().name() << std::endl;
    compute::command_queue queue { GetContext(), GetDevice() };

    auto image = ScalarField::Read({"/Users/alexm/Desktop/image.nii"});
    auto momentum = ScalarField::Read({"/Users/alexm/Desktop/momentum.nii"});
    GeoShoot gs { CopyOnDevice(image, queue), CopyOnDevice(momentum, queue), queue };
    gs.Shoot(10);
    queue.finish();
}

int main(int argc, char ** argv) {
    /*float A[3] = { 0, 2, 3 };
    float B[3] = { 1, 1, 1 };
    float C[3] = { 4, 5, 0 };
    float D[3] = { 1, 1, 1 };
    float X[3];
    TridiagonalSolveFloat(A, B, C, D, X, 3);
    for (auto && x : X)
        std::cout << x << std::endl;
    return 0;*/

    clfftSetupData data;
    clfftSetup(&data);
    Run(argc, argv);
    clfftTeardown();
    return 0;
}
