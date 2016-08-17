/*=========================================================================
 
 Regions.cpp
 App

 Author: Alexandre Martin, Laurent Risser
 Copyright © 2016 scalexm, lrisser. All rights reserved.
 
 Disclaimer: This software has been developed for research purposes only, and hence should 
 not be used as a diagnostic tool. In no event shall the authors or distributors
 be liable to any direct, indirect, special, incidental, or consequential 
 damages arising of the use of this software, its documentation, or any 
 derivatives thereof, even if the authors have been advised of the possibility 
 of such damage. 

 =========================================================================*/

#include "../GeoShoot/FFTConvolver.hpp"
#include <limits>
#include <algorithm>

ScalarField ComputePOUFromImage(const ScalarField & image) {
    std::vector<float> regions;
    auto epsilon = std::numeric_limits<float>::epsilon();

    for (auto z = 0; z < image.NZ(); ++z) {
        for (auto y = 0; y < image.NY(); ++y) {
            for (auto x = 0; x < image.NX(); ++x) {
                bool newRegion = true;
                auto val = image.G<0>(x, y, z);
                for (auto && r : regions) {
                    if (fabs(r - val) < epsilon)
                        newRegion = false;
                }

                if (newRegion)
                    regions.emplace_back(val);
            }
        }
    }

    std::sort(regions.begin(), regions.end());

    auto POU = ScalarField { image.NX(), image.NY(), image.NZ(), (int) regions.size() };

    // fill POU with 0s
    for (auto r = 0; r < POU.NT(); ++r) {
        POU.ChangeChannel(r);
        POU.Fill(0.f);
    }

    for (auto z = 0; z < image.NZ(); ++z) {
        for (auto y = 0; y < image.NY(); ++y) {
            for (auto x = 0; x < image.NX(); ++x) {
                int idRegion = 0;
                auto val = image.G<0>(x, y, z);
                for (auto region = 0; region < regions.size(); ++region) {
                    if (fabs(regions[region] - val) < epsilon) {
                        idRegion = region;
                        break;
                    }
                }

                POU.ChangeChannel(idRegion);
                POU.P({ 1.f }, x, y, z);
            }
        }
    }

    return POU;
}

void SmoothPOU(ScalarField & POU, float sigmaPOU, compute::command_queue & queue) {
    FFTConvolver cnv(queue);

    auto && mat = POU.Image2World();
    float xmm = sqrt(mat[0][0]*mat[0][0]+mat[0][1]*mat[0][1]+mat[0][2]*mat[0][2]);
    float ymm = sqrt(mat[1][0]*mat[1][0]+mat[1][1]*mat[1][1]+mat[1][2]*mat[1][2]);
    float zmm = sqrt(mat[2][0]*mat[2][0]+mat[2][1]*mat[2][1]+mat[2][2]*mat[2][2]);

    cnv.InitiateConvolver(
        POU.NX(),
        POU.NY(),
        POU.NZ(),
        { 1.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f },
        { sigmaPOU / xmm, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f },
        { sigmaPOU / ymm, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f },
        { sigmaPOU / zmm, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f }
    );

    auto dims = cnv.Dims();
    auto acc = GPUVectorField<2> { dims[0], dims[1], dims[2], 1, queue.get_context() };
    auto signal = GPUScalarField { POU.NX(), POU.NY(), POU.NZ(), 1, queue.get_context() };

    for (auto r = 0; r < POU.NT(); ++r) {
        POU.ChangeChannel(r);
        compute::copy(POU.Begin(), POU.End(), signal.Begin(), queue);
        cnv.Convolution(signal, acc);
        compute::copy(signal.Begin(), signal.End(), POU.Begin(), queue);
    }
}