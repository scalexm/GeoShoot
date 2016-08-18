/*=========================================================================
 
 main.cpp
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

#include <iostream>
#include <chrono>
#include <clFFT/clFFT.h>
#include "../GeoShoot/GPU.hpp"
#include "../GeoShoot/GeoShoot.hpp"

ScalarField ComputePOUFromImage(const ScalarField &);
void SmoothPOU(ScalarField &, float, compute::command_queue &);

void usage() {
    std::cerr << "Usage: geoshoot [Source] [Target] [RegionsMask] [SigmaPOU] <options>\n";
    std::cerr << "  Mandatory parameters:\n";
    std::cerr << "   [Source]                     is the source (template/moving) image\n";
    std::cerr << "   [Target]                     is the target (fixed) image\n";
    std::cerr << "   [RegionsMask]                is a 3D mask representing the regions of the partition of unity (POU). It is in the [Source] image domain and has integer intensities only.\n";
    std::cerr << "   [SigmaPOU]                   To define the POU, the regions of [PartitionOfUnity] are smoothed with a Gaussian kernel of std [SigmaPOU]. Note that [RegionsMask] will be considered as the actual 3D+t POU if [SigmaPOU]<0.\n";
    std::cerr << "  Primary options:\n";
    std::cerr << "    <-iterations n>             Number of iterations (default=10)\n";
    std::cerr << "    <-subdivisions n>           Number of subdivisons between t=0 and t=1 (default=10)\n";
    std::cerr << "    <-alpha n>                  Weight of the norm in the cost function (Default=0.001)\n";
    std::cerr << "    <-SetRegionKernel n>        Set the kernel in region [Reg]. Region 0 is the one with the lowest intensity in [RegionsMask] and so on.\n";
    std::cerr << "                                The kernel is the sum of [N] Gaussian kernels where [wn][sn] are the weight and std dev of n'th Gaussian kernels.\n";
    std::cerr << "                                Parameters are n=([Reg]  [N][w1][s1]...[wN][sN]).\n";
    std::cerr << "                                Default kernel in all regions and background is a Gaussian kernel with a std dev of 3mm.\n";
    std::cerr << "    <-M_Gauss_easier n>         Sum of 7 linearly sampled Gaussian kernels with apparent weights = 1 in region Reg    --- n = Reg Smax Smin  (S. in mm)\n" ;
    std::cerr << "  Inputs (default: nothing):\n";
    std::cerr << "    <-InputInitMomentum n>      Initial Momentum to initiate the gradient descent (default=\"Null\")\n";
    std::cerr << "    <-affineT n>                Affine transfo from Trg to Template in the world domain. The 4*3 parameters are: r_xx r_xy r_xz t_x  r_yx ... t_z\n";
    std::cerr << "    <-affineT_txt n>            Affine transfo from Trg to Template in the world domain. The 4*4 matrix is an ascii text file.\n";
    std::cerr << "  Outputs (default: initial momentum in a nifti file):\n";
    std::cerr << "    <-OutputPath n>             Output directory (default = \".\")";
    std::cerr << "  Secondary options:\n";
    std::cerr << "    <-MaxVeloUpdt n>            Size of the maximum updates of the vector field (Default=0.5 voxels)\n";
    std::cerr << "  GPU options:\n";
    std::cerr << "    <-Device n>                 Name of the device used for computations\n";
    std::cerr << "    <-ShowDevices>              Show available devices and exit\n";
    std::cerr << "    <-KernelSource n>           Path for the OpenCL kernels source (default=\"./OpenCL.cl\")\n";
}

int Run(int argc, char ** argv) {
    int N = 10;
    int nbIterations = 10;
    std::string momentumPath, outputPath = "./", deviceName, sourcePath = "./OpenCL.cl";

    if (argc < 5) {
        usage();
        return 1;
    }

    auto image = ScalarField::Read({ argv[1] });
    auto target = ScalarField::Read({ argv[2] });

    float sigmaPOU = atof(argv[4]);

    ScalarField regions;
    bool needSmoothing = false;
    try {
        regions = ScalarField::Read({ argv[3] });
        if (regions.NT() == 1 && sigmaPOU >= 0.f) {
            std::cout << "Computing POU..." << std::endl;
            regions = ComputePOUFromImage(regions);
            needSmoothing = true;
        }
        std::cout << regions.NT() << " regions found" << std::endl;
    } catch (const std::invalid_argument &) {
        regions = ScalarField { 0, 0, 0, 1 }; // dummy region
    }

    argc -= 4;
    argv += 4;

    Matrix<4, 4> transfo;
    memset(&transfo[0], 0, 16 * sizeof(float));
    transfo[0][0] = transfo[1][1] = transfo[2][2] = transfo[3][3] = 1;

    ConvolverCnf base;
    base.Weights = { 1.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f };
    base.SigmaXs = { 3.f, 3.f, 3.f, 3.f, 3.f, 3.f, 3.f };
    base.SigmaYs = { 3.f, 3.f, 3.f, 3.f, 3.f, 3.f, 3.f };
    base.SigmaZs = { 3.f, 3.f, 3.f, 3.f, 3.f, 3.f, 3.f };
    std::vector<ConvolverCnf> configs(regions.NT(), base);

    float alpha = 0.001f, maxVeloUpdate = 0.5f;

    while (argc > 1) {
        auto arg = std::string { argv[1] };
        --argc; ++argv;
        if (arg == "-subdivisions" && argc > 1) {
            N = atoi(argv[1]);
            --argc; ++argv;
        } else if (arg == "-iterations" && argc > 1) {
            nbIterations = atoi(argv[1]);
            --argc; ++argv;
        } else if (arg == "-InputInitMomentum" && argc > 1) {
            momentumPath = argv[1];
            --argc; ++argv;
        } else if (arg == "-OutputPath" && argc > 1) {
            outputPath = argv[1];
            --argc; ++argv;
        } else if (arg == "-Device" && argc > 1) {
            deviceName = argv[1];
            --argc; ++argv;
        } else if (arg == "-ShowDevices") {
            for (auto && d : compute::system::devices())
                std::cout << d.name() << std::endl;
            return 0;
        } else if (arg == "-affineT" && argc > 12) {
            for (auto i = 1; i <= 12; ++i)
                transfo[(i - 1) / 4][(i - 1) % 4] = atof(argv[i]);
            argc -= 12;
            argv += 12;
        } else if (arg == "-SetRegionKernel" && argc > 2) {
            auto region = atoi(argv[1]);
            --argc; ++argv;
            auto & cnf = configs.at(region);
            auto temp = atoi(argv[1]);
            --argc; ++argv;
            for (auto i = 1; i <= 7; ++i) {
                if (temp >= i && argc > 2) {
                    cnf.Weights[i - 1] = atof(argv[1]);
                    cnf.SigmaXs[i - 1] = cnf.SigmaYs[i - 1] = cnf.SigmaZs[i - 1] = atof(argv[2]);
                    argc -= 2;
                    argv += 2;
                }
            }
        } else if (arg == "-M_Gauss_easier" && argc > 3) {
            int region = atoi(argv[1]);
            auto & cnf = configs.at(region);
            cnf.SigmaXs[0] = atof(argv[2]);
            cnf.SigmaXs[6] = atof(argv[3]);
            argc -= 3;
            argv += 3;

            if (cnf.SigmaXs[0] < cnf.SigmaXs[6])
                std::swap(cnf.SigmaXs[0], cnf.SigmaXs[6]);

            cnf.SigmaYs[0] = cnf.SigmaZs[0] = cnf.SigmaXs[0];
            cnf.SigmaYs[6] = cnf.SigmaZs[6] = cnf.SigmaXs[6];

            cnf.Weights[0] = 0.f;

            auto a = (cnf.SigmaYs[6] - cnf.SigmaYs[0]) / 6.f;
            auto b = cnf.SigmaYs[0] - a;

            for (auto i = 2; i <= 6; ++i)
                cnf.SigmaXs[i - 1] = cnf.SigmaYs[i - 1] = cnf.SigmaZs[i - 1] = i * a + b;
        } else if (arg == "-alpha" && argc > 1) {
            alpha = atof(argv[1]);
            --argc; ++argv;
        } else if (arg == "-MaxVeloUpdate" && argc > 1) {
            maxVeloUpdate = atof(argv[1]);
            --argc; ++argv;
        } else if (arg == "-KernelSource" && argc > 1) {
            sourcePath = argv[1];
            --argc; ++argv;
        } else {
            usage();
            return 1;
        }
    }

    ScalarField momentum;
    if (momentumPath.empty()) {
        momentum = ScalarField { image.NX(), image.NY(), image.NZ() };
        momentum.Fill(0.f);
    } else
        momentum = ScalarField::Read({ momentumPath.c_str() });


    if (deviceName.empty())
        SetDevice(compute::system::default_device());
    else
        SetDevice(compute::system::find_device(deviceName));

    std::cout << "OpenCL will use " << GetDevice().name() << std::endl;
    compute::command_queue queue { GetContext(), GetDevice() };
    SetSourcePath(std::move(sourcePath));

    if (needSmoothing) {
        std::cout << "Smoothing POU..." << std::endl;
        SmoothPOU(regions, sigmaPOU, queue);
    }

    auto tp = std::chrono::system_clock::now();

    GeoShoot gs {
        image,
        target,
        momentum,
        std::move(regions),
        transfo,
        N,
        queue
    };

    gs.ConvolverConfigs = std::move(configs);
    gs.Alpha = alpha;
    gs.MaxUpdate = maxVeloUpdate;

    gs.Run(nbIterations);
    queue.finish();

    std::cout << (std::chrono::system_clock::now() - tp).count() << std::endl;

    gs.Save(outputPath);

    return 0;
}

int main(int argc, char ** argv) {
    clfftSetupData data;
    clfftSetup(&data);
    auto ret = Run(argc, argv);
    clfftTeardown();
    return ret;
}
