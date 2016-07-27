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

void usage() {
  std::cerr << "Usage: geoshoot [Template] [Target] <options>\n";
  std::cerr << "Where <options> are one or more of the following:\n";
  std::cerr << "  Primary options:\n";
  std::cerr << "    <-iterations n>             Number of iterations (default=10)\n";
  std::cerr << "    <-subdivisions n>           Number of subdivisons between t=0 and t=1 (default=10)\n";
  //cerr << "    <-UnderSampleTpl n>         Undersample the template image with a factor n (default n = 1)\n";
  std::cerr << "    <-alpha n>                  Weight of the norm in the cost function (Default=0.001)\n";
  std::cerr << "  Kernels (Default: -Gauss 3):\n";
  std::cerr << "    <-Gauss n>                  Gaussian kernel of std. dev. Sigma (in mm)\n";
  std::cerr << "    <-M_Gauss n>                Sum of Gaussian kernels (max 7)   --- n = k W1 S1 ... Wk Sk   (k=[#kernels], W.=weight, S.=Sigma in mm)\n" ;
  std::cerr << "    <-M_Gauss_easier n>         Sum of 7 linearly sampled Gaussian kernels with apparent weights = 1    --- n = Smax Smin  (S. in mm)\n" ;
  std::cerr << "  Inputs (default: nothing):\n";
  std::cerr << "    <-InputInitMomentum n>      Initial Momentum to initiate the gradient descent (default=\"Null\")\n";
  std::cerr << "    <-affineT n>                Affine transfo from Trg to Template in the world domain. The 4*3 parameters are: r_xx r_xy r_xz t_x  r_yx ... t_z\n";
  //cerr << "    <-affineT_txt n>            Affine transfo from Trg to Template in the world domain. The 4*4 matrix is an ascii text file.\n";
  std::cerr << "  Outputs (default: initial momentum in a nifti file):\n";
  std::cerr << "    <-OutputPath n>             Output directory (default = \".\")";
  //cerr << "    <-OutFinalDef>              Outputs the deformed template (Nothing by default)\n";
  //cerr << "    <-OutDispField>             Outputs the displacement field in mm (Nothing by default)\n";
  //cerr << "    <-OutDispFieldEvo>          Outputs the evolution of the displacement field in mm along the diffeomorphism (Nothing by default)\n";
  //cerr << "    <-OutIniMoTxt n>            Outputs the initial momentum in an ascii file (default=\"Null\")\n";
  //cerr << "    <-OutVeloField>             Outputs the 3D+t velocity field in voxels (Nothing by default)\n";
  //cerr << "    <-OutDistEnSim>             Outputs the distance, enrgy and similarity measure (Nothing by default)\n";
  //cerr << "    <-OutDeformation>           Outputs the 3D+t deformation (Nothing by default)\n";
  //cerr << "    <-OutDiff_Tpl_DefTrg>       Outputs the difference between the template and the deformed target image (Nothing by default)\n";
  std::cerr << "  Secondary options:\n";
  std::cerr << "    <-MaxVeloUpdt n>            Size of the maximum updates of the vector field (Default=0.5 voxels)\n";
  //std::cerr << "    <-margins n>                Margin of the image where the calculations are reduced  (default=3 voxels)\n";
  //std::cerr << "    <-GreyLevAlign n>           Grey level linear alignment of each channel -- n = [Padding Src] [Padding Trg]\n";
  //std::cerr << "    <-Mask n>                   Mask in which the momenta are computed (values!=0  -- in the template image domain)\n";
  std::cerr << "  GPU options:\n";
  std::cerr << "    <-Device n>                 Name of the device used for computations\n";
  std::cerr << "    <-ShowDevices>              Show available devices and exit\n";
  std::cerr << "    <-KernelSource n>           Path for the OpenCL kernels source (default=\"./OpenCL.cl\")\n";
}

int Run(int argc, char ** argv) {
    int N = 10;
    int nbIterations = 10;
    std::string momentumPath, outputPath = "./", deviceName, sourcePath = "./OpenCL.cl";

    if (argc < 3) {
        usage();
        return 1;
    }

    auto image = ScalarField::Read({ argv[1] });
    auto target = ScalarField::Read({ argv[2] });

    argc -= 2;
    argv += 2;

    Matrix<4, 4> transfo;
    memset(&transfo[0], 0, 16 * sizeof(float));
    transfo[0][0] = transfo[1][1] = transfo[2][2] = transfo[3][3] = 1;

    std::array<float, 7> weights = {{ 100.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f }},
        sigmaXs = {{ 3.f, 3.f, 3.f, 3.f, 3.f, 3.f, 3.f }},
        sigmaYs = {{ 3.f, 3.f, 3.f, 3.f, 3.f, 3.f, 3.f }},
        sigmaZs = {{ 3.f, 3.f, 3.f, 3.f, 3.f, 3.f, 3.f }};

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
        } else if (arg == "-Gauss" && argc > 1) {
            sigmaXs[0] = sigmaYs[0] = sigmaZs[0] = atof(argv[1]);
            --argc; ++argv;
        } else if (arg == "-M_gauss" && argc > 1) {
            auto temp = atoi(argv[1]);
            --argc; ++argv;
            for (auto i = 1; i <= 7; ++i) {
                if (temp >= i && argc > 2) {
                    weights[i - 1] = atof(argv[1]);
                    sigmaXs[i - 1] = sigmaYs[i - 1] = sigmaZs[i - 1] = atof(argv[2]);
                    argc -= 2;
                    argv += 2;
                }
            }
        } else if (arg == "-M_Gauss_easier" && argc > 2) {
            sigmaXs[0] = atof(argv[1]);
            sigmaXs[6] = atof(argv[2]);
            argc -= 2;
            argv += 2;

            if (sigmaXs[0] < sigmaXs[6])
                std::swap(sigmaXs[0], sigmaXs[6]);

            sigmaYs[0] = sigmaZs[0] = sigmaXs[0];
            sigmaYs[6] = sigmaZs[6] = sigmaXs[6];

            weights[0] = 0.f;

            auto a = (sigmaYs[6] - sigmaYs[0]) / 6.f;
            auto b = sigmaYs[0] - a;

            for (auto i = 2; i <= 6; ++i)
                sigmaXs[i - 1] = sigmaYs[i - 1] = sigmaZs[i - 1] = i * a + b;
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

    /*auto i = ScalarField::Read({"/Users/alexm/Desktop/image.nii"});
    compute::image3d field(GetContext(), i.NX(), i.NY(), i.NZ(),
        compute::image_format { compute::image_format::r, compute::image_format::float32 },
        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, &i.Buffer()[0]);

    compute::image3d grad(GetContext(), i.NX(), i.NY(), i.NZ(),
        compute::image_format { compute::image_format::rgb, compute::image_format::float32 },
        CL_MEM_READ_WRITE);

    auto ffield = GPUScalarField { i.NX(), i.NY(), i.NZ(), 1, GetContext() };
    auto ggrad = GPUVectorField<3> { i.NX(), i.NY(), i.NZ(), 1, GetContext() };
    compute::copy(i.Begin(), i.End(), ffield.Begin(), queue);

    std::vector<float> v(3 * i.NX() * i.NY() * i.NZ());

    auto tp = std::chrono::system_clock::now();

    auto kernel = GetProgram().create_kernel("gradientTest");
    size_t workDim[3] = { (size_t) i.NX(), (size_t) i.NY(), (size_t) i.NZ() };
    kernel.set_arg(0, grad);
    kernel.set_arg(1, field);
    size_t origin[3] = { 0, 0, 0 };
    size_t region[3] = { (size_t) i.NX(), (size_t) i.NY(), (size_t) i.NZ() };
    for (auto t = 0; t < 1; ++t) {
        queue.enqueue_nd_range_kernel(kernel, 3, NULL, workDim, NULL);
        queue.enqueue_read_image(grad, origin, region, 0, 0, &v[0]);
    }

    std::cout << (std::chrono::system_clock::now() - tp).count() << std::endl;

    auto g = VectorField<3> { i.NX(), i.NY(), i.NZ() };
    for (auto z = 0; z < i.NZ(); ++z) {
        for (auto y = 0; y < i.NY(); ++y) {
            for (auto x = 0; x < i.NX(); ++x) {
                auto ind = x + y * i.NX() + z * i.NX() * i.NZ();
                g.P({ v[3*ind], v[3*ind + 1], v[3*ind + 2] }, x, y, z);
            }
        }
    }
    g.Write({"/Users/alexm/Desktop/grad_x.nii", "/Users/alexm/Desktop/grad_y.nii", "/Users/alexm/Desktop/grad_z.nii"});
    return 0;*/

    GeoShoot gs { std::move(image), std::move(target), std::move(momentum), transfo, N, queue };

    gs.Weights = std::move(weights);
    gs.SigmaXs = std::move(sigmaXs);
    gs.SigmaYs = std::move(sigmaYs);
    gs.SigmaZs = std::move(sigmaZs);
    gs.Alpha = alpha;
    gs.MaxUpdate = maxVeloUpdate;

    gs.Run(nbIterations);
    queue.finish();

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
