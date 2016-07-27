/*=========================================================================
 
 GPU.cpp
 GeoShoot

 Author: Alexandre Martin, Laurent Risser
 Copyright © 2016 scalexm, lrisser. All rights reserved.
 
 Disclaimer: This software has been developed for research purposes only, and hence should 
 not be used as a diagnostic tool. In no event shall the authors or distributors
 be liable to any direct, indirect, special, incidental, or consequential 
 damages arising of the use of this software, its documentation, or any 
 derivatives thereof, even if the authors have been advised of the possibility 
 of such damage. 

 =========================================================================*/

#include "GPU.hpp"

static compute::context GlobalContext;
static compute::device GlobalDevice;
static std::string SourcePath;

compute::context & GetContext() {
    return GlobalContext;
}

compute::device & GetDevice() {
    return GlobalDevice;
}

void SetDevice(const compute::device & dev) {
    GlobalDevice = dev;
    GlobalContext = compute::context { GlobalDevice };
}

void SetSourcePath(std::string path) {
    SourcePath = std::move(path);
}

compute::program & GetProgram() {
    static compute::program prog;
    static bool built = false;
    if (!built) {
        prog = compute::program::create_with_source_file(SourcePath, GetContext());
        prog.build();
        built = true;
    }
    return prog;
}