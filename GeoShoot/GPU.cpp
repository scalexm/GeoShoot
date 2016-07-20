//
//  GPU.cpp
//  GeoShoot
//
//  Created by Alexandre Martin on 22/06/2016.
//  Copyright © 2016 scalexm. All rights reserved.
//

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