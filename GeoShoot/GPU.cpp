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