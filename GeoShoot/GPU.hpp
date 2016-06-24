//
//  GPU.hpp
//  GeoShoot
//
//  Created by Alexandre Martin on 22/06/2016.
//  Copyright © 2016 scalexm. All rights reserved.
//

#ifndef GPU_HPP
#define GPU_HPP

#define BOOST_COMPUTE_DEBUG_KERNEL_COMPILATION
#include <boost/compute.hpp>

namespace compute = boost::compute;

compute::context & GetContext();
compute::device & GetDevice();
void SetDevice(const compute::device &);

#define MAKE_PROGRAM(source, context) \
    static compute::program prog; \
    static bool built = false; \
    if (!built) { \
        prog = compute::program::create_with_source(source, context); \
        prog.build(); \
        built = true; \
    } \
    return prog;

#endif
