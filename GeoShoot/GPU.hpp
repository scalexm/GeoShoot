/*=========================================================================
 
 GPU.hpp
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

#ifndef GPU_HPP
#define GPU_HPP

#define BOOST_COMPUTE_DEBUG_KERNEL_COMPILATION
#include <boost/compute.hpp>

namespace compute = boost::compute;

compute::context & GetContext();
compute::device & GetDevice();
void SetDevice(const compute::device &);
void SetSourcePath(std::string);
compute::program & GetProgram();

#endif