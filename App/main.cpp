//
//  main.cpp
//  App
//
//  Created by Alexandre Martin on 21/06/2016.
//  Copyright © 2016 scalexm. All rights reserved.
//

#include <iostream>
#include "../GeoShoot/VectorField.hpp"
#include "../GeoShoot/GPU.hpp"

int main(int argc, char *argv[])
{
    SetDevice(compute::system::default_device());
    std::cout << "OpenCL will use " << GetDevice().name() << std::endl;
    /*auto f = ScalarField::Read({ "/Users/alexm/Desktop/utilzreg-code/DATA/S02.Iso1mm.nii" });
    std::cout << f.NX() << "," << f.NY() << "," << f.NZ() << std::endl;
    f.Write({ "/Users/alexm/Desktop/lol.nii" });*/

    return 0;
}
