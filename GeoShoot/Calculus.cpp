/*=========================================================================
 
 Calculus.cpp
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

#include "Calculus.hpp"
#include "GPU.hpp"
#include <tuple>

void ComputeGradScalarField(const GPUScalarField & field, GPUVectorField<3> & gradient,
                            compute::command_queue & queue) {
    assert(field.NX() == gradient.NX());
    assert(field.NY() == gradient.NY());
    assert(field.NZ() == gradient.NZ());

    auto kernel = GetProgram().create_kernel("gradient");
    size_t workDim[3] = { (size_t) field.NX(), (size_t) field.NY(), (size_t) field.NZ() };
    kernel.set_arg(0, gradient.Buffer());
    kernel.set_arg(1, field.Buffer());
    kernel.set_arg(2, field.Dims());

    queue.enqueue_nd_range_kernel(kernel, 3, NULL, workDim, NULL);
}

void ComputeDivVectorField(const GPUVectorField<3> & field, GPUScalarField & div,
                           compute::command_queue & queue) {
    assert(field.NX() == div.NX());
    assert(field.NY() == div.NY());
    assert(field.NZ() == div.NZ());

    auto kernel = GetProgram().create_kernel("div");
    size_t workDim[3] = { (size_t) field.NX(), (size_t) field.NY(), (size_t) field.NZ() };
    kernel.set_arg(0, div.Buffer());
    kernel.set_arg(1, field.Buffer());
    kernel.set_arg(2, field.Dims());

    queue.enqueue_nd_range_kernel(kernel, 3, NULL, workDim, NULL);
}

void TransportImage(const GPUScalarField & src, const GPUVectorField<3> & diffeo,
                    GPUScalarField & dst, compute::command_queue & queue) {
    assert(src.NX() == diffeo.NX());
    assert(src.NY() == diffeo.NY());
    assert(src.NZ() == diffeo.NZ());
    assert(dst.NX() == src.NX());
    assert(dst.NY() == src.NY());
    assert(dst.NZ() == src.NZ());

    auto kernel = GetProgram().create_kernel("transportImage");
    size_t workDim[3] = { (size_t) src.NX(), (size_t) src.NY(), (size_t) src.NZ() };
    kernel.set_arg(0, dst.Buffer());
    kernel.set_arg(1, src.Buffer());
    kernel.set_arg(2, diffeo.Buffer());
    kernel.set_arg(3, src.Dims());

    queue.enqueue_nd_range_kernel(kernel, 3, NULL, workDim, NULL);
}

void TransportMomentum(const GPUScalarField & src, const GPUVectorField<3> & diffeo,
                       GPUScalarField & dst, compute::command_queue & queue) {
    assert(src.NX() == diffeo.NX());
    assert(src.NY() == diffeo.NY());
    assert(src.NZ() == diffeo.NZ());
    assert(dst.NX() == src.NX());
    assert(dst.NY() == src.NY());
    assert(dst.NZ() == src.NZ());

    auto kernel = GetProgram().create_kernel("transportMomentum");
    size_t workDim[3] = { (size_t) src.NX(), (size_t) src.NY(), (size_t) src.NZ() };
    kernel.set_arg(0, dst.Buffer());
    kernel.set_arg(1, src.Buffer());
    kernel.set_arg(2, diffeo.Buffer());
    kernel.set_arg(3, src.Dims());

    queue.enqueue_nd_range_kernel(kernel, 3, NULL, workDim, NULL);
}

void UpdateDiffeo(const GPUVectorField<3> & velocity, GPUVectorField<3> & diffeo, float deltaT,
                  compute::command_queue & queue) {
    assert(velocity.NX() == diffeo.NX());
    assert(velocity.NY() == diffeo.NY());
    assert(velocity.NZ() == diffeo.NZ());

    auto kernel = GetProgram().create_kernel("updateDiffeo");
    size_t workDim[3] = { (size_t) diffeo.NX(), (size_t) diffeo.NY(), (size_t) diffeo.NZ() };
    kernel.set_arg(0, diffeo.Buffer());
    kernel.set_arg(1, velocity.Buffer());
    kernel.set_arg(2, diffeo.Dims());
    kernel.set_arg(3, deltaT);

    queue.enqueue_nd_range_kernel(kernel, 3, NULL, workDim, NULL);
}

void UpdateInvDiffeo(const GPUVectorField<3> & velocity, GPUVectorField<3> & diffeo,
                     GPUVectorField<3> & accumulator, float deltaT,
                     compute::command_queue & queue) {
    assert(velocity.NX() == diffeo.NX());
    assert(velocity.NY() == diffeo.NY());
    assert(velocity.NZ() == diffeo.NZ());
    assert(accumulator.NX() == diffeo.NX());
    assert(accumulator.NY() == diffeo.NY());
    assert(accumulator.NZ() == diffeo.NZ());

    compute::copy(diffeo.Begin(), diffeo.End(), accumulator.Begin(), queue);
    compute::vector<int> cfl(1, queue.get_context());
    compute::fill(cfl.begin(), cfl.end(), 0, queue);

    auto kernel = GetProgram().create_kernel("updateInvDiffeo");
    size_t workDim[3] = { (size_t) diffeo.NX(), (size_t) diffeo.NY(), (size_t) diffeo.NZ() };
    kernel.set_arg(0, diffeo.Buffer());
    kernel.set_arg(1, velocity.Buffer());
    kernel.set_arg(2, accumulator.Buffer());
    kernel.set_arg(3, cfl);
    kernel.set_arg(4, diffeo.Dims());
    kernel.set_arg(5, deltaT);

    queue.enqueue_nd_range_kernel(kernel, 3, NULL, workDim, NULL);

    int res;
    compute::copy(cfl.begin(), cfl.end(), &res, queue);

    if (res == 1)
        std::cout << "CFL condition not respected" << std::endl;
}

void ScalarFieldTimesVectorField(const GPUScalarField & src1, const GPUVectorField<3> & src2,
                                 GPUVectorField<3> & dst, float factor,
                                 compute::command_queue & queue) {
    assert(src1.NX() == src2.NX());
    assert(src1.NY() == src2.NY());
    assert(src1.NZ() == src2.NZ());
    assert(src1.NX() == dst.NX());
    assert(src1.NY() == dst.NY());
    assert(src1.NZ() == dst.NZ());

    auto kernel = GetProgram().create_kernel("stimesv");
    size_t workDim[3] = { (size_t) src1.NX(), (size_t) src1.NY(), (size_t) src1.NZ() };
    kernel.set_arg(0, dst.Buffer());
    kernel.set_arg(1, src1.Buffer());
    kernel.set_arg(2, src2.Buffer());
    kernel.set_arg(3, src1.Dims());
    kernel.set_arg(4, factor);

    queue.enqueue_nd_range_kernel(kernel, 3, NULL, workDim, NULL);
}

void ScalarProduct(const GPUVectorField<3> & src1, const GPUVectorField<3> & src2,
                   GPUScalarField & dst, float factor, compute::command_queue & queue) {
    assert(src1.NX() == src2.NX());
    assert(src1.NY() == src2.NY());
    assert(src1.NZ() == src2.NZ());
    assert(src1.NX() == dst.NX());
    assert(src1.NY() == dst.NY());
    assert(src1.NZ() == dst.NZ());

    auto kernel = GetProgram().create_kernel("scalarProduct");
    size_t workDim[3] = { (size_t) src1.NX(), (size_t) src1.NY(), (size_t) src1.NZ() };
    kernel.set_arg(0, dst.Buffer());
    kernel.set_arg(1, src1.Buffer());
    kernel.set_arg(2, src2.Buffer());
    kernel.set_arg(3, src1.Dims());
    kernel.set_arg(4, factor);

    queue.enqueue_nd_range_kernel(kernel, 3, NULL, workDim, NULL);
}

float DotProduct(const GPUScalarField & src1, const GPUScalarField & src2,
                 GPUScalarField & accumulator, compute::command_queue & queue) {
    assert(src1.NX() == src2.NX());
    assert(src1.NY() == src2.NY());
    assert(src1.NZ() == src2.NZ());
    assert(src1.NX() == accumulator.NX());
    assert(src1.NY() == accumulator.NY());
    assert(src1.NZ() == accumulator.NZ());

    using compute::lambda::get;
    using compute::_1;

    compute::for_each(
        compute::make_zip_iterator(
            boost::make_tuple(src1.Begin(), src2.Begin(), accumulator.Begin())
        ),
        compute::make_zip_iterator(
            boost::make_tuple(src1.End(), src2.End(), accumulator.End())
        ),
        get<2>(_1) = get<0>(_1) * get<1>(_1),
        queue
    );

    auto res = 0.f;
    compute::reduce(accumulator.Begin(), accumulator.End(), &res, compute::plus<float>(), queue);
    return res;
}

void ProjectImage(const GPUScalarField & src, GPUScalarField & dst, const Matrix<4, 4> & transfo,
                  compute::command_queue & queue) {
    assert(src.NX() == dst.NX());
    assert(src.NY() == dst.NY());
    assert(src.NZ() == dst.NZ());

    compute::array<float, 16> deviceTransfo(queue.get_context());
    compute::copy(&transfo[0][0], &transfo[0][0] + 16, deviceTransfo.begin(), queue);

    auto kernel = GetProgram().create_kernel("projectImage");
    size_t workDim[3] = { (size_t) src.NX(), (size_t) src.NY(), (size_t) src.NZ() };
    kernel.set_arg(0, dst.Buffer());
    kernel.set_arg(1, src.Buffer());
    kernel.set_arg(2, deviceTransfo);
    kernel.set_arg(3, src.Dims());

    queue.enqueue_nd_range_kernel(kernel, 3, NULL, workDim, NULL);
}