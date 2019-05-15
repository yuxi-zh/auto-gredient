import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

hlr = os.environ['HL_ROOT']
cwd = os.getcwd()

setup(
    name='halideslice',
    ext_modules=[
        CUDAExtension(name='halideslice',
                      sources=['slice_layer_forward_affine.cpp', 'slice_layer_backward_grad_coeff.cpp',
                               'slice_layer_backward_grad_guide.cpp', 'pybind.cpp'],
                      extra_compile_args={
                          'cxx': ['-O2', '-DHL_PT_CUDA'], 'nvcc': ['-O2']},
                      extra_objects=[
                          cwd + '/slice_layer_forward_affine.a', cwd + '/slice_layer_backward_grad_coeff.a', cwd + '/slice_layer_backward_grad_guide.a'],
                      include_dirs=[os.environ['HL_ROOT'] + '/include'],
                      library_dirs=[os.environ['HL_ROOT'] + '/bin'],
                      libraries=['Halide'])

    ],
    cmdclass={
        'build_ext': BuildExtension
    })
