import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

print(os.environ['HL_ROOT'])
print(os.getcwd())

setup(
    name='halideslice',
    ext_modules=[
        CUDAExtension(name='halideslice',
                      sources=['slice_layer_backward_grad_coeff.cpp',
                               'slice_layer_backward_grad_guide.cpp'],
                      extra_compile_args={'cxx': ['-O2'], 'nvcc': ['-O2']},
                      include_dirs=[os.environ['HL_ROOT'] + '/include'],
                      library_dirs=[
                          os.getcwd(), os.environ['HL_ROOT'] + '/bin'],
                      libraries=['slice_layer_backward_grad_coeff', 'slice_layer_backward_grad_guide', 'Halide'])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
