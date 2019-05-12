
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='slice',
    ext_modules=[
        CUDAExtension(name='slice',
                      sources=['slice_cuda.cpp', 'slice_cuda_kernel.cu'],
                      extra_compile_args={'cxx': ['-O2'], 'nvcc': ['-O2']})
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
