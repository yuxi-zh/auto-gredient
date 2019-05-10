
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='slice',
    ext_modules=[
        CUDAExtension('slice', [
            'slice_cuda.cpp',
            'slice_cuda_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
