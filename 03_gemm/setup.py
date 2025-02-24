import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="gemm_plugin",
    ext_modules=[
        CUDAExtension(
            name="gemm_plugin",
            sources=["gemm.cu"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},

)
