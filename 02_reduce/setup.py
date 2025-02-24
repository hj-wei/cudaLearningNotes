import torch
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension


setup(
    name="torch_reduce_sum",
    ext_modules=[CUDAExtension(name="torch_reduce_sum", sources=["warp_sum.cu"])],
    cmdclass={"build_ext": BuildExtension},
)
