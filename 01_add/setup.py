from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name="torch_add_test",
    ext_modules=[CUDAExtension(name="torch_add_test", sources=["add_cuda.cu"])],
    cmdclass={"build_ext": BuildExtension},
)
