from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="ChamferDistanceAPI",
    version="2.0.0",
    ext_modules=[CUDAExtension("ChamferDistanceAPI", ["ChamferDistance.cpp", "ChamferDistance_cuda.cu",],
    				extra_compile_args=['-g']),],
    cmdclass={"build_ext": BuildExtension},
)
