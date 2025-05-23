import setuptools
import sys
import glob
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

import unlanedet

long_description = "A Tookit for lane detection based on PyTorch"

with open("requirements.txt") as file:
    REQUIRED_PACKAGES = file.read()

def get_extensions():
    extensions = []

    op_files = glob.glob('./unlanedet/layers/ops/csrc/*.c*')
    op_files_ad = glob.glob('./unlanedet/layers/ops/csrc/adnet/*.c*')
    op_files_sr = glob.glob('./unlanedet/layers/ops/csrc/srnet/*.c*')
    op_files_dcn = glob.glob('./unlanedet/layers/ops/dcn/src/conv/*.c*')
#    op_files_dcn_pool = glob.glob('./unlanedet/layers/ops/dcn/src/pool/*.c*')
    
    
    nms_ext_name = 'unlanedet.layers.ops.nms_impl'
    nms_ext_ops = CUDAExtension(
        name=nms_ext_name,
        sources=op_files
    )
    extensions.append(nms_ext_ops)

    # for adnet
    nms_ad_ext_name = 'unlanedet.layers.ops.nms_ad_impl'
    nms_ad_ext_ops = CUDAExtension(
        name=nms_ad_ext_name,
        sources=op_files_ad
    )
    extensions.append(nms_ad_ext_ops)

    nms_sr_ext_name = 'unlanedet.layers.ops.nms_sr_impl'
    nms_ad_ext_ops = CUDAExtension(
        name=nms_sr_ext_name,
        sources=op_files_sr
    )
    extensions.append(nms_ad_ext_ops)

    define_macros = []
    define_macros += [('WITH_CUDA', None)]
    extra_compile_args = {'cxx': []}
    extra_compile_args['nvcc'] = [
        '-D__CUDA_NO_HALF_OPERATORS__',
        '-D__CUDA_NO_HALF_CONVERSIONS__',
        '-D__CUDA_NO_HALF2_OPERATORS__',
    ]
    nms_dcn_ext_name = 'unlanedet.layers.ops.dcn.deform_conv_ext'
    nms_dcn_ext_ops = CUDAExtension(
        name=nms_dcn_ext_name,
        sources=op_files_dcn,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args
    )
    extensions.append(nms_dcn_ext_ops)
    
#    nms_dcn_pool_ext_name = 'unlanedet.layers.ops.dcn.deform_pool_ext'
#    nms_dcn_pool_ext_ops = CUDAExtension(
#        name=nms_dcn_pool_ext_name,
#        sources=op_files_dcn_pool
#    )
#    extensions.append(nms_dcn_pool_ext_ops)

    return extensions

setuptools.setup(
    name="unlanedet",
    version='0.0.2',
    author="kunyangzhou",
    author_email="zhoukunyangmcgill@163.com",
    description=long_description,
    long_description=long_description,
    long_description_content_type="text/plain",
    url="https://github.com/zkyntu/UnLanedet",
    packages=setuptools.find_packages(),
    include_package_data=True,
    setup_requires=['cython', 'numpy', 'pytest-runner'],
    install_requires=REQUIRED_PACKAGES,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache 2.0 license",
        "Operating System :: OS Independent",
    ],
    tests_require=['pytest'],
    ext_modules=get_extensions(),
    cmdclass={'build_ext': BuildExtension},
    license='Apache 2.0 license',
    entry_points={'console_scripts': ['unlanedet=unlanedet.command:main', ]})
