#!/usr/bin/env python
# Copyright (c) Megvii, Inc. and its affiliates. All Rights Reserved

import re
import setuptools
import glob
from os import path
import torch
from torch.utils.cpp_extension import CppExtension
from Cython.Build import cythonize
from setuptools import Extension
import numpy

torch_ver = [int(x) for x in torch.__version__.split(".")[:2]]
assert torch_ver >= [1, 3], "Requires PyTorch >= 1.3"


def get_extensions():
    this_dir = path.dirname(path.abspath(__file__))
    # Updated path to aerotrack
    extensions_dir = path.join(this_dir, "aerotrack", "layers", "csrc")

    main_source = path.join(extensions_dir, "vision.cpp")
    sources = glob.glob(path.join(extensions_dir, "**", "*.cpp"))

    sources = [main_source] + sources
    extension = CppExtension

    extra_compile_args = {"cxx": ["-O3"]}
    define_macros = []

    include_dirs = [extensions_dir]

    ext_modules = [
        extension(
            "aerotrack._C", # Updated extension name
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]

    return ext_modules


# Updated path to aerotrack's __init__.py
with open("aerotrack/__init__.py", "r") as f:
    version = re.search(
        r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]',
        f.read(), re.MULTILINE
    ).group(1)

with open("README.md", "r") as f:
    long_description = f.read()

# 1. On récupère les extensions existantes de YOLOX (désormais aerotrack)
ext_modules = get_extensions()

# 2. On déclare notre nouvelle extension Cython NWD avec les nouveaux chemins
nwd_extension = Extension(
    name="aerotrack.tracker.cython_nwd",
    sources=["aerotrack/tracker/cython_nwd.pyx"],
    include_dirs=[numpy.get_include()]
)

# 3. On fusionne les deux processus de compilation
ext_modules.extend(cythonize([nwd_extension], compiler_directives={'language_level': "3"}))

setuptools.setup(
    name="aerotrack", # Updated package name
    version=version,
    author="basedet team",
    python_requires=">=3.6",
    long_description=long_description,
    ext_modules=ext_modules, # On utilise notre liste fusionnée
    classifiers=["Programming Language :: Python :: 3", "Operating System :: OS Independent"],
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
    packages=setuptools.find_namespace_packages(),
)