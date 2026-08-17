#!/usr/bin/env python
# Copyright (c) Megvii, Inc. and its affiliates. All Rights Reserved

import re
import setuptools
import glob
import os
from os import path
from Cython.Build import cythonize
from setuptools import Extension
import numpy

try:
    import torch
    from torch.utils.cpp_extension import CppExtension, BuildExtension
except Exception:
    torch = None
    CppExtension = None
    BuildExtension = None


def get_extensions():
    if CppExtension is None:
        return []

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

build_ext_mode = os.environ.get("AEROTRACK_BUILD_EXT", "auto").lower()
is_arm_platform = any(token in os.uname().machine.lower() for token in ("aarch64", "arm64", "armv8"))
disable_native_extensions = build_ext_mode in {"0", "false", "no", "off"} or (build_ext_mode == "auto" and is_arm_platform)

if disable_native_extensions:
    ext_modules = []

# 2. On déclare notre nouvelle extension Cython NWD avec les nouveaux chemins
if not disable_native_extensions:
    nwd_extension = Extension(
        name="aerotrack.tracker.cython_nwd",
        sources=["aerotrack/tracker/cython_nwd.pyx"],
        include_dirs=[numpy.get_include()]
    )

# 3. On fusionne les deux processus de compilation
if not disable_native_extensions:
    ext_modules.extend(cythonize([nwd_extension], compiler_directives={'language_level': "3"}))
else:
    print("AeroTrack: native extensions disabled for this platform; using Python fallbacks.")

setup_kwargs = {
    "name": "aerotrack",
    "version": version,
    "author": "basedet team",
    "python_requires": ">=3.6",
    "long_description": long_description,
    "ext_modules": ext_modules,
    "classifiers": ["Programming Language :: Python :: 3", "Operating System :: OS Independent"],
    "packages": setuptools.find_namespace_packages(),
}

if BuildExtension is not None and not disable_native_extensions:
    setup_kwargs["cmdclass"] = {"build_ext": BuildExtension}

setuptools.setup(
    **setup_kwargs,
)