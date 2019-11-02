# from setuptools import setup
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [
    Extension("cmonkeynn.dpoint",  ["pyxies/dpoint.pyx"]),
    Extension("cmonkeynn.monkeyindex",  ["pyxies/monkeyindex.pyx"]),
    Extension("cmonkeynn.ndim",  ["pyxies/ndim.pyx"]),
    Extension("cmonkeynn.toplist",  ["pyxies/toplist.pyx"]),
    Extension("cmonkeynn.topNtree",  ["pyxies/topNtree.pyx"]),
]

setup(
    name="cmonkeynn",
    cmdclass={'build_ext': build_ext},
    ext_modules=ext_modules,
    version="0.0.1",
    packages=["cmonkeynn"]
)
