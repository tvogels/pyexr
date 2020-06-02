from setuptools import setup, find_packages

setup(
    name="PyEXR",
    version="0.3.8",
    description="One line EXR manipulation library",
    author="Thijs Vogels",
    author_email="t.vogels@me.com",
    url="https://github.com/tvogels/pyexr",
    install_requires=["OpenEXR", "numpy", "future"],
    dependency_links=[
        "https://github.com/jamesbowman/openexrpython/tarball/master#egg=OpenEXR-1.3.0"
    ],
    packages=find_packages(),
)
