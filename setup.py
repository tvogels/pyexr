from setuptools import setup, find_packages

setup(
  name             = "PyEXR",
  version          = "0.2.1",
  description      = "One line EXR manipulation library",
  author           = "Thijs Vogels",
  author_email     = "t.vogels@me.com",
  url              = "https://github.com/tvogels/pyexr",
  install_requires = ["OpenEXR","numpy","future"],
  packages         = find_packages(),
)
