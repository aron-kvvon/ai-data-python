import io
from setuptools import find_packages, setup


# Read in the README for the long description on PyPI
def long_description():
    with io.open('READ.rst', 'r', encoding='utf-8') as f:
        readme = f.read()
    return readme


setup(name='ai_dataset',
      version='0.0.1',
      description='high-level dataset for AI',
      author='aron',
      author_email='aron.kvvon@gmail.com',
      license='MIT',
      packages=find_packages(),
      classifiers=[
          'Programming Language :: Python :: 3.7',
      ],
      zip_safe=False)