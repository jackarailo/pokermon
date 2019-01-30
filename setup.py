#!/usr/bin/python
from setuptools import setup, find_packages

def read_readme():
    with open('README.md', 'r') as f:
        return f.read()

setup(
      name='pokermon',
      version='0.0.1',
      url='https://github.com/jackarailo/pokermon',
      packages=find_packages(),
      test_require=[],
      description="Texas Holdem Agents",
      long_description=read_readme(),
      long_description_content_type="text/markdown",
      license='Apache License 2.0',
      python_requires='>=3.6',
      install_requires=['gym', 'torch'],
      classifiers=[
        'License :: OSI Approved :: Apache Software License',  
        'Operating System :: OS Independent',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Games/Entertainment',
        'Topic :: Games/Entertainment :: Turn Based Strategy',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules'
      ],
      keywords='rl, agent, poker, texas-holdem, gym, torch',
      author='Ioannis Arailopoulos',
      author_email='jackarailo@gmail.com',
      maintainer='Ioannis Arailopoulos'
)
