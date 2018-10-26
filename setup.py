from setuptools import setup

setup(name='resutils',
      version='0.1.4a',
      description='Controller for CircuitSymphony - distributed SPICE',
      url='https://github.com/nfrik/ResSymphony',
      author='Nikolay Frick',
      author_email='nvfrik@ncsu.edu',
      license='MIT',
      packages=['resutils'],
      install_requires=[
          'tqdm',
      ],
      zip_safe=False)
