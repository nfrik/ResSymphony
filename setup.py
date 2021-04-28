from setuptools import setup

version = '5.1j'

setup(name='ressymphony',
      version=version,
      description='Controller for CircuitSymphony - distributed SPICE',
      url='https://github.com/nfrik/ResSymphony',
      author='Nikolay Frick',
      author_email='nvfrik@ncsu.edu',
      license='MIT',
      packages=['resutils'],
      install_requires=[
          'tqdm','sklearn-deap'
      ],
      zip_safe=False)
