from setuptools import setup

<<<<<<< HEAD
version = '0.3.6f'
=======
version = '0.3.6i'
>>>>>>> ed3fcd78afd9bf014daa4b55ee8fa30bf5e84ce0

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
