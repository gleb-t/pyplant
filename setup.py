from setuptools import setup

setup(name='pyplant',
      version='0.5.3',
      description='Python function pipeline',
      license='MIT',
      author='gleb-t',
      url='https://github.com/gleb-t/pyplant',

      packages=['pyplant', 'pyplant.test'],
      zip_safe=False, install_requires=['dateutils', 'numpy'],
      classifiers=[
            'Development Status :: 4 - Beta',
            'Intended Audience :: Developers',
            'Topic :: Software Development :: Libraries',
            'License :: OSI Approved :: MIT License',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
      ],
)
