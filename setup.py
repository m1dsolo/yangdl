from setuptools import (
    setup,
    find_packages,
)


setup(
    name='yangdl',
    version='0.1.0',
    packages=find_packages(),
    url='https://github.com/m1dsolo/yangdl',
    license='MIT',
    author='m1dsolo',
    author_email='yx1053532442@gmail.com',
    description='A simple pytorch-based framework for multi-fold train, val, test, predict!',
    long_description='https://github.com/m1dsolo/yangdl',
    install_requires=[
        'numpy',
        'scikit-learn',
        'rich',
        'tensorboard',
        'torch',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
    ],
    python_requires='>=3.10',
)
