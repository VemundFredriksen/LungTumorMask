from setuptools import setup, find_packages

setup(
    name="lungtumormask",
    packages=find_packages(),
    version='1.2.1',
    author="Svein Ole M Sevle, Vemund Fredriksen, and André Pedersen",
    url="https://github.com/VemundFredriksen/LungTumorMask",
    license="MIT",
    python_requires='>=3.7',
    install_requires=[
        'numpy<=1.23.2',
        'monai<=0.8.1',
        'lungmask@git+https://github.com/andreped/lungmask',
        'nibabel',
        'scikit-image>=0.17.0',
        'torch>=1.10.2,<=1.11',
    ],
    entry_points={
        'console_scripts': [
            'lungtumormask = lungtumormask.__main__:main'
        ]
    },
    classifiers=[
         "Programming Language :: Python :: 3.6",
         "Programming Language :: Python :: 3.7",
         "Programming Language :: Python :: 3.8",
         "Programming Language :: Python :: 3.9",
         "Programming Language :: Python :: 3.10",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
)
