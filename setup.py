from setuptools import setup, find_packages

setup(
    name="lungtumormask",
    packages=find_packages(),
    version='1.1.0',
    author="Svein Ole M Sevle, Vemund Fredriksen, and André Pedersen",
    url="https://github.com/VemundFredriksen/LungTumorMask",
    license="MIT",
    python_requires='>=3.6',
    install_requires=[
        'numpy',
        'monai',
        'lungmask@git+https://github.com/andreped/lungmask',
        'nibabel',
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
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
)
