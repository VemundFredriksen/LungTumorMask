from setuptools import setup, find_packages

setup(
    name="LungTumorMask",
    packages=find_packages(),
    version='1.0.1',
    author="Svein Ole M Sevle, Vemund Fredriksen, Andre Pedersen",
    url="https://github.com/VemundFredriksen/LungTumorMask",
    license="MIT",
    python_requires='>=3.6',
    install_requires=[
        'numpy',
        'monai',
        'git+https://github.com/JoHof/lungmask@master#egg=lungmask',
    ],
    entry_points={
        'console_scripts': [
            'lungtumormask = lungtumormask.__main__:main'
        ]
    }
)