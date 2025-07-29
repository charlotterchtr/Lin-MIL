from setuptools import setup, find_packages

setup(
    name="lin_mil",
    version="1.0.1",
    author="Charlotte Richter",
    author_email="richter_charlotte@t-online.de",
    description="Lin-MIL: Linear Attention for Whole Slide Image Analysis",
    url="https://github.com/charlotterchtr/Lin-MIL",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "torchvision",
        "einops",
        "numpy",
        "scipy",
    ],
    extras_require={
        "flash": [
            "flash-attn>=2.0.0",
            "xformers",
        ]
    },
    python_requires=">=3.7",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)