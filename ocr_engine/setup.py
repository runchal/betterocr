"""
Setup script for Adaptive Multi-Engine OCR Library
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="adaptive-multi-engine-ocr",
    version="1.0.0",
    author="OCR Engine Team",
    author_email="ocr@example.com",
    description="Adaptive multi-engine OCR library with learning capabilities",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/example/adaptive-ocr",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.812"
        ],
        "gpu": [
            "torch[cuda]>=2.0.0",
            "torchvision[cuda]>=0.15.0"
        ]
    },
    entry_points={
        "console_scripts": [
            "adaptive-ocr=src.main_ocr:main",
        ],
    },
    include_package_data=True,
    package_data={
        "src": ["data/*.db", "data/*.json"],
    },
)