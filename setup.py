"""
Setup configuration for fraud detection package
"""
from setuptools import setup, find_packages

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="fraud-detection",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A comprehensive machine learning system for credit card fraud detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/fraud_detection",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/fraud_detection/issues",
        "Documentation": "https://github.com/yourusername/fraud_detection#readme",
        "Source Code": "https://github.com/yourusername/fraud_detection",
    },
    packages=find_packages(exclude=["tests*", "notebooks*", "docs*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "flake8>=6.1.0",
            "isort>=5.12.0",
            "mypy>=1.5.1",
        ],
        "docs": [
            "sphinx>=5.3.0",
            "sphinx-rtd-theme>=1.3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "fraud-detection-api=src.api.app:main",
            "fraud-detection-train=src.models.train:main",
            "fraud-detection-predict=src.models.predict:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yml", "*.yaml", "*.json"],
    },
    zip_safe=False,
    keywords=[
        "fraud detection",
        "machine learning",
        "credit card fraud",
        "anomaly detection",
        "classification",
        "fastapi",
        "scikit-learn",
    ],
)
