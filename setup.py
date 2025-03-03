"""
Setup configuration for the NeuraFlux package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="neuraflux",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="An autonomous LLM agent built from scratch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/neuraflux",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.24.0",
        "requests>=2.31.0",
        "tqdm>=4.65.0",
        "colorama>=0.4.6",
        "python-dotenv>=1.0.0",
        "beautifulsoup4>=4.12.0",
        "streamlit>=1.32.0",
        "streamlit-chat>=0.1.1",
    ],
    entry_points={
        "console_scripts": [
            "neuraflux=neuraflux.__main__:main",
            "neuraflux-web=neuraflux.interface.web_ui:main",
        ],
    },
) 