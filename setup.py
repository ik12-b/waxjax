from pathlib import Path
from setuptools import setup, find_packages

here = Path(__file__).parent
long_description = (here / "README.md").read_text(encoding="utf-8") if (here / "README.md").exists() else ""

setup(
    name="waxjax",
    version="0.1.0",
    description="Load HuggingFace SafeTensors models to JAX/Flax",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Name",
    license="Apache-2.0",
    url="https://github.com/ik12-b/waxjax",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "safetensors>=0.4.0",
        "numpy>=1.24",
        "jax>=0.4.0",
    ],
    extras_require={
        "torch": ["torch>=2.0"],
        "flax": ["flax>=0.7"],
        "all": ["torch>=2.0", "flax>=0.7", "ml_dtypes"],
        "dev": ["pytest", "black", "ruff"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)