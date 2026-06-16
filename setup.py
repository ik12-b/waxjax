from setuptools import setup, find_packages

setup(
    name             = "safejax",
    version          = "0.1.0",
    description      = "Load HuggingFace SafeTensors models ke JAX — otomatis",
    author           = "Your Name",
    license          = "Apache 2.0",
    packages         = find_packages(),
    python_requires  = ">=3.9",
    install_requires = [
        "safetensors>=0.4.0",
        "numpy>=1.24",
        "jax>=0.4.0",
    ],
    extras_require   = {
        "torch"  : ["torch>=2.0"],
        "flax"   : ["flax>=0.7"],
        "all"    : ["torch>=2.0", "flax>=0.7", "ml_dtypes"],
        "dev"    : ["pytest", "black", "ruff"],
    },
)