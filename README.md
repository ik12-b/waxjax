# waxjax

Lightweight helper to load HuggingFace SafeTensors checkpoints into JAX/Flax.

Install (latest from source):

```bash
pip install git+https://github.com/ik12-b/waxjax.git
```

Build and publish (recommended to test to TestPyPI first): see `RELEASE.md` or run locally:

```bash
python -m pip install --upgrade build twine
python -m build
twine upload --repository testpypi dist/*
```
