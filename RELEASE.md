# Release and publish

This repository includes a GitHub Actions workflow that builds the package and uploads to PyPI when you create a GitHub Release.

Workflow behavior
- If the GitHub Release is marked *prerelease*, the workflow uploads to TestPyPI.
- Otherwise the workflow uploads to the real PyPI.

Prepare tokens
1. Create a PyPI API token on https://pypi.org/manage/account/#api-tokens (scope: whole account or project).
2. Create a TestPyPI API token on https://test.pypi.org/manage/account/#api-tokens.
3. In the repository Settings → Secrets → Actions add two secrets:
   - `PYPI_API_TOKEN` — your PyPI token value
   - `TEST_PYPI_API_TOKEN` — your TestPyPI token value

Quick local publish (test)

```bash
python -m pip install --upgrade build twine
python -m build
# upload to TestPyPI
twine upload --repository-url https://test.pypi.org/legacy/ -u __token__ -p $TEST_PYPI_API_TOKEN dist/*
```

Create a GitHub Release
- Bump the version in `setup.py` (or your source of truth).
- Create a git tag and push it:

```bash
git tag v0.1.0
git push origin v0.1.0
```

- Create a GitHub Release for that tag via the web UI or `gh release create`.
- If you mark the release as *prerelease* the workflow will upload to TestPyPI; otherwise it will upload to PyPI.

If you prefer manual control, you can also run the local build and `twine upload` commands above.
