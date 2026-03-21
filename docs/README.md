# Docs Structure

This folder stores Sphinx **source** documentation only.

## Source files

- `index.rst`: table of contents root
- `conf.py`: Sphinx configuration
- `requirements.txt` (top-level): full project dependencies (core + GUI + docs)
- `Makefile`: local build helper
- `*.md` and `*.rst`: guide/reference source pages

## Generated output

Generated HTML (`docs/_build/`) is not tracked in git.

Build locally when needed:

```bash
sphinx-build -b html docs docs/_build/html
```
