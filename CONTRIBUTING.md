```markdown
# Contributing to QENS

Thanks for contributing! Here's how to get started.

## Setup

```bash
git clone https://github.com/ShubhamX57/QENS.git
cd QENS
pip install -e .
pip install flake8 pytest black isort
```

## Workflow

1. Create a branch: `git checkout -b feature/your-feature`
2. Make changes and test: `pytest`
3. Format code: `black qens/ && isort qens/`
4. Commit and push
5. Open a Pull Request

## Guidelines

- Follow PEP 8 (max line length: 127)
- Add docstrings to public functions
- One feature/fix per PR
- Link related issues (e.g., "Fixes #12")

## Reporting Bugs

Include: Python version, QENS version, error traceback, and minimal code to reproduce.

## Questions?

Open an [issue](https://github.com/ShubhamX57/QENS/issues).
```
