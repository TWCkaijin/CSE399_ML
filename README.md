# prerequirement

## Install dependencies

```bash
uv sync
uv pip install -e .
```

## add entry points

Add the following to `pyproject.toml`:

```toml
[project.scripts]
short_cut = "father.child.file:function"
```
