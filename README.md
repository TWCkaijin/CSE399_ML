# prerequirement

## Install dependencies

```bash
uv sync
```

If you got local package not found error, run:

```bash
uv pip install -e .       # 安裝本地子套件（editable）
```

## add entry points

If you want to make a shortcut call of a file, add the following to `pyproject.toml`:

```toml
[project.scripts]
short_cut = "dir1.dir2.file:function"
```

For example, in this project, call the `main()` function of `HW1/p1/p1_a.py` by calling `uv run p1_a`. To do so, add the following to `pyproject.toml`:

```toml
[project.scripts]
p1_a = "HW1.p1.p1_a:main"
```

（有問題再來問我）
