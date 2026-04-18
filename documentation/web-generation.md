# Documentation Web Generation

Documentation websites can be generated automatically from two sources that already exist in the codebase: **docstrings** (Python function and module docstrings) and **Markdown files** (guides, READMEs, changelogs). The generator combines both into a searchable, navigable site without manual HTML authoring.

```
docstrings (Python)  ─┐
                       ├─→ documentation generator → static HTML site
markdown files        ─┘
```

---

## Tools

| Tool | Best for | Format | Output |
|------|----------|--------|--------|
| **MkDocs + mkdocstrings** | ML/Python projects — simple setup, clean output | Markdown + NumPy/Google docstrings | Static HTML |
| **Sphinx + autodoc** | Large libraries, API references, cross-references | reStructuredText or Markdown (MyST) | HTML, PDF, ePub |
| **pdoc** | Lightweight API docs only, zero config | NumPy/Google docstrings | Static HTML |

**MkDocs with mkdocstrings** is the recommended choice for ML projects — it uses the Markdown files you already write, reads NumPy-style docstrings directly, and produces a clean site with minimal configuration.

---

## MkDocs + mkdocstrings Setup

### Install

```bash
pip install mkdocs mkdocs-material mkdocstrings[python]
```

### Project structure

```
project/
├── docs/
│   ├── index.md              # landing page
│   ├── guides/
│   │   ├── training.md       # hand-written guides
│   │   └── inference.md
│   └── api/
│       ├── model.md          # API reference pages (thin wrappers)
│       └── ladder.md
├── src/
│   └── nba_bb_regrade/
│       ├── ml/
│       │   ├── model.py      # docstrings pulled from here
│       │   └── ladder.py
│       └── ...
└── mkdocs.yml                # site configuration
```

### mkdocs.yml

```yaml
site_name: Model A
site_description: Model A — ML documentation
repo_url: https://gitlab.com/org/model-a

theme:
  name: material
  features:
    - navigation.tabs
    - navigation.sections
    - search.suggest

plugins:
  - search
  - mkdocstrings:
      handlers:
        python:
          options:
            docstring_style: numpy        # matches project convention
            show_source: true             # link to source code
            show_root_heading: true
            members_order: source         # preserve source file order
            separate_signature: true

nav:
  - Home: index.md
  - Guides:
    - Training Pipeline: guides/training.md
    - Inference Service: guides/inference.md
  - API Reference:
    - Model: api/model.md
    - Ladder: api/ladder.md
```

### API reference pages

Each API reference page is a thin Markdown file that pulls docstrings automatically using the `::: module.path` directive:

```markdown
# Model

::: model_a.model.Model
    options:
      show_source: true
      members:
        - __init__
        - preprocess
        - predict
        - postprocess
```

The generator reads the NumPy-style docstrings from the source file and renders them as formatted HTML — parameters table, returns, raises, examples, and notes all become structured sections automatically.

### Build and serve

```bash
# Serve locally with live reload
mkdocs serve

# Build static site
mkdocs build --site-dir site/

# Output: site/ directory — deploy to any static host
```

---

## Sphinx + autodoc Setup

Sphinx is more powerful for large projects with complex cross-references, but requires more configuration. Use it when you need PDF output, versioned docs, or deep interlinking between modules.

### Install

```bash
pip install sphinx sphinx-rtd-theme sphinx-autodoc-typehints myst-parser
```

### Quick start

```bash
sphinx-quickstart docs/
```

### conf.py (key settings)

```python
extensions = [
    "sphinx.ext.autodoc",        # pulls docstrings automatically
    "sphinx.ext.napoleon",       # parses NumPy and Google style docstrings
    "sphinx.ext.viewcode",       # adds links to source code
    "sphinx.ext.autosummary",    # generates summary tables
    "myst_parser",               # enables Markdown files alongside .rst
    "sphinx_autodoc_typehints",  # renders type hints in parameter tables
]

html_theme = "sphinx_rtd_theme"

# Napoleon settings — match project docstring style
napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_include_private_with_doc = True   # include _private methods if they have docstrings
napoleon_use_param = True
napoleon_use_rtype = True

# autodoc settings
autodoc_default_options = {
    "members": True,
    "undoc-members": False,      # skip functions without docstrings
    "private-members": True,     # include _private methods
    "show-inheritance": True,
}
```

### Auto-generate API stubs

```bash
# Generates .rst stub files for every module — run once, then customise
sphinx-apidoc -o docs/api src/model_a/ --force
```

### Build

```bash
make html        # builds to docs/_build/html/
make latexpdf    # builds PDF (requires LaTeX)
```

---

## How Docstrings Map to Generated Output

Given a NumPy-style docstring:

```python
class DataSampler:
    """
    Stratified sampler for imbalanced classification datasets.

    Splits a dataset into train and validation sets while preserving
    the class distribution of the target label. Supports reproducible
    sampling via a fixed random seed.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataset containing features and the target column.
    target_col : str
        Name of the binary target column to stratify on.
    seed : int, optional
        Random seed for reproducibility. Defaults to 42.

    Attributes
    ----------
    class_counts : dict[str, int]
        Row counts per class label after loading.

    Notes
    -----
    Stratification is applied at the row level using the target column.
    Rows with null values in `target_col` are dropped before splitting.

    See Also
    --------
    - evaluation.metrics — computes AUC and F1 on the validation split
    - training.runner — calls DataSampler as the first pipeline step
    """

    def split(
        self,
        test_size: float = 0.2,
        columns: list[str] | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split the dataset into stratified train and validation sets.

        Parameters
        ----------
        test_size : float, optional
            Proportion of rows to allocate to the validation set.
            Must be in (0, 1). Defaults to 0.2.
        columns : list[str] | None, optional
            Subset of feature columns to include in the output. If None,
            all columns are returned.

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            A (train, validation) tuple, both with the same column schema.

        Raises
        ------
        ValueError
            If `test_size` is not in (0, 1).
        KeyError
            If any column in `columns` does not exist in the dataset.

        Notes
        -----
        Class proportions in both splits are guaranteed to match the
        original dataset within rounding error.

        Examples
        --------
        >>> sampler = DataSampler(df, target_col="churned", seed=0)
        >>> train, val = sampler.split(test_size=0.2)
        >>> len(train) + len(val) == len(df)
        True
        """
```

The generator renders this as:

| Docstring section | Rendered as |
|------------------|-------------|
| One-line summary | Page heading / tooltip in IDE |
| Long description | Prose paragraph |
| `Parameters` | Formatted table with name, type, description |
| `Returns` | Return type and description |
| `Raises` | Exception table with conditions |
| `Notes` | Highlighted notes block |
| `Examples` | Code block, optionally runnable as doctests |

---

## Markdown Guides Alongside API Docs

Hand-written Markdown guides explain concepts, data flows, and usage patterns that docstrings cannot — they are the "why" layer above the "what" of API references.

```markdown
<!-- docs/guides/inference.md -->
# Inference Pipeline

The inference pipeline follows three stages: `preprocess → predict → postprocess`.
Each stage receives and returns a dict (the "data bag" pattern).

## Data Flow

```
request → preprocess() → predict() → postprocess() → response
              ↓               ↓              ↓
         validation,        Model          response
         fetch, transform   scoring        format
```

See the [Model API reference](../api/model.md) for full parameter documentation.
```

The `See Also` links in module docstrings and the cross-references in Markdown guides create a navigable web between conceptual guides and API references.

---

## CI/CD Integration

Generate and publish docs automatically on every merge to main:

```yaml
# GitLab CI
pages:
  stage: deploy
  script:
    - pip install mkdocs mkdocs-material mkdocstrings[python]
    - mkdocs build --site-dir public/
  artifacts:
    paths:
      - public/
  rules:
    - if: '$CI_COMMIT_BRANCH == "main"'
```

```yaml
# GitHub Actions
- name: Deploy docs
  run: |
    pip install mkdocs mkdocs-material mkdocstrings[python]
    mkdocs gh-deploy --force
```

---

## Rules

- Every public function needs a NumPy-style docstring — undocumented functions are invisible in generated docs
- Module docstrings become the landing page for each module in the API reference — write them as if they are the first thing a new team member reads
- Keep Markdown guides and docstrings in sync — a guide that references a function signature that has changed is worse than no guide
- Run `mkdocs serve` locally before merging documentation changes — rendering errors are not always obvious in raw Markdown
- Use `See Also` in module docstrings and cross-links in Markdown guides to connect the two layers
- Publish on every merge to main — stale docs are not trusted and stop being read
