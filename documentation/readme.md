# Documentation

Good documentation lives at three levels: module docstrings (what the file does), function docstrings (what the function does and why), and inline comments (why a specific decision was made). Each level has a distinct purpose — they are not interchangeable.

---

## Module-Level Docstrings

Every module must have a comprehensive docstring at the top of the file. It is the entry point for anyone reading the module for the first time.

```python
"""
Module Title – Short Description

This module implements ...

Functions / Routines
--------------------
function_name
    One-line description.

Notes
-----
Key design decisions, data flow, or usage context.

See Also
--------
- Related module references
"""
```

### Rules

- Every module must have one — no exceptions
- The `Functions / Routines` section lists every public function with a one-line description
- The `Notes` section explains design decisions, data flow, and usage context — not just what the module contains
- The `See Also` section links to related modules so readers can navigate the codebase

---

## Function Docstrings

All public and private functions use NumPy-style docstrings. This format is machine-readable (used by documentation generators) and human-readable.

```python
def compute_rolling_average(
    values: np.ndarray,
    window: int,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute a weighted rolling average over a 1-D array.

    Applies a sliding window of fixed size, optionally weighting each
    observation by a decay factor. Positions with fewer than `window`
    preceding values are filled with np.nan.

    Parameters
    ----------
    values : np.ndarray
        Input observations, shape (n,). Must be 1-D and finite.
    window : int
        Number of preceding observations to include in each average.
        Must be >= 1.
    weights : np.ndarray | None, optional
        Decay weights applied to each position within the window, shape
        (window,). If None, uniform weights are used. Weights are
        normalised internally and do not need to sum to 1.

    Returns
    -------
    np.ndarray
        Rolling weighted averages, shape (n,). The first `window - 1`
        positions are np.nan.

    Raises
    ------
    ValueError
        If `window` is less than 1.
    ValueError
        If `weights` is provided but its length does not equal `window`.

    Notes
    -----
    Weights are normalised by their sum before application — passing
    [1, 2, 4] and [0.14, 0.29, 0.57] produce identical results.

    Examples
    --------
    >>> compute_rolling_average(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), window=3)
    array([nan, nan, 2.0, 3.0, 4.0])
    """
```

Key Python 3.12 type hint conventions used above:

| Old (`typing` module) | Modern |
|-----------------------|-------------------------------------|
| `List[int]` | `list[int]` |
| `Dict[str, float]` | `dict[str, float]` |
| `Optional[float]` | `float \| None` |
| `Tuple[int, ...]` | `tuple[int, ...]` |
| `Union[str, int]` | `str \| int` |
| `Callable[[int], str]` | `collections.abc.Callable[[int], str]` |

### Rules

- The one-line summary is mandatory — it appears in auto-generated docs and IDE tooltips
- `Parameters` and `Returns` are required for every non-trivial function
- `Raises` must document every exception the function can raise and the condition that triggers it
- `Notes` is where business rules and constraints belong — not in the code itself
- `Examples` should use doctest-style syntax for pure functions — they can be executed as tests
- Private functions (`_load_model`, `_encode_features`) require docstrings too — they are often the most complex

---

## Inline Comments

Inline comments explain **why**, not **what**. The code already shows what is happening — comments add the reasoning that the code cannot express.

### Section dividers

Use a divider line to separate logical blocks within long functions:

```python
# ------------------------------------------------------------------
# Stage 1: validate request and extract product features
# ------------------------------------------------------------------
```

### Business rules and constraints

Mark critical constraints explicitly so they are never accidentally removed:

```python
# CRITICAL: do not change window size — agreed with data science team, validated against
# 6 months of A/B test results. Larger windows smooth out genuine signal.
rolling_avg = compute_rolling_average(values, window=21)

# Business Rule: validation split must preserve class ratio — a random split would
# produce an artificially low minority class in validation and inflate AUC estimates
train, val = sampler.split(test_size=0.2, stratify=True)

# CRITICAL: null target rows must be dropped before split, not after — including them
# in the stratification pool skews the class counts used to compute split proportions
df = df.dropna(subset=[target_col])
```

### Known gaps

Use `# TODO` for known gaps — never leave silent workarounds uncommented:

```python
# TODO: weight decay schedule is hardcoded — should be loaded from training config
# once the config schema is extended in v2.1
weights = np.array([0.5, 0.75, 1.0])

# TODO: class_counts is computed on the full dataset including the validation split;
# move this calculation to after the split to avoid data leakage in reported metrics
self.class_counts = data[target_col].value_counts().to_dict()
```

### Rules

- Comment **why**, not **what** — `# increment i` is noise; `# skip boundary record to avoid double-counting` is useful
- Every non-obvious decision needs a comment — if you had to think about it, document it
- Never leave a silent workaround without a comment explaining why it exists
- `# CRITICAL` marks constraints that must never be changed without understanding the downstream impact
- `# Business Rule N:` marks rules that originate from business requirements, not technical ones

---

## Naming as Documentation

Good names reduce the need for comments. Follow these conventions consistently:

| Context | Convention | Examples |
|---------|-----------|---------|
| Variables, functions, modules | `snake_case` | `predicted_score`, `load_model`, `feature_store` |
| Classes | `PascalCase` | `Model`, `RequestHandler`, `ArrowShuffleDataset` |
| Module-level constants | `UPPER_SNAKE_CASE` | `WORKFLOW_DIR`, `ARTIFACT_ROOT` |
| Private / internal methods | `_prefix` | `_load_model`, `_encode_static_features` |

### Step-logging pattern

In inference and pipeline code, log every significant step with a consistent format so logs are self-documenting:

```python
self.logger.info("Step 1: data_load COMPLETE. rows=%s", rows_loaded)
self.logger.info("Step 2: extract_valid_closest_code COMPLETE. Current Rank: %s", _hier)
self.logger.info("Step 3: predict COMPLETE. requestId: %s", request_id)
```

The pattern `"Step N: function_name COMPLETE/PASSED"` makes log traces readable without needing to cross-reference the code.

---

## Import Organisation

Imports are grouped in this order, enforced by isort with `profile = "black"`:

```python
# 1. Standard library
import os
import time
from collections.abc import Callable  # use collections.abc, not typing, for abstract types

# 2. Third-party
import numpy as np
import pandas as pd
from fastapi import HTTPException

# 3. Internal / private packages
from custom_package import CustomModule

# 4. Local application imports
from src.model import Model
```

In Python 3.12, built-in types are directly subscriptable — `list[str]`, `dict[str, float]`, `tuple[int, ...]`. Never import `Dict`, `List`, `Tuple`, or `Optional` from `typing` — they are deprecated since 3.9 and removed in 3.12. Use `X | None` instead of `Optional[X]`.

Consistent import order means any reader knows exactly where to find a dependency without scanning the whole import block.

---

## Anti-Patterns

- **No module docstring** — a module without a docstring forces readers to scan the entire file to understand its purpose
- **Docstring that only restates the function name** — `"""Load the model."""` on `def load_model()` adds nothing
- **Commenting what, not why** — `# add 1 to i` is noise; explain the reason, not the mechanics
- **Silent workarounds** — a hack without a comment is a future bug waiting to be misunderstood
- **Missing `Raises` section** — callers need to know what exceptions to handle
- **Outdated docstrings** — a docstring that no longer matches the code is worse than no docstring; update them when the function changes
