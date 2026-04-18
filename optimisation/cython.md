# Cython

Cython is a superset of Python that compiles to C. It eliminates Python interpreter overhead entirely — typed Cython code runs at near-C speed while remaining callable from Python like a normal function.

Use Cython when:
- Profiling has confirmed a loop-based bottleneck that NumPy vectorisation cannot express
- The algorithm has loop-carried dependencies (each iteration depends on the previous)
- You need to release the GIL for true parallelism

Do not use Cython when NumPy vectorisation can express the algorithm — NumPy is easier to maintain and fast enough for most cases. See [profiling.md](profiling.md) to confirm the bottleneck first.

---

## How Cython Works

```
.pyx source file → Cython compiler → .c file → C compiler → .so shared library → import in Python
```

Python calls the compiled `.so` file like any other module — no change to the calling code.

---

## Setup

### Install

```bash
pip install cython
```

### Project structure

```
my_package/
├── src/
│   └── my_package/
│       ├── fast/
│       │   ├── __init__.py
│       │   └── kernels.pyx      # Cython source
│       └── ml/
│           └── model.py
├── setup.py                     # build configuration
└── pyproject.toml
```

### setup.py

```python
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        name="my_package.fast.kernels",
        sources=["src/my_package/fast/kernels.pyx"],
        include_dirs=[np.get_include()],   # required for NumPy C headers
        extra_compile_args=["-O3", "-march=native"],
    )
]

setup(
    name="my-package",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,     # disable array bounds checking in production
            "wraparound": False,      # disable negative indexing support
            "cdivision": True,        # use C division (no Python ZeroDivisionError)
        },
    ),
)
```

### Build

```bash
python setup.py build_ext --inplace   # compiles .pyx → .so in place
```

---

## Writing Cython — From Slow to Fast

### Step 1 — Pure Python baseline (valid Cython, but no speedup)

```python
# kernels.pyx
def cumulative_clip(values, lower, upper):
    result = []
    running_max = lower
    for v in values:
        clipped = min(max(v, lower), upper)
        running_max = max(running_max, clipped)
        result.append(running_max)
    return result
```

This is valid Cython but compiles to code that still calls the Python runtime on every operation — no speedup yet.

### Step 2 — Add C type declarations

Declaring C types eliminates Python object overhead on every variable access and arithmetic operation:

```python
# kernels.pyx
def cumulative_clip(
    list values,
    double lower,
    double upper,
) -> list:
    cdef int i, n
    cdef double v, clipped, running_max
    cdef list result

    n           = len(values)
    result      = [0.0] * n
    running_max = lower

    for i in range(n):
        v           = values[i]
        clipped     = min(max(v, lower), upper)
        running_max = max(running_max, clipped)
        result[i]   = running_max

    return result
```

`cdef` declares a C-level variable — no Python object, no reference counting, no GIL required for arithmetic.

### Step 3 — Use typed memory views for NumPy arrays

Typed memory views give direct C-level access to NumPy array buffers — no Python object overhead per element:

```python
# kernels.pyx
import numpy as np

def cumulative_clip(
    double[::1] values,    # 1-D C-contiguous double array
    double lower,
    double upper,
) -> np.ndarray:
    cdef int i, n
    cdef double v, clipped, running_max
    cdef double[::1] result_view

    n           = values.shape[0]
    result      = np.empty(n, dtype=np.float64)
    result_view = result
    running_max = lower

    for i in range(n):
        v           = values[i]
        clipped     = min(max(v, lower), upper)
        running_max = max(running_max, clipped)
        result_view[i] = running_max

    return result
```

`double[::1]` declares a typed memory view — `[::1]` means C-contiguous (row-major). Element access `values[i]` compiles to a direct C pointer dereference — no Python overhead.

### Step 4 — Release the GIL with `nogil`

Releasing the GIL allows multiple threads to run Cython code in parallel:

```python
# kernels.pyx
from cython.parallel import prange
import numpy as np

def batch_normalise(
    double[:, ::1] features,   # 2-D C-contiguous array (n_samples, n_features)
    double[::1] mean,
    double[::1] std,
) -> np.ndarray:
    cdef int i, j, n_samples, n_features
    cdef double[:, ::1] result_view

    n_samples  = features.shape[0]
    n_features = features.shape[1]
    result     = np.empty((n_samples, n_features), dtype=np.float64)
    result_view = result

    # prange releases the GIL and parallelises across threads
    for i in prange(n_samples, nogil=True):
        for j in range(n_features):
            result_view[i, j] = (features[i, j] - mean[j]) / std[j]

    return result
```

`prange` is Cython's parallel range — it uses OpenMP to distribute iterations across threads. Compile with OpenMP support:

```python
# setup.py — add OpenMP flags
Extension(
    ...
    extra_compile_args=["-O3", "-fopenmp"],
    extra_link_args=["-fopenmp"],
)
```

---

## Typed Memory View Reference

| Declaration | Meaning |
|-------------|---------|
| `double[::1]` | 1-D C-contiguous `float64` array |
| `float[::1]` | 1-D C-contiguous `float32` array |
| `long[::1]` | 1-D C-contiguous `int64` array |
| `double[:, ::1]` | 2-D C-contiguous `float64` array (row-major) |
| `double[::1, :]` | 2-D Fortran-contiguous `float64` array (column-major) |
| `double[:, :]` | 2-D array, any layout (slower — no contiguity guarantee) |

Always prefer `[::1]` (C-contiguous) — it matches NumPy's default layout and enables the compiler to generate sequential memory access patterns.

---

## Compiler Directives

Set globally in `setup.py` or per-file with a comment at the top of the `.pyx` file:

```python
# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
```

| Directive | Default | Effect when disabled |
|-----------|---------|---------------------|
| `boundscheck` | `True` | Removes array index bounds checking — raises no `IndexError` on out-of-bounds access |
| `wraparound` | `True` | Disables negative index support (`a[-1]`) — slightly faster positive indexing |
| `cdivision` | `False` | Uses C integer division — no Python `ZeroDivisionError`, faster |
| `language_level` | `"2"` | Set to `"3"` for Python 3 semantics |

Only disable `boundscheck` and `wraparound` after the code is tested and correct — they hide bugs in development.

---

## Numba vs Cython

For many use cases, Numba is a better first choice than Cython — it requires no compilation step, no `setup.py`, and no `.pyx` files.

```python
from numba import njit
import numpy as np

# Same algorithm as the Cython example — zero rewrite, compiled on first call
@njit
def cumulative_clip(
    values: np.ndarray,
    lower: float,
    upper: float,
) -> np.ndarray:
    n           = len(values)
    result      = np.empty(n, dtype=np.float64)
    running_max = lower

    for i in range(n):
        v              = min(max(values[i], lower), upper)
        running_max    = max(running_max, v)
        result[i]      = running_max

    return result
```

| | Numba | Cython |
|---|-------|--------|
| Setup | `pip install numba`, add `@njit` | `setup.py`, `.pyx` file, compile step |
| First-call overhead | JIT compile on first call (~seconds) | Pre-compiled — no first-call cost |
| Control over C types | Limited | Full — `cdef`, typed memory views |
| GIL release | `@njit(nogil=True)` | `nogil=True` in `prange` |
| GPU support | Yes — `@cuda.jit` | No |
| Maintenance | Easier — pure Python syntax | Harder — Cython-specific syntax |

**Use Numba when:** the function is NumPy-heavy and you want speed without a build step.

**Use Cython when:** you need full control over C types, are integrating with C libraries, or need fine-grained memory management.

---

## Annotated HTML — Identify Python Overhead

Cython can generate an annotated HTML file showing which lines still call the Python runtime (yellow = slow, white = pure C):

```bash
cython -a src/my_package/fast/kernels.pyx
# Opens kernels.html — yellow lines are Python overhead, white lines are pure C
```

The goal is to make the inner loop entirely white. Any yellow line inside a loop is a performance problem.

---

## Rules

- Profile first — only write Cython for a confirmed bottleneck
- Try Numba before Cython — same speed, far less setup
- Always use typed memory views for NumPy arrays — untyped array access is as slow as pure Python
- Use `[::1]` (C-contiguous) memory views — matches NumPy default layout
- Disable `boundscheck` and `wraparound` only after the code is tested
- Use `cython -a` to check for remaining Python overhead in the inner loop
- Keep Cython modules small and focused — one `.pyx` file per algorithm, not a full module rewrite
