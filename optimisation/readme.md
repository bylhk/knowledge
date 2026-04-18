# Python Optimisation

Optimisation in Python follows a strict hierarchy. Jumping straight to Cython or multiprocessing when the real problem is a Python loop or a wrong data structure wastes time and adds complexity. Always work top-down — the highest-level fix is almost always the cheapest to implement and the easiest to maintain.

Further reading: [pythonspeed.com](https://pythonspeed.com/) — practical articles on Python performance, memory, and production optimisation.

```
1. Algorithm & data structure   — biggest gains, zero dependencies
2. Avoid repeated work          — caching, pre-computation
3. NumPy vectorisation          — replace Python loops with C-level array ops
4. Memory layout & dtypes       — contiguous arrays, float32 over float64
5. Concurrency                  — I/O-bound: threading / asyncio; CPU-bound: multiprocessing
6. Compiled extensions          — Cython, Numba, C extensions
7. Rewrite in a faster language — Go, Rust, C++ for hot paths
```

Profile before moving to the next level. See [profiling.md](profiling.md) for how to find the actual bottleneck.

---

## 1. Algorithm & Data Structure

The fastest code is code that does less work. A better algorithm beats any low-level optimisation.

```python
# ❌ Bad — O(n²) membership test in a list
def find_duplicates(values: list[int]) -> list[int]:
    seen = []
    duplicates = []
    for v in values:
        if v in seen:           # O(n) scan on every iteration → O(n²) total
            duplicates.append(v)
        else:
            seen.append(v)
    return duplicates


# ✅ Good — O(n) with a set
def find_duplicates(values: list[int]) -> list[int]:
    seen = set()
    duplicates = []
    for v in values:
        if v in seen:           # O(1) hash lookup
            duplicates.append(v)
        else:
            seen.add(v)
    return duplicates
```

### Choose the right data structure

| Need | Use | Avoid |
|------|-----|-------|
| Membership test | `set` — O(1) | `list` — O(n) |
| Key-value lookup | `dict` — O(1) | `list` of tuples — O(n) |
| Sorted insertion | `heapq` / `sortedcontainers` | Re-sorting a list — O(n log n) |
| Deque operations | `collections.deque` — O(1) both ends | `list.insert(0, x)` — O(n) |
| Counting | `collections.Counter` | Manual dict loop |
| Ordered unique | `dict` (Python 3.7+) | `list` + dedup |

---

## 2. Avoid Repeated Work

### Cache expensive results

Use `functools.cache` (Python 3.9+) for pure functions called repeatedly with the same arguments:

```python
from functools import cache

# ❌ Bad — recomputes the same coefficient on every call
def score(entity_id: str, value: float) -> float:
    coeff = load_coefficient(entity_id)   # expensive DB lookup every time
    return coeff * value


# ✅ Good — result cached after first call
@cache
def load_coefficient(entity_id: str) -> float:
    return db.fetch(entity_id)            # called once per unique entity_id

def score(entity_id: str, value: float) -> float:
    return load_coefficient(entity_id) * value
```

Use `functools.lru_cache(maxsize=N)` when memory is a concern — limits the cache to the N most recent calls.

### Pre-compute outside loops

```python
# ❌ Bad — threshold recomputed on every iteration
for batch in batches:
    threshold = np.percentile(reference_scores, 90)   # same result every time
    flags = batch > threshold

# ✅ Good — compute once before the loop
threshold = np.percentile(reference_scores, 90)
for batch in batches:
    flags = batch > threshold
```

### Avoid redundant I/O

```python
# ❌ Bad — reads the config file on every request
def get_config_value(key: str) -> str:
    config = json.load(open("config.json"))
    return config[key]


# ✅ Good — load once at module level
with open("config.json") as f:
    _CONFIG = json.load(f)

def get_config_value(key: str) -> str:
    return _CONFIG[key]
```

---

## 3. NumPy Vectorisation — Replace Python Loops

Python loops over arrays are the single biggest performance anti-pattern in ML code. NumPy operations execute C-level loops internally — 10–100x faster than equivalent Python loops.

```python
import numpy as np

# ❌ Bad — Python loop, ~100x slower
def normalise_loop(values: np.ndarray) -> np.ndarray:
    result = np.empty_like(values)
    mean = values.mean()
    std  = values.std()
    for i in range(len(values)):
        result[i] = (values[i] - mean) / std
    return result


# ✅ Good — vectorised, single C-level operation
def normalise(values: np.ndarray) -> np.ndarray:
    mean = values.mean()
    std  = values.std()
    return (values - mean) / std
```

### Broadcasting eliminates explicit loops

```python
# ❌ Bad — loop over rows
scores = np.empty((n_samples, n_features))
for i in range(n_samples):
    scores[i] = features[i] * weights    # element-wise per row

# ✅ Good — broadcasting applies weights to all rows simultaneously
scores = features * weights              # weights shape (n_features,) broadcasts over (n_samples, n_features)
```

### numpy.vectorize — apply a Python function element-wise

`numpy.vectorize` wraps a scalar Python function so it can accept array inputs. It is useful when you have a function with complex conditional logic that is difficult to express as pure NumPy array operations.

```python
import numpy as np

# Scalar function with branching logic — hard to vectorise manually
def classify_score(score: float, low: float, high: float) -> str:
    if score < low:
        return "below"
    elif score > high:
        return "above"
    return "within"


# ❌ Bad — Python loop over array
scores = np.array([0.2, 0.5, 0.8, 0.95])
result = [classify_score(s, 0.3, 0.7) for s in scores]


# ✅ Good — vectorize wraps the scalar function for array input
vec_classify = np.vectorize(classify_score)
result = vec_classify(scores, 0.3, 0.7)
# array(['below', 'within', 'above', 'above'], dtype='<U6')
```

**Important caveat:** `numpy.vectorize` is a convenience wrapper — it still calls the Python function once per element internally. It is not faster than a Python loop for computation. Its value is cleaner syntax and broadcasting support, not speed.

For performance-critical paths, replace conditional logic with NumPy operations directly:

```python
# ✅ Fastest — pure NumPy, no Python function call per element
def classify_scores_fast(
    scores: np.ndarray,
    low: float,
    high: float,
) -> np.ndarray:
    result = np.full(scores.shape, "within", dtype=object)
    result[scores < low]  = "below"
    result[scores > high] = "above"
    return result
```

Use `numpy.vectorize` when the logic is complex and readability matters more than raw speed. Use direct NumPy operations when the function is called in a tight loop or on large arrays.

```python
# ❌ Creates two intermediate arrays
result = np.exp(values) / np.sum(np.exp(values))

# ✅ Write into pre-allocated buffer — no intermediate allocation
buffer = np.empty_like(values)
np.exp(values, out=buffer)
buffer /= buffer.sum()
```

See [data/data-types.md](../data/data-types.md) for dtype selection and zero-copy patterns.

---

## 3b. Avoid Unnecessary Copies

Every time a variable is reassigned to a transformed version of itself, Python or NumPy may allocate a new object in memory. In tight loops or large-array pipelines, unnecessary copies silently double or triple memory usage and add allocation overhead.

### Python objects — avoid rebuilding containers

```python
# ❌ Bad — builds a new list on every iteration
def accumulate(batches: list[list[float]]) -> list[float]:
    result = []
    for batch in batches:
        result = result + batch    # creates a new list object every iteration
    return result


# ✅ Good — extend mutates in place, no new list allocated
def accumulate(batches: list[list[float]]) -> list[float]:
    result = []
    for batch in batches:
        result.extend(batch)       # in-place, no copy
    return result
```

### NumPy — use views instead of copies where possible

> **Warning — views share memory with the original array.** If any downstream code modifies the view, the original is also modified silently. Always confirm that neither the caller nor the callee will mutate the array for a different purpose before using a view. When in doubt, call `.copy()` explicitly — a silent mutation bug is far more expensive to debug than an unnecessary copy.
>
> ```python
> a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
> b = a[:]        # view — shares memory with a
> b[0] = 99.0     # also modifies a — silent bug if a is used later
> print(a)        # [99. 2. 3.] — unexpected
>
> # ✅ Copy when downstream code may mutate the array for a different purpose
> b = a.copy()
> b[0] = 99.0     # a is unchanged
> print(a)        # [1. 2. 3.] — safe
> ```

```python
import numpy as np

a = np.random.rand(1_000_000).astype(np.float32)

# ❌ Bad — astype always allocates a new array even if dtype already matches
b = a.astype(np.float32)             # copy, even though dtype is the same

# ✅ Good — asarray returns the original object if dtype already matches
b = np.asarray(a, dtype=np.float32)  # no copy if dtype matches

# ❌ Bad — flatten always copies
flat = a.reshape(100, -1).flatten()  # copy

# ✅ Good — ravel returns a view when the array is contiguous
flat = a.reshape(100, -1).ravel()    # view, no copy
```

### Avoid reassigning large arrays in loops

```python
# ❌ Bad — new array allocated on every iteration
result = np.zeros(n, dtype=np.float32)
for i, batch in enumerate(batches):
    result = result + batch          # allocates a new array each time

# ✅ Good — in-place addition, no new allocation
result = np.zeros(n, dtype=np.float32)
for batch in batches:
    result += batch                  # mutates result in place
```

### Avoid copying when slicing is enough

```python
# ❌ Bad — copies the first 1000 rows into a new array
subset = features[:1000].copy()
process(subset)

# ✅ Good — pass a view directly; only copy if the callee must own the data
process(features[:1000])            # view — no allocation
```

### String concatenation in loops

Python strings are immutable — every `+` creates a new string object. In a loop this is O(n²).

```python
# ❌ Bad — O(n²) string copies
def build_report(lines: list[str]) -> str:
    report = ""
    for line in lines:
        report = report + line + "\n"   # new string object every iteration
    return report


# ✅ Good — join allocates once
def build_report(lines: list[str]) -> str:
    return "\n".join(lines)
```

### Rules

- Use `+=` and in-place NumPy operators (`np.add(..., out=...)`) instead of `= a + b` inside loops
- Use `np.asarray` instead of `astype` when the dtype may already match
- Use `ravel()` instead of `flatten()` — `ravel()` returns a view when possible
- Pass array slices as views to functions; only call `.copy()` when the callee must own independent data
- Use `str.join()` instead of string concatenation in loops

---

## 4. Memory Layout

### Contiguous arrays are faster

NumPy operations on C-contiguous arrays (row-major) are faster because they access memory sequentially, maximising CPU cache usage.

```python
a = np.random.rand(1000, 1000).astype(np.float32)

# ✅ C-contiguous — row access is sequential in memory
row = a[0, :]                        # fast — sequential memory access

# ❌ Fortran-contiguous — column access is sequential, row access is not
a_f = np.asfortranarray(a)
row = a_f[0, :]                      # slower — strided memory access

# Check contiguity
print(a.flags["C_CONTIGUOUS"])       # True
print(a_f.flags["C_CONTIGUOUS"])     # False

# Force contiguous copy when needed (e.g. after transpose)
a_t = np.ascontiguousarray(a.T)
```

### Avoid memory spikes from intermediate arrays

```python
# ❌ Bad — three full arrays allocated: a*b, (a*b)+c, result
result = a * b + c

# ✅ Good — in-place operations, no intermediate allocation
np.multiply(a, b, out=a)    # a = a * b  (in-place)
np.add(a, c, out=a)         # a = a + c  (in-place)
result = a
```

---

## 5. Concurrency

Python has a Global Interpreter Lock (GIL) — only one thread executes Python bytecode at a time. The right concurrency model depends on whether the bottleneck is I/O-bound or CPU-bound.

```
I/O-bound (network, disk, DB)  → threading or asyncio
CPU-bound (computation)        → multiprocessing or compiled extensions
```

### I/O-bound — asyncio

```python
import asyncio
import httpx

# ❌ Bad — sequential, each request waits for the previous
async def fetch_all_sequential(urls: list[str]) -> list[str]:
    results = []
    async with httpx.AsyncClient() as client:
        for url in urls:
            response = await client.get(url)
            results.append(response.text)
    return results


# ✅ Good — concurrent, all requests in flight simultaneously
async def fetch_all_concurrent(urls: list[str]) -> list[str]:
    async with httpx.AsyncClient() as client:
        responses = await asyncio.gather(*[client.get(url) for url in urls])
    return [r.text for r in responses]
```

### CPU-bound — multiprocessing

```python
from concurrent.futures import ProcessPoolExecutor
import numpy as np

def process_batch(batch: np.ndarray) -> np.ndarray:
    # CPU-intensive computation
    return np.fft.fft(batch)


# ❌ Bad — sequential, one core used
results = [process_batch(b) for b in batches]

# ✅ Good — parallel across CPU cores
with ProcessPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(process_batch, batches))
```

### Threading for releasing the GIL

NumPy and I/O operations release the GIL — threading works for these even though it does not for pure Python:

```python
from concurrent.futures import ThreadPoolExecutor

# NumPy releases the GIL — threading is effective here
with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(np.fft.fft, batches))
```

---

## 6. Compiled Extensions

When NumPy vectorisation cannot express the algorithm (loop-carried dependencies, custom iteration patterns), compiled extensions provide near-C speed.

| Tool | Best for | Effort |
|------|----------|--------|
| **Numba** | Drop-in JIT for NumPy-heavy functions — zero rewrite | Low |
| **Cython** | Loop-heavy algorithms, full control over C types | Medium |
| **C extension** | Maximum control, integrating existing C libraries | High |

### Numba — zero-rewrite JIT

```python
from numba import njit
import numpy as np

# ❌ Pure Python loop — slow
def cumulative_max(values: np.ndarray) -> np.ndarray:
    result = np.empty_like(values)
    result[0] = values[0]
    for i in range(1, len(values)):
        result[i] = max(result[i - 1], values[i])
    return result


# ✅ Numba JIT — same code, compiled to machine code on first call
@njit
def cumulative_max_fast(values: np.ndarray) -> np.ndarray:
    result = np.empty_like(values)
    result[0] = values[0]
    for i in range(1, len(values)):
        result[i] = max(result[i - 1], values[i])
    return result
```

Numba compiles the function to LLVM machine code on the first call. Subsequent calls run at near-C speed with no Python overhead.

See [cython.md](cython.md) for the full Cython guide.

---

## 7. Rewrite Hot Paths in a Faster Language

When a bottleneck is proven, isolated, and performance-critical, rewriting it in Go, Rust, or C++ and calling it from Python via a binding is the final option.

```
Python (orchestration) → calls → Go/Rust/C++ (hot path) via ctypes / cffi / PyO3
```

This is the highest-effort option and should only be considered when:
- Profiling confirms the bottleneck is in a small, isolated function
- NumPy vectorisation and Cython have been tried and are insufficient
- The function is called millions of times per second

See [language/readme.md](../language/readme.md) for the language selection guide.

---

## Decision Framework

```
Is the algorithm optimal?           ──no──→  Fix the algorithm first
        │ yes
Is work being repeated?             ──yes──→  Cache or pre-compute
        │ no
Is there a Python loop over arrays? ──yes──→  NumPy vectorisation
        │ no
Is memory layout causing slowness?  ──yes──→  Contiguous arrays, float32, in-place ops
        │ no
Is the bottleneck I/O-bound?        ──yes──→  asyncio / threading
        │ no
Is the bottleneck CPU-bound?        ──yes──→  multiprocessing → Numba → Cython
        │ no
Is it a proven isolated hot path?   ──yes──→  Rewrite in Go / Rust / C++
```

---

## Anti-Patterns

| Anti-pattern | Fix |
|-------------|-----|
| Optimising before profiling | Profile first — [profiling.md](profiling.md) |
| Python loop over a NumPy array | Vectorise with NumPy broadcasting or ufuncs |
| Recomputing the same value in a loop | Pre-compute before the loop or use `@cache` |
| `float64` everywhere by default | Use `float32` for ML workloads — see [data/data-types.md](../data/data-types.md) |
| Threading for CPU-bound work | Use `multiprocessing` or Numba/Cython |
| Cython before trying NumPy | NumPy vectorisation solves most bottlenecks without compilation |
| Intermediate array allocations in tight loops | Use `out=` parameter and in-place operators |
| Unnecessary array copies in loops | Use `+=`, `np.asarray`, `ravel()` over `flatten()` |
| String concatenation in a loop | Use `str.join()` |
