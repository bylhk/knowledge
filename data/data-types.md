# Data Types Matter in Python

Data type choices have a direct impact on speed, memory usage, and correctness in Python ML code. Most Python users default to float64 and convert types carelessly — this silently doubles memory, triggers unnecessary copies, and kills performance.

---

## The Cost of Wrong Data Types

| Mistake | Consequence |
|---------|------------|
| Using float64 when float32 is sufficient | 2x memory, 2x slower on CPU, incompatible with most GPU defaults |
| Changing a variable's dtype mid-pipeline | Triggers a full array copy — doubles memory temporarily |
| Converting between NumPy and PyTorch unnecessarily | Breaks zero-copy sharing, allocates new memory |
| Mixing dtypes in operations | Silent upcasting to float64, unexpected memory spikes |

---

## float32 vs float64

NumPy and Pandas default to float64. For most ML workloads, float32 is sufficient and significantly faster.

```python
import numpy as np

# ❌ Default — float64, slow, 2x memory
prices = np.array([1.5, 2.3, 4.1])                  # float64

# ✅ Explicit — float32, faster, half the memory
prices = np.array([1.5, 2.3, 4.1], dtype=np.float32) # float32
```

| Dtype | Size per element | Relative speed (CPU) | Precision |
|-------|-----------------|---------------------|-----------|
| float16 | 2 bytes | Fastest (but limited precision) | ~3 decimal digits |
| float32 | 4 bytes | ~1.5–2x faster than float64 | ~7 decimal digits |
| float64 | 8 bytes | Baseline | ~15 decimal digits |

### When float32 is enough

- Model training and inference (XGBoost, PyTorch, TensorFlow all default to float32)
- Feature arrays, prediction scores, probabilities
- Any value where 7 digits of precision is sufficient

### When you need float64

- Financial calculations requiring exact decimal precision
- Cumulative sums over very large arrays (float32 accumulation error grows)
- Scientific computing where precision loss compounds

### Integer types too

The same principle applies to integers — don't use int64 when int32 or int16 is enough:

```python
# ❌ Default — int64, 8 bytes per element
ids = np.array([1, 2, 3])                        # int64

# ✅ Explicit — int32, half the memory
ids = np.array([1, 2, 3], dtype=np.int32)         # int32

# ✅ Even smaller if values are bounded
flags = np.array([0, 1, 1, 0], dtype=np.int8)     # 1 byte per element
```

---

## Don't Change Data Types Mid-Pipeline

Every dtype conversion creates a full copy of the array. In a pipeline, careless conversions can double or triple memory usage.

```python
import numpy as np

# ❌ Bad — creates a copy on every conversion
data = np.random.rand(10_000_000)              # float64, ~80MB
data = data.astype(np.float32)                  # copy → float32, ~40MB (but 120MB peak)
data = data.astype(np.float64)                  # copy again → float64, ~80MB (120MB peak)

# ✅ Good — set the correct dtype from the start
data = np.random.rand(10_000_000).astype(np.float32)  # or use a generator that outputs float32
```

### Rules

- Decide the dtype at array creation — don't convert later
- If you must convert, do it once at the boundary (e.g. when loading data), not in the middle of computation
- Use `np.asarray(x, dtype=np.float32)` instead of `x.astype(np.float32)` — it avoids a copy if the dtype already matches:
  ```python
  # astype always copies (unless copy=False and dtype matches)
  y = x.astype(np.float32)

  # asarray returns the same object if dtype already matches — no copy
  y = np.asarray(x, dtype=np.float32)
  ```

---

## Zero-Copy Sharing Between NumPy Arrays

NumPy arrays can share the same underlying memory through views, slicing, and reshaping — avoiding copies entirely. Understanding when NumPy copies vs shares is critical for performance.

### Views (shared memory)

```python
import numpy as np

a = np.array([1, 2, 3, 4, 5], dtype=np.float32)

# Slicing creates a view — shared memory, no copy
b = a[1:4]
b[0] = 99
print(a)  # [1, 99, 3, 4, 5] — a is modified because b shares memory

# Reshape creates a view — shared memory, no copy
c = a.reshape(1, 5)
c[0, 0] = 42
print(a)  # [42, 99, 3, 4, 5] — a is modified

# Transpose creates a view
d = a.reshape(5, 1).T
print(np.shares_memory(a, d))  # True
```

### Copies (separate memory)

```python
# Fancy indexing always copies
a = np.array([1, 2, 3, 4, 5], dtype=np.float32)
b = a[[0, 2, 4]]              # copy! fancy index
b[0] = 99
print(a)                       # [1, 2, 3, 4, 5] — a is unchanged

# .copy() explicitly copies
c = a.copy()                   # copy!
print(np.shares_memory(a, c))  # False

# Boolean masking copies
d = a[a > 2]                   # copy!
print(np.shares_memory(a, d))  # False

# .astype() with a different dtype copies
e = a.astype(np.float64)       # copy! dtype changed
print(np.shares_memory(a, e))  # False

# .flatten() always copies; .ravel() returns a view when possible
f = a.flatten()                # copy!
g = a.ravel()                  # view (if contiguous)
print(np.shares_memory(a, f))  # False
print(np.shares_memory(a, g))  # True
```

### Quick reference

| Operation | Copy or view? | Notes |
|-----------|--------------|-------|
| `a[1:4]` (slicing) | View | Shared memory |
| `a.reshape(...)` | View | Shared if contiguous |
| `a.T` / `a.transpose()` | View | Shared memory |
| `a.ravel()` | View | View if contiguous, copy otherwise |
| `a[[0, 2]]` (fancy indexing) | Copy | Always copies |
| `a[a > 2]` (boolean mask) | Copy | Always copies |
| `a.flatten()` | Copy | Always copies — use `ravel()` instead |
| `a.copy()` | Copy | Explicit copy |
| `a.astype(other_dtype)` | Copy | Copy if dtype changes |
| `np.asarray(a, dtype=same)` | View | No copy if dtype matches |
| `np.ascontiguousarray(a)` | View or copy | View if already contiguous |

### Check if memory is shared

```python
# Use np.shares_memory() to verify
a = np.array([1, 2, 3, 4], dtype=np.float32)
b = a[::2]
print(np.shares_memory(a, b))  # True — b is a view of a

c = a[[0, 2]]
print(np.shares_memory(a, c))  # False — fancy indexing copied
```

### Avoid accidental copies in functions

```python
# ❌ Bad — boolean mask inside function creates a copy every call
def get_positive(arr):
    return arr[arr > 0]  # copy every time

# ✅ Good — use np.where or in-place operations to avoid copies
def clip_negative(arr):
    np.clip(arr, 0, None, out=arr)  # in-place, no copy
    return arr

# ✅ Good — use out= parameter to write into pre-allocated array
result = np.empty_like(a)
np.multiply(a, 2, out=result)  # no intermediate array
```

### In-place operations

Use in-place operators and the `out=` parameter to avoid allocating new arrays:

```python
a = np.array([1.0, 2.0, 3.0], dtype=np.float32)

# ❌ Creates a new array
b = a * 2

# ✅ In-place — no new allocation
a *= 2

# ✅ out= parameter — writes into existing array
np.multiply(a, 2, out=a)

# ✅ Useful for large arrays in tight loops
buffer = np.empty(10_000_000, dtype=np.float32)
for batch in batches:
    np.add(batch, offset, out=buffer[:len(batch)])
    process(buffer[:len(batch)])
```

---

## Zero-Copy Sharing Between NumPy and PyTorch

NumPy arrays and PyTorch tensors can share the same underlying memory — but only if you do it correctly. A wrong conversion silently copies the entire array.

### Zero-copy (shared memory)

```python
import numpy as np
import torch

# NumPy → Torch (zero-copy — shared memory)
np_array = np.array([1.0, 2.0, 3.0], dtype=np.float32)
tensor = torch.from_numpy(np_array)
# tensor and np_array share memory — modifying one changes the other

# Torch → NumPy (zero-copy — shared memory, CPU only)
tensor = torch.tensor([1.0, 2.0, 3.0])
np_array = tensor.numpy()
# same shared memory
```

### Triggers a copy (breaks sharing)

```python
# ❌ dtype mismatch — forces a copy
np_array = np.array([1.0, 2.0, 3.0], dtype=np.float64)
tensor = torch.from_numpy(np_array)       # shared, but float64
tensor = tensor.float()                    # copy! now float32, no longer shared

# ❌ torch.tensor() always copies
np_array = np.array([1.0, 2.0, 3.0], dtype=np.float32)
tensor = torch.tensor(np_array)            # copy! use torch.from_numpy() instead

# ❌ GPU transfer always copies
tensor_cpu = torch.from_numpy(np_array)    # shared
tensor_gpu = tensor_cpu.cuda()             # copy to GPU — no longer shared

# ❌ .numpy() on a GPU tensor — must detach and move to CPU first
tensor_gpu = torch.tensor([1.0], device="cuda")
np_array = tensor_gpu.cpu().numpy()        # two copies: GPU→CPU, then tensor→numpy
```

### Rules for zero-copy

| Operation | Zero-copy? | Notes |
|-----------|-----------|-------|
| `torch.from_numpy(np_array)` | ✅ Yes | Shared memory, same dtype |
| `tensor.numpy()` | ✅ Yes | CPU tensors only, same dtype |
| `torch.tensor(np_array)` | ❌ No | Always copies — use `from_numpy` |
| `tensor.float()` / `tensor.double()` | ❌ No | Dtype cast creates a copy |
| `tensor.cuda()` / `tensor.cpu()` | ❌ No | Device transfer always copies |
| `torch.as_tensor(np_array)` | ✅ Yes | Like `from_numpy`, avoids copy if possible |

### Best practice

Match dtypes before crossing the NumPy/PyTorch boundary:

```python
# ✅ Set float32 in NumPy, then zero-copy to PyTorch
features = np.array(raw_data, dtype=np.float32)
tensor = torch.from_numpy(features)  # zero-copy, shared memory
```

---

## Common Mistakes

### 1. Letting Pandas upcast to float64

```python
import pandas as pd

# ❌ Pandas defaults to float64
df = pd.read_csv("features.csv")
print(df["price"].dtype)  # float64

# ✅ Specify dtypes on load
df = pd.read_csv("features.csv", dtype={"price": "float32", "quantity": "int32"})
```

### 2. Mixing dtypes in NumPy operations

```python
# ❌ Silent upcast — result is float64
a = np.array([1.0, 2.0], dtype=np.float32)
b = np.array([3.0, 4.0], dtype=np.float64)
c = a + b  # float64! silently upcasted

# ✅ Ensure both arrays have the same dtype
b = np.array([3.0, 4.0], dtype=np.float32)
c = a + b  # float32
```

### 3. Repeated astype calls in a loop

```python
# ❌ Copies the array on every iteration
for batch in batches:
    batch = batch.astype(np.float32)  # copy every time
    process(batch)

# ✅ Convert once at data loading, not in the loop
batches = [b.astype(np.float32) for b in raw_batches]
for batch in batches:
    process(batch)
```

### 4. Using torch.tensor instead of torch.from_numpy

```python
# ❌ Always copies
tensor = torch.tensor(np_array)

# ✅ Zero-copy
tensor = torch.from_numpy(np_array)
```

### 5. Ignoring dtype in random generation

```python
# ❌ Default float64
noise = np.random.randn(1000000)                          # float64

# ✅ Generate in float32 directly
rng = np.random.default_rng()
noise = rng.standard_normal(1000000, dtype=np.float32)     # float32
```

---

## Memory Impact Summary

For a 10M element array:

| Dtype | Memory | Relative |
|-------|--------|----------|
| float16 | ~20 MB | 0.25x |
| float32 | ~40 MB | 0.5x |
| float64 | ~80 MB | 1x (default) |
| int8 | ~10 MB | 0.125x |
| int32 | ~40 MB | 0.5x |
| int64 | ~80 MB | 1x (default) |

Switching from float64 to float32 halves memory usage and improves cache locality — which translates directly to faster computation.
