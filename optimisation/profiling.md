# Profiling Python Code

Never optimise without profiling first. Intuition about where the bottleneck is is almost always wrong — the actual slow line is usually not where you expect it. Profiling tells you exactly where time and memory are spent so you fix the right thing.

```
measure → identify bottleneck → optimise → measure again
```

---

## Types of Profiling

| Type | What it measures | Tool |
|------|-----------------|------|
| CPU time | Where the program spends time executing | `cProfile`, `py-spy`, `line_profiler` |
| Line-level CPU | Time per line inside a function | `line_profiler` |
| Memory usage | Peak memory, allocations over time | `memory_profiler`, `tracemalloc` |
| Sampling profiler | Low-overhead profiling of running processes | `py-spy` |

---

## 1. cProfile — Function-Level CPU Profiling

`cProfile` is the standard library profiler. It measures cumulative time spent in every function call. Use it first to identify which functions are the bottleneck.

```python
import cProfile
import pstats
import io

def run():
    # the code you want to profile
    data = load_data()
    features = engineer_features(data)
    scores = predict(features)
    return scores

# Profile and print top 20 functions by cumulative time
profiler = cProfile.Profile()
profiler.enable()
run()
profiler.disable()

stream = io.StringIO()
stats  = pstats.Stats(profiler, stream=stream)
stats.sort_stats("cumulative")
stats.print_stats(20)
print(stream.getvalue())
```

Or run directly from the command line without changing the code:

```bash
python -m cProfile -s cumulative my_script.py | head -30
```

### Reading the output

```
ncalls  tottime  percall  cumtime  percall  filename:lineno(function)
  1000    2.341    0.002    5.123    0.005  features.py:42(engineer_features)
     1    0.001    0.001    2.782    2.782  model.py:18(predict)
```

| Column | Meaning |
|--------|---------|
| `ncalls` | Number of times the function was called |
| `tottime` | Time spent in this function only (excluding callees) |
| `cumtime` | Total time including all functions called from here |
| `percall` | Time per call |

Focus on `cumtime` to find the overall bottleneck, `tottime` to find where time is actually spent within a function.

---

## 2. line_profiler — Line-Level CPU Profiling

Once `cProfile` identifies the slow function, `line_profiler` shows exactly which lines inside it are slow.

```bash
pip install line_profiler
```

```python
# Decorate the function you want to profile
from line_profiler import profile

@profile
def engineer_features(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    normalised = (values - values.mean()) / values.std()   # line 1
    weighted   = normalised * weights                       # line 2
    clipped    = np.clip(weighted, -3.0, 3.0)              # line 3
    return clipped
```

```bash
kernprof -l -v my_script.py
```

### Reading the output

```
Line #   Hits    Time   Per Hit   % Time  Line Contents
     1      1   45230   45230.0     89.2  normalised = (values - values.mean()) / values.std()
     2      1    3210    3210.0      6.3  weighted   = normalised * weights
     3      1    2210    2210.0      4.4  clipped    = np.clip(weighted, -3.0, 3.0)
```

`% Time` shows the proportion of the function's total time spent on each line — the highest percentage is where to focus.

---

## 3. py-spy — Sampling Profiler for Running Processes

`py-spy` attaches to a running Python process without modifying the code. It is the best tool for profiling production services or long-running jobs.

```bash
pip install py-spy

# Profile a running process by PID
py-spy top --pid 12345

# Record a flame graph
py-spy record -o profile.svg --pid 12345

# Profile a script from the start
py-spy record -o profile.svg -- python my_script.py
```

The flame graph shows the call stack over time — wide bars are where time is spent. It is the fastest way to understand a complex call hierarchy.

---

## 4. timeit — Micro-benchmarking

Use `timeit` to compare two implementations of the same function. It runs the code many times and reports the average, eliminating noise.

```python
import timeit
import numpy as np

values = np.random.rand(1_000_000).astype(np.float32)

# Compare loop vs vectorised
loop_time = timeit.timeit(
    "[(v - 0.5) / 0.2 for v in values]",
    globals={"values": values},
    number=10,
)

vec_time = timeit.timeit(
    "(values - 0.5) / 0.2",
    globals={"values": values},
    number=10,
)

print(f"Loop:       {loop_time:.3f}s")
print(f"Vectorised: {vec_time:.3f}s")
print(f"Speedup:    {loop_time / vec_time:.1f}x")
```

In a Jupyter notebook, use `%timeit` for the same result with less boilerplate:

```python
%timeit (values - 0.5) / 0.2
# 2.1 ms ± 45 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
```

---

## 5. memory_profiler — Line-Level Memory Profiling

`memory_profiler` shows memory usage per line — essential for finding where large arrays are allocated or where memory is not being released.

```bash
pip install memory_profiler
```

```python
from memory_profiler import profile

@profile
def build_feature_matrix(n_samples: int, n_features: int) -> np.ndarray:
    raw    = np.random.rand(n_samples, n_features)          # line 1
    scaled = (raw - raw.mean(axis=0)) / raw.std(axis=0)    # line 2
    clipped = np.clip(scaled, -3.0, 3.0)                   # line 3
    return clipped
```

```bash
python -m memory_profiler my_script.py
```

### Reading the output

```
Line #   Mem usage    Increment   Line Contents
     1    120.3 MiB   +76.3 MiB   raw    = np.random.rand(n_samples, n_features)
     2    196.6 MiB   +76.3 MiB   scaled = (raw - raw.mean(axis=0)) / raw.std(axis=0)
     3    272.9 MiB   +76.3 MiB   clipped = np.clip(scaled, -3.0, 3.0)
```

`Increment` shows how much memory each line allocates. Line 2 allocates a full copy because `(raw - mean) / std` creates intermediate arrays — this is where to apply `out=` or in-place operations.

---

## 6. tracemalloc — Built-in Memory Tracing

`tracemalloc` is in the standard library and traces memory allocations with full stack traces — useful for finding where memory is allocated in complex call chains.

```python
import tracemalloc
import numpy as np

tracemalloc.start()

# Code to profile
features = np.random.rand(1_000_000, 10).astype(np.float32)
scores   = features @ np.random.rand(10).astype(np.float32)

snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics("lineno")

print("Top 5 memory allocations:")
for stat in top_stats[:5]:
    print(stat)
```

---

## Profiling Workflow

```
1. Run cProfile          → find the slow function (cumtime)
2. Run line_profiler     → find the slow line inside that function
3. Run memory_profiler   → check if the slow line is also allocating memory
4. Fix the bottleneck
5. Run timeit            → confirm the fix is actually faster
6. Repeat
```

### Rules

- Always profile before optimising — never guess
- Profile with realistic data sizes — a bottleneck on 1K rows may not exist on 1M rows, and vice versa
- Measure after optimising — confirm the change actually improved performance; sometimes it does not
- Profile in isolation — remove unrelated I/O and setup from the profiled section so the numbers reflect the actual computation
- A 10% speedup on a function that takes 1ms is irrelevant — focus on the functions with the highest `cumtime`
