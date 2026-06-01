---
name: fix-paddle-api
description: Systematic guide for fixing PyTorch→Paddle API incompatibilities
---

# Fixing Paddle API Differences

When porting torchmetrics code to PaddlePaddle, these are the systematic patterns that cause failures.
Check each one when a test fails with a confusing error.

## Diagnosis Flow

1. Read the traceback — identify which Paddle op or pattern failed
2. Look up the pattern below
3. Apply the fix
4. Re-run the test

## Common Patterns

### 1. `paddle.sort()` returns single tensor
```python
# PyTorch: values, indices = torch.sort(x)
# Paddle:
values = paddle.sort(x)
indices = paddle.argsort(x)
```

### 2. `paddle.max(x, axis)` returns values only
```python
# PyTorch: values, indices = x.max(dim)
# Paddle:
values = x.max(axis)
indices = x.argmax(axis)
```

### 3. `paddle.max(a, b)` is reduction, not element-wise
```python
# PyTorch: torch.max(a, b) — element-wise maximum
# Paddle:
result = paddle.maximum(a, b)
```

### 4. `.view()` requires contiguous
```python
# PyTorch: x.view(shape)  # works on non-contiguous
# Paddle:
x = x.reshape(shape)  # always works
```

### 5. `one_hot` returns float32
```python
# PyTorch: torch.one_hot(x) returns int64
# Paddle:
result = paddle.nn.functional.one_hot(x, num_classes)  # float32
result = result.cast("int64")  # if int64 needed
```

### 6. `paddle.unique` no uint8 GPU kernel
```python
# Fails on CUDA with uint8
x_int = x.cast("int32")
result = paddle.unique(x_int)
```

### 7. `paddle.trapezoid` requires float
```python
# PyTorch: torch.trapezoid(int_tensor, dx=1)
# Paddle:
result = paddle.trapezoid(x.cast("float64"), dx=1)
```

### 8. `paddle.nan_to_num` requires float32/64
```python
# Fails on float16
x = x.cast("float32")
x = paddle.nan_to_num(x)
```

### 9. `paddle.equal` returns element-wise tensor
```python
# PyTorch: torch.equal(a, b) returns bool scalar
# Paddle:
equal = paddle.all(paddle.equal(a, b))  # scalar bool
```

### 10. `argsort` propagates `requires_grad`
```python
# PyTorch: argsort never has grad
# Paddle:
indices = paddle.argsort(x).detach()
```

### 11. `scatter_add_` requires same ndim
```python
# If src is 2D and index is 1D, flatten first
src_flat = src.reshape([-1])
paddle.scatter_(target, index, src_flat)
```

### 12. `paddle.linalg.norm` with p=2 on multi-axis = spectral norm
```python
# PyTorch: torch.norm(x, p='fro') = Frobenius
# Paddle — WRONG: paddle.linalg.norm(x, p=2, axis=[1,2])  # spectral norm!
# Paddle — RIGHT:
result = paddle.linalg.norm(x, p="fro", axis=[1, 2])
```

### 13. `Tensor.gt(int)` requires tensor arg
```python
# PyTorch: x.gt(5)
# Paddle:
result = paddle.greater_than(x, paddle.to_tensor(5, dtype=x.dtype))
```

### 14. `paddle.split(n)` = n chunks, not chunk size
```python
# PyTorch: torch.split(x, 3)  # chunks of size 3
# Paddle:
chunks = paddle.chunk(x, chunks=3)  # 3 chunks
# or calculate: n_chunks = x.shape[0] // 3; paddle.split(x, num_or_sections=n_chunks)
```

### 15. Strict type promotion — no auto-promotion
```python
# PyTorch: int_tensor + 0.5 works (auto-promotes to float)
# Paddle:
result = int_tensor.cast("float32") + 0.5
```

### 16. CUDA dtype limits
```python
# Many CUDA kernels don't support bool/uint8/int8
# Cast to int32 or float32 before GPU operations
mask = mask.cast("int32")  # instead of bool
x = x.cast("float32")  # instead of float16 for some ops
```

### 17. `nn.Layer.__setattr__` drops Tensor to `_buffers`
```python
# Problem: self.my_tensor = paddle.zeros([]) goes to _buffers, but __getattr__ reads class attr
# Fix:
self.__dict__["my_tensor"] = paddle.zeros([])
```

### 18. `nn.Layer` deepcopy closure binding
```python
# Problem: deepcopy copies closures that reference original self
# Fix: use pickle round-trip
import pickle
clone = pickle.loads(pickle.dumps(self))
```

### 19. Boolean indexing on GPU
```python
# PyTorch: result = tensor[bool_mask]
# Paddle (GPU-safe):
result = paddle.masked_select(tensor, mask)
# or
indices = paddle.nonzero(mask.cast("bool")).squeeze()
result = paddle.gather(tensor, indices)
```

### 20. `paddle.bincount` requires non-negative int
```python
# PyTorch: torch.bincount(x) works with any int
# Paddle:
x = x.cast("int64")
# Ensure all values >= 0
result = paddle.bincount(x[x >= 0])
```

## Quick Test After Fix

```bash
pytest tests/unittests/<domain>/test_<metric>.py::TestClass::test_method -v --tb=short
```
