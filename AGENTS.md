# AGENTS.md

## Typing guidelines

- Avoid `typing.cast` and `# type: ignore`, especially as workarounds for narrowing problems; use the style above
  instead. If one is truly unavoidable, keep it as narrow as possible and add a comment explaining why
  (e.g. `# FIXME: ...`).
- Avoid creating functions that return unions (except if it's something like `T | None`). If a function has different
  paths to return different types, it should probably be two functions or at least an overload.

### Union narrowing style

Narrow unions by asserting what a value is **not**, then use `typing.assert_type` to state (and have the type checker
verify) what it is:

```python
from typing import assert_type

transformation: BaseTransformation | dict = ...

# Don't
assert isinstance(transformation, BaseTransformation)

# Do
assert not isinstance(transformation, dict)
assert_type(transformation, BaseTransformation)
```

When all branches of a union have been handled, use `typing.assert_never` instead of a generic "should not reach here"
error, so the type checker tells us if a new type is added to the union and not handled:

```python
from typing import assert_never

if isinstance(data, DataArray):
    ...
elif isinstance(data, DataTree):
    ...
else:
    assert_never(data)  # instead of: raise RuntimeError("Should not reach here")
```

#### Why

Because of Python's (multiple-)inheritance model, a positive `isinstance` assertion can never be rejected by a type
checker, even when it contradicts the declared type:

```python
a: int = 123
assert isinstance(a, list)  # accepted; the checker infers `a: <subclass of int and list>`
```

So a positive assertion can silently force a value to a type it should never have.

#### Examples

```python
transformations: BaseTransformation | dict[str, BaseTransformation] = get_transformation(element, get_all=True)
assert not isinstance(transformations, BaseTransformation)
assert_type(transformations, dict[str, BaseTransformation])
```

Optional values are already narrowed "negatively", so keep using `is not None`:

```python
assert target_depth is not None
```
