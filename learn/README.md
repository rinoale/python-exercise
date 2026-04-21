# Python Syntax Course

A structured plan to learn Python syntax and features, from beginner basics
to modern (3.10+) features. Lesson files will be created later — this is the
curriculum plan only.

## Prerequisites
- `pyenv local 3.14.2`
- A terminal and a text editor

## How to run

Lessons are interactive — they alternate between explanations and quizzes
where you type a Python expression and get graded.

```bash
python learn/01_values_and_variables.py
python learn/02_numbers_and_operators.py
# ... etc
```

In a quiz prompt:
- `?`  show the hint
- `s`  skip this quiz
- `.`  on its own line (or Ctrl-D) — submit a multi-line answer

## Current status — all 26 lessons implemented

| # | Topic | File |
|---|---|---|
| 01 | Values and variables | `01_values_and_variables.py` |
| 02 | Numbers and operators | `02_numbers_and_operators.py` |
| 03 | Strings | `03_strings.py` |
| 04 | Collections (list/tuple/dict/set) | `04_collections.py` |
| 05 | Control flow | `05_control_flow.py` |
| 06 | Functions | `06_functions.py` |
| 07 | I/O basics | `07_io_basics.py` |
| 08 | Advanced functions (closures, nonlocal) | `08_advanced_functions.py` |
| 09 | Comprehensions and generators | `09_comprehensions_and_generators.py` |
| 10 | Decorators | `10_decorators.py` |
| 11 | Exceptions | `11_exceptions.py` |
| 12 | Modules and packages | `12_modules_and_packages.py` |
| 13 | Classes and OOP | `13_classes_and_oop.py` |
| 14 | Dunder methods | `14_dunder_methods.py` |
| 15 | Context managers | `15_context_managers.py` |
| 16 | Standard library tour | `16_stdlib_tour.py` |
| 17 | Type hints | `17_type_hints.py` |
| 18 | Dataclasses | `18_dataclasses.py` |
| 19 | ABCs and protocols | `19_abcs_and_protocols.py` |
| 20 | OOP internals (descriptors, slots) | `20_oop_internals.py` |
| 21 | Iterators and lazy evaluation | `21_iterators_and_lazy.py` |
| 22 | Concurrency | `22_concurrency.py` |
| 23 | Pattern matching | `23_pattern_matching.py` |
| 24 | Modern syntax (3.8+ onwards) | `24_modern_syntax.py` |
| 25 | Packaging and tooling | `25_packaging_and_tooling.py` |
| 26 | Performance and profiling | `26_performance_and_profiling.py` |

## What you already touched

Based on files in the parent directory, you've already had some exposure to:
- Basic printing, string formatting, indexing
- Lists, slicing, tuples, destructuring
- Sets, dicts, `zip()`
- `for` loops, list comprehensions
- Functions, `lambda`, generators, decorators
- Class definition basics, scope, module import

The beginner track revisits these with structure. Intermediate/advanced go
deeper into areas you haven't explored yet.

---

## Level 1 — Beginner (the foundation)

Goal: comfortably read and write small Python scripts.

### 1.1 Values and variables
- Literals: `int`, `float`, `bool`, `str`, `None`
- Variable assignment, naming conventions (`snake_case`)
- Type conversion: `int()`, `float()`, `str()`, `bool()`
- `type()` and `isinstance()`

### 1.2 Numbers and operators
- Arithmetic: `+ - * / // % **`
- Comparison: `== != < <= > >=`
- Boolean: `and`, `or`, `not`
- Truthy/falsy values (0, "", [], {}, None are falsy)

### 1.3 Strings
- Quotes: `'...'`, `"..."`, `"""..."""`
- f-strings: `f"x = {x}"`, format specifiers (`:.2f`, `:>10`, `:0>3`)
- Common methods: `.upper()`, `.lower()`, `.strip()`, `.split()`, `.join()`,
  `.replace()`, `.startswith()`, `.endswith()`, `.find()`
- Slicing: `s[1:4]`, `s[::-1]`

### 1.4 Collections
- **list**: ordered, mutable — `append`, `extend`, `pop`, `sort`, slicing
- **tuple**: ordered, immutable — unpacking, when to prefer over list
- **dict**: key→value — `.get()`, `.keys()`, `.values()`, `.items()`
- **set**: unique, unordered — `|`, `&`, `-`, `^`

### 1.5 Control flow
- `if` / `elif` / `else`
- `for` loops, `range()`, `enumerate()`, `zip()`
- `while` loops, `break`, `continue`, `else` on loops
- Conditional expression: `x if cond else y`

### 1.6 Functions
- `def`, parameters, default values, return values
- Positional vs keyword arguments
- Docstrings (what, not how)
- Scope: local vs global (brief — deep dive in intermediate)

### 1.7 I/O basics
- `print()` arguments (`sep`, `end`, `file`)
- `input()`
- Reading/writing files with `with open(...)`

---

## Level 2 — Intermediate (the working Pythonista)

Goal: write idiomatic Python, understand classes, use the standard library.

### 2.1 Advanced functions
- `*args`, `**kwargs`
- Keyword-only and positional-only parameters (`*`, `/` in signatures)
- First-class functions, higher-order functions
- `lambda` — when to use, when not to
- Closures and `nonlocal`

### 2.2 Comprehensions and generators
- List / dict / set comprehensions
- Nested comprehensions
- Generator expressions `(x*x for x in ...)`
- `yield`, generator functions
- `yield from`

### 2.3 Decorators
- Functions that wrap functions
- `@functools.wraps`
- Decorators with arguments
- Class decorators vs function decorators (preview)

### 2.4 Errors and exceptions
- `try` / `except` / `else` / `finally`
- Raising: `raise ValueError(...)`
- Custom exception classes
- Exception chaining: `raise X from e`
- EAFP vs LBYL (Python style)

### 2.5 Modules and packages
- `import x`, `from x import y`, `as`
- `__name__ == "__main__"` pattern
- Package structure: `__init__.py`
- Relative vs absolute imports
- `pyproject.toml` / `pip` basics

### 2.6 Classes and OOP
- `class`, attributes, methods
- `self`, instance vs class attributes
- `@classmethod`, `@staticmethod`
- `@property`, getters/setters
- Inheritance, `super()`, MRO (method resolution order)
- Composition over inheritance

### 2.7 Dunder (magic) methods — deep dive
- Construction: `__init__`, `__new__`
- Representation: `__repr__`, `__str__`, `__format__`
- Comparison: `__eq__`, `__hash__`, `__lt__` (+ `functools.total_ordering`)
- Containers: `__len__`, `__getitem__`, `__setitem__`, `__contains__`, `__iter__`, `__next__`
- Callable: `__call__`
- Context manager: `__enter__`, `__exit__`
- Arithmetic: `__add__`, `__mul__`, etc.

### 2.8 Context managers
- `with` statement
- Writing one as a class (`__enter__` / `__exit__`)
- Writing one with `@contextlib.contextmanager`

### 2.9 Standard library tour
- `collections`: `Counter`, `defaultdict`, `deque`, `namedtuple`
- `itertools`: `chain`, `groupby`, `combinations`, `product`
- `functools`: `reduce`, `partial`, `cache`, `lru_cache`
- `pathlib`: modern file paths
- `datetime`, `json`, `re`

---

## Level 3 — Advanced & Modern Python

Goal: write production-quality Python; use features from 3.10+.

### 3.1 Type hints (static typing)
- Built-in generics: `list[int]`, `dict[str, int]` (3.9+)
- `Optional`, `Union` / `X | Y` (3.10+)
- `Callable`, `Iterable`, `Sequence` from `typing`
- `TypeVar`, generics, `Protocol` (structural typing)
- `Self` type (3.11+)
- `TypedDict`, `Literal`, `Final`
- Running `mypy` / `pyright`

### 3.2 Dataclasses and alternatives
- `@dataclass`: auto-generated `__init__`, `__repr__`, `__eq__`
- `frozen=True`, `slots=True` (3.10+)
- `field()`, `default_factory`
- `NamedTuple` with types
- When to use which

### 3.3 Abstract base classes & protocols
- `abc.ABC`, `@abstractmethod`
- `Protocol` — duck typing with type checks
- `runtime_checkable`
- Why protocols often beat ABCs in modern code

### 3.4 Advanced OOP internals
- Descriptors (how `@property` actually works)
- `__slots__` — memory & speed
- Metaclasses (brief — rarely needed)
- `__init_subclass__` as a lighter alternative

### 3.5 Iterators and lazy evaluation
- Iterator protocol in depth
- Infinite iterators
- `itertools.islice`, `itertools.tee`
- Memory benefits of generators vs lists

### 3.6 Concurrency
- Threads vs processes vs async — when each wins
- `threading`, `concurrent.futures.ThreadPoolExecutor`
- `multiprocessing`, `ProcessPoolExecutor`
- The GIL — what it actually blocks (CPU) vs not (I/O)
- `asyncio`: `async def`, `await`, `asyncio.gather`
- `async for`, `async with`

### 3.7 Pattern matching (3.10+)
- `match` / `case`
- Literal, capture, wildcard patterns
- Class patterns, sequence patterns, mapping patterns
- Guards: `case X if cond:`

### 3.8 Modern syntax niceties
- Walrus operator `:=` (3.8+)
- f-string improvements (3.12 allows nested quotes)
- `str.removeprefix` / `removesuffix` (3.9+)
- Dictionary merge `|` and update `|=` (3.9+)
- Exception groups `except*` (3.11+)
- `tomllib` in stdlib (3.11+)

### 3.9 Packaging and tooling
- `pyproject.toml`, `build`, `pip install -e .`
- Virtual environments: `venv`, `pyenv`, `uv`
- Formatters/linters: `ruff`, `black`
- Type checkers: `mypy`, `pyright`
- Testing: `pytest` basics, fixtures, parametrize

### 3.10 Performance & profiling
- `timeit` for microbenchmarks
- `cProfile` + `snakeviz`
- When to reach for NumPy / Cython / PyPy
- Quick wins: local variable access, `__slots__`, built-ins over loops

---

## Suggested path

1. Skim Level 1 — jump straight to anything unfamiliar. (You've seen most
   of this already in the parent `.py` files.)
2. Work Level 2 end-to-end — this is where idiomatic Python lives.
3. Cherry-pick Level 3 topics as your projects demand them (type hints
   and dataclasses first — they pay off immediately).

Once Level 2 is solid, the AI course (`../ai/COURSE.md`) will feel much
lighter on syntax friction.
