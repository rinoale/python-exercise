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

---

# Closure

## What is a closure?

A **closure** is a function that remembers variables from the scope where it
was defined, even after that scope has finished executing.

Three conditions make a closure:
1. There is a **nested function** (a function inside another function).
2. The inner function **references a variable** from the outer function.
3. The outer function **returns the inner function**.

## The simplest example

### Python

```python
def make_greeter(greeting):
    def greet(name):
        return f"{greeting}, {name}!"   # 'greeting' comes from outer scope
    return greet

hello = make_greeter("Hello")
hi = make_greeter("Hi")

hello("Alice")   # "Hello, Alice!"
hi("Alice")      # "Hi, Alice!"
```

`make_greeter` has finished running, but `hello` still remembers that
`greeting` is `"Hello"`. That remembered environment is the closure.

### JavaScript (same concept)

```javascript
function makeGreeter(greeting) {
    return function(name) {
        return `${greeting}, ${name}!`;   // 'greeting' from outer scope
    };
}

const hello = makeGreeter("Hello");
const hi = makeGreeter("Hi");

hello("Alice");   // "Hello, Alice!"
hi("Alice");      // "Hi, Alice!"
```

Identical pattern. Closures work the same way in both languages.

## How it actually works

When Python creates the inner function, it attaches a reference to the
outer variable — not a copy of the value.

```python
def make_counter():
    count = 0
    def increment():
        nonlocal count    # needed to reassign the outer variable
        count += 1
        return count
    return increment

c = make_counter()
c()   # 1
c()   # 2
c()   # 3  — 'count' persists between calls
```

```javascript
// JavaScript equivalent — no 'nonlocal' needed
function makeCounter() {
    let count = 0;
    return function() {
        count += 1;
        return count;
    };
}

const c = makeCounter();
c();   // 1
c();   // 2
c();   // 3
```

**Key difference**: Python requires `nonlocal` to reassign an outer
variable. JavaScript doesn't — `let` is freely reassignable from inner
scopes. If you only read or mutate (e.g., append to a list), Python
doesn't need `nonlocal` either.

## Why `nonlocal`?

Without `nonlocal`, Python treats `count = ...` as creating a new local
variable, shadowing the outer one:

```python
def broken_counter():
    count = 0
    def increment():
        count += 1       # UnboundLocalError! Python thinks count is local
        return count
    return increment
```

`nonlocal count` tells Python: "this name belongs to the enclosing scope,
don't create a new local."

## Reference, not copy

Closures capture the **reference** to the variable, not its value at
creation time. This is a common gotcha:

```python
functions = []
for i in range(3):
    functions.append(lambda: i)

functions[0]()   # 2  (not 0!)
functions[1]()   # 2  (not 1!)
functions[2]()   # 2
```

All three lambdas share the same `i`, which ends up as `2` after the loop.

Fix — capture the current value with a default argument:

```python
functions = []
for i in range(3):
    functions.append(lambda i=i: i)   # default arg copies current value

functions[0]()   # 0
functions[1]()   # 1
functions[2]()   # 2
```

Same gotcha exists in JavaScript with `var` (but not with `let`, which
creates a new binding per iteration).

## Why closures are useful

### 1. Data privacy (encapsulation without classes)

```python
def make_bank_account(initial):
    balance = initial
    def deposit(amount):
        nonlocal balance
        balance += amount
        return balance
    def get_balance():
        return balance
    return deposit, get_balance

deposit, get_balance = make_bank_account(100)
deposit(50)        # 150
get_balance()      # 150
# 'balance' is not accessible from outside — it's private
```

### 2. Function factories

```python
def make_multiplier(factor):
    return lambda x: x * factor

double = make_multiplier(2)
triple = make_multiplier(3)

double(5)    # 10
triple(5)    # 15
```

### 3. Decorators (built on closures)

```python
def track_calls(fn):
    def wrapper(*args, **kwargs):
        wrapper.call_count += 1
        return fn(*args, **kwargs)
    wrapper.call_count = 0
    return wrapper       # closure — remembers 'fn'
```

`wrapper` is a closure over `fn`. Every decorator you write uses closures.

### 4. Callbacks and event handlers

```python
def make_handler(button_name):
    def on_click():
        print(f"{button_name} was clicked")
    return on_click

save_handler = make_handler("Save")
save_handler()   # "Save was clicked"
```

## Closure vs class

Closures and classes can solve the same problems. Use whichever is simpler:

```python
# Closure version
def make_counter():
    count = 0
    def increment():
        nonlocal count
        count += 1
        return count
    return increment

# Class version
class Counter:
    def __init__(self):
        self.count = 0
    def increment(self):
        self.count += 1
        return self.count
```

Rule of thumb:
- **One or two functions** with shared state → closure is simpler
- **Many methods** or complex state → use a class

## Summary

| Concept | Python | JavaScript |
|---|---|---|
| Nested function captures outer variable | Yes | Yes |
| Keyword to reassign outer variable | `nonlocal` | not needed (`let`) |
| Captures reference, not value | Yes | Yes (same gotcha with loops) |
| Used in decorators | Yes (`@decorator`) | Yes (HOCs in React, etc.) |

Closures are not a language-specific trick — they are a fundamental concept
in any language with first-class functions. Once you understand them in one
language, the pattern transfers everywhere.

---

# Concurrency — Python vs Node.js

## Node.js — Single thread + event loop

```
Main Thread (the only one)
  │
  ├─ your code runs here
  ├─ callbacks run here
  ├─ async/await runs here
  │
  ↓
Event Loop (checks: "any I/O finished? any timers done?")
  │
  ├─ fs.readFile done?     → run its callback
  ├─ HTTP response arrived? → run its callback
  ├─ setTimeout expired?   → run its callback
  │
  ↓ repeat forever
```

**One thread does everything.** When you call `fs.readFile()`, Node hands
the actual file reading to the OS (or libuv's thread pool), then keeps
running your code. When the OS finishes, it puts the callback in a queue.
The event loop picks it up when the main thread is free.

```javascript
console.log("A");
setTimeout(() => console.log("B"), 0);
console.log("C");
// Output: A, C, B  — "B" waits for the event loop
```

**The consequence:** If you block the main thread with heavy computation
(a giant `for` loop), everything stops — no callbacks fire, no requests
get handled.

```javascript
// This freezes your entire server
app.get("/slow", (req, res) => {
    let sum = 0;
    for (let i = 0; i < 10_000_000_000; i++) sum += i;  // blocks everything
    res.send(sum);
});
```

**For CPU work**, Node has `worker_threads` — real OS threads, but you
manage them manually. Most Node developers never use them.

## Python — Three concurrency models

Python gives you three tools. You pick the right one for the job.

### 1. Threading (`threading`) — real threads, fake parallelism

```python
import threading

def work():
    print("working")

t = threading.Thread(target=work)
t.start()     # starts a real OS thread
t.join()      # wait for it to finish
```

Python has real OS threads, BUT the **GIL (Global Interpreter Lock)**
prevents two threads from running Python code at the same time:

```
Thread 1:  ████░░░░████░░░░████
Thread 2:  ░░░░████░░░░████░░░░
                                    ← only one runs at a time
GIL:       1111222211112222111
```

**So what's the point?** The GIL releases during I/O operations:

```python
import threading, requests

# These run truly in parallel — GIL is released during network I/O
t1 = threading.Thread(target=requests.get, args=("https://api1.com",))
t2 = threading.Thread(target=requests.get, args=("https://api2.com",))
t1.start(); t2.start()
t1.join(); t2.join()
```

**Good for:** I/O-bound work (HTTP requests, file reads, database queries).
**Bad for:** CPU-bound work (math, image processing) — GIL blocks real
parallelism.

### 2. Multiprocessing (`multiprocessing`) — real parallelism

```python
from multiprocessing import Process

def heavy_math():
    sum(range(100_000_000))

# Each process has its own Python interpreter and its own GIL
p1 = Process(target=heavy_math)
p2 = Process(target=heavy_math)
p1.start(); p2.start()    # truly parallel on different CPU cores
p1.join(); p2.join()
```

```
Process 1:  ████████████████████  (own GIL, own memory)
Process 2:  ████████████████████  (own GIL, own memory)
                                   ← both run simultaneously
```

**Good for:** CPU-bound work.
**Bad for:** Sharing data (processes have separate memory — you need
pipes/queues).

### 3. Asyncio (`asyncio`) — Node.js style, single thread + event loop

```python
import asyncio

async def fetch(url):
    print(f"start {url}")
    await asyncio.sleep(1)     # non-blocking wait (like setTimeout)
    print(f"done {url}")

async def main():
    # both run concurrently on one thread
    await asyncio.gather(
        fetch("api1"),
        fetch("api2"),
    )

asyncio.run(main())
# start api1
# start api2
# (1 second passes)
# done api1
# done api2
```

This is **the same model as Node.js** — one thread, event loop, `await`
instead of callbacks.

## Side by side

```
Node.js                          Python
─────────────────────────────────────────────────────────
Single-threaded by default       Multi-model: you choose

Event loop + callbacks/await     asyncio = same as Node
                                 threading = real threads (but GIL)
                                 multiprocessing = real parallelism

CPU work blocks everything       CPU work → use multiprocessing
                                 or release GIL (C extensions, NumPy)

worker_threads (rarely used)     threading (commonly used for I/O)

Everything is async by default   Sync by default, async opt-in
(fs, http, timers)               (you choose sync or async)

npm ecosystem: async-first       Python ecosystem: mixed
```

## The key difference in philosophy

**Node.js:** "Everything is async. You have one thread. Deal with it."
Forced simplicity — you can't accidentally share state between threads
because there's only one.

**Python:** "Here are three tools. Pick the right one."

| Problem | Python tool |
|---|---|
| 10,000 HTTP requests | `asyncio` or `threading` |
| Image processing on 8 cores | `multiprocessing` |
| Read 50 files from disk | `threading` or `asyncio` |
| Web server | `asyncio` (FastAPI, aiohttp) |
| ML training | `multiprocessing` (PyTorch DataLoader uses this) |

## The GIL — will it go away?

Python 3.13 introduced an **experimental** no-GIL mode (`--disable-gil`).
If it stabilizes, Python threads will get true parallelism like Java or
C++. But it's not production-ready yet — maybe by 3.15 or 3.16.
