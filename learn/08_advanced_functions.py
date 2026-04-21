"""Lesson 08: Advanced Functions — closures, nonlocal, kw-only/pos-only"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from quiz_player import QuizPlayer, check_eval, check_function, check_code, cyan

player = QuizPlayer("Lesson 08: Advanced Functions")

player.explain("Keyword-only and positional-only parameters", f"""\
  {cyan("Keyword-only: everything after * must be passed by name.")}

      def greet(msg, *, name):           # `name` is keyword-only
          return f"{{msg}}, {{name}}!"
      greet("Hi", name="Alice")          # OK
      greet("Hi", "Alice")               # TypeError

  {cyan("Positional-only: everything before / must be positional (3.8+).")}

      def divide(a, b, /):               # `a`, `b` are positional-only
          return a / b
      divide(10, 2)                      # OK → 5.0
      divide(a=10, b=2)                  # TypeError

  {cyan("Use them to lock API shape: positional for math-like functions,")}
  {cyan("keyword for functions with many optional arguments.")}""")

player.quiz(
    'Define  make_greeting(msg, *, name)  returning  f"{msg}, {name}!".\n'
    '  It must reject positional `name`.',
    check_function("make_greeting", [
        (("Hi",), {"name": "Alice"}, "Hi, Alice!"),
        (("Hey",), {"name": "Bob"}, "Hey, Bob!"),
    ]),
    multiline=True,
)

player.explain("Closures — inner functions capture outer variables", f"""\
  When an inner function refers to a name in an enclosing scope, that
  name gets {cyan("captured")}. The inner function keeps a reference —
  even after the outer function returns.

      def make_multiplier(n):
          def multiply(x):
              return x * n      # `n` is captured
          return multiply

      double = make_multiplier(2)
      double(5)          → 10

  {cyan("Each call creates a new closure:")}
      triple = make_multiplier(3)
      double(5)          → 10      (still sees n=2)
      triple(5)          → 15      (sees n=3)""")

player.quiz(
    "Define  make_adder(n)  returning a function that adds n to its argument.\n"
    "  Example:  make_adder(10)(5) → 15",
    check_code(lambda ns: (
        (ns["make_adder"](10)(5) == 15 and ns["make_adder"](3)(4) == 7),
        "works" if ns["make_adder"](10)(5) == 15 else f"make_adder(10)(5) gave {ns['make_adder'](10)(5)}",
    )),
    hint="def make_adder(n):\\n    def add(x): return x + n\\n    return add",
    multiline=True,
)

player.explain("nonlocal — rebinding a captured variable", f"""\
  Reading a captured variable works automatically. {cyan("Assigning")}
  to it creates a new local unless you say {cyan("nonlocal")}.

      def counter():
          count = 0
          def inc():
              nonlocal count      # without this: UnboundLocalError
              count += 1
              return count
          return inc

      c = counter()
      c(); c(); c()       → 1, 2, 3

  {cyan("global vs nonlocal:")}
      global    — rebind a module-level name
      nonlocal  — rebind a name in the nearest enclosing function scope""")

player.quiz(
    "Define  make_counter()  that returns a function; each call returns 1, 2, 3, ...\n"
    "  The counter should NOT be visible outside the closure.",
    check_code(lambda ns: (
        (lambda c: (c() == 1 and c() == 2 and c() == 3))(ns["make_counter"]()),
        "counts correctly",
    )),
    hint="need `nonlocal` to mutate the captured count",
    multiline=True,
)

player.explain("Mutable default arguments — a classic trap", f"""\
  Defaults are evaluated {cyan("once")}, at function-definition time.
  A mutable default (list, dict, set) is SHARED across all calls:

      def append_one(x, items=[]):    # DANGER
          items.append(x)
          return items

      append_one(1)   → [1]
      append_one(2)   → [1, 2]        # same list!
      append_one(3)   → [1, 2, 3]

  {cyan("Fix:")} use None and build inside the function:
      def append_one(x, items=None):
          if items is None: items = []
          items.append(x)
          return items""")

player.quiz(
    "Define  append_one(x, items=None)  that appends x and returns a NEW list\n"
    "  each time it's called without `items`.\n"
    "  So  append_one(1)  twice should both give [1], not [1] then [1, 1].",
    check_code(lambda ns: (
        ns["append_one"](1) == [1] and ns["append_one"](1) == [1] and ns["append_one"](9, [1, 2]) == [1, 2, 9],
        "defaults are fresh each call",
    )),
    hint="if items is None: items = []",
    multiline=True,
)

# ── Notes ──────────────────────────────────────────────

player.explain("Note A — the  /  marker (positional-only)", f"""\
  {cyan("/  is a marker, not a parameter.")} It says: every parameter
  BEFORE  /  must be passed positionally — never by keyword.

      def divide(a, b, /):
          return a / b

      divide(10, 2)        # OK    → 5.0
      divide(a=10, b=2)    # TypeError — a, b are positional-only

  Inside the body,  /  still means division. The two roles (separator
  in a signature, operator in an expression) are disambiguated by
  context — they don't interfere.

  {cyan("Added in Python 3.8 (PEP 570).")} Built-ins like  len(obj, /)
  use it to prevent  len(obj=[1,2])  style calls.""")

player.explain("Note B — the  *  marker (keyword-only)", f"""\
  {cyan("*  on its own is the mirror image of  /")} — every parameter
  AFTER  *  must be passed by keyword.

      def greet(msg, *, name):
          return f"{{msg}}, {{name}}!"

      greet("Hi", name="Alice")      # OK
      greet("Hi", "Alice")           # TypeError — name is keyword-only

  Don't confuse the {cyan("marker *")} with the {cyan("collector *args")}:

      def f(a, *, b):          # *  alone = marker; b must be keyword
      def f(a, *args, b):      # *args collects extras; b must be keyword

  {cyan("Full parameter order, when you use everything:")}
      def f(pos_only, /, normal, *args, kw_only, **kwargs): ...
         │            │  │         │       │          │
         │            │  │         │       │          └─ rest of keywords
         │            │  │         │       └── keyword-only
         │            │  │         └─ variadic positionals
         │            │  └─ regular (either way)
         │            └─ separator: everything before is positional-only
         └─ before the  /  marker""")

player.explain("Note C — annotations are NOT default values", f"""\
  In a signature, the colon introduces a {cyan("type annotation")}, not
  a default.  Defaults use  =.

      def test(name, prefix: "Hi"):    # prefix has NO default!
          ...
      test("Alice")                     # TypeError — missing `prefix`
      test.__annotations__              # → {{'prefix': 'Hi'}}

      def test(name, prefix="Hi"):      # THIS has a default
      def test(name, prefix: str = "Hi"): # annotation + default

  {cyan("Ruby vs Python — same visual, different meaning:")}
      Ruby:    def test(name, prefix: 'Hi')      # keyword arg, default 'Hi'
      Python:  def test(name, prefix: 'Hi')      # annotated, no default

  {cyan("Annotations are metadata.")} Python does not enforce them at
  runtime — any expression is accepted (that's why  prefix: 'Hi'
  didn't raise a SyntaxError). They exist for humans and tools.""")

player.explain("Note D — when to use annotations", f"""\
  {cyan("Use them for:")}
    - Public APIs / library functions — callers see types in their IDE
    - Catching bugs via  mypy  /  pyright  in CI
    - Return types when a function can return multiple types or None
    - Frameworks that read them at runtime (dataclass, pydantic, FastAPI)

  {cyan("Skip them for:")}
    - Local variables with obvious types:  count = 0   (not  count: int = 0)
    - Tiny throwaway scripts
    - Lambdas (syntax doesn't allow it anyway)

  {cyan("Rule of thumb: annotate the BOUNDARIES — signatures, class fields,")}
  {cyan("public APIs — and let inference handle the insides.")}

  {cyan("Biggest wins:")}
      def find_user(id: int) -> User | None: ...     # forces caller to handle None
      def tags(items: list[Item]) -> set[str]: ...   # element types, not bare list/set
      def retry(fn: Callable[[], T]) -> T: ...       # generic signature

  Lesson 17 goes deep on the type system (TypeVar, Protocol, Self, ...).""")

player.play()
