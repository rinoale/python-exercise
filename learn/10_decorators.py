"""Lesson 10: Decorators"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from quiz_player import QuizPlayer, check_eval, check_code, cyan

player = QuizPlayer("Lesson 10: Decorators")

player.explain("A decorator is a function that wraps a function", f"""\
  {cyan("@decorator above def  is syntax sugar for:  f = decorator(f)")}

      def loud(fn):
          def wrapper(*args, **kwargs):
              result = fn(*args, **kwargs)
              return str(result).upper()
          return wrapper

      @loud
      def greet(name):
          return f"hi, {{name}}"

      greet("alice")      → "HI, ALICE"

  Equivalent without @:
      def greet(name): ...
      greet = loud(greet)""")

player.quiz(
    "Define a decorator  double_result  that doubles whatever the wrapped function returns.\n"
    "  Then apply it to a function  f(x) = x + 1.\n"
    "  After decoration,  f(5)  should equal  12  (not 6).\n"
    "  Define BOTH  double_result  and the decorated  f  — top-level names.",
    check_code(lambda ns: (
        ns["f"](5) == 12 and ns["f"](0) == 2,
        "decorator doubles the return value",
    )),
    hint="def double_result(fn):\\n    def w(*a, **k): return fn(*a, **k) * 2\\n    return w",
    multiline=True,
)

player.explain("functools.wraps — preserve the original metadata", f"""\
  Without @wraps, the wrapper replaces the function's name and docstring:

      import functools

      def log(fn):
          @functools.wraps(fn)           # copy __name__, __doc__, etc.
          def wrapper(*args, **kwargs):
              print(f"calling {{fn.__name__}}")
              return fn(*args, **kwargs)
          return wrapper

  Always use {cyan("@functools.wraps(fn)")} on your wrapper. Otherwise
  debuggers, docs, and introspection get confused.""")

player.explain("functools.lru_cache — free memoization", f"""\
      from functools import lru_cache

      @lru_cache(maxsize=None)           # or just @cache in 3.9+
      def fib(n):
          if n < 2: return n
          return fib(n-1) + fib(n-2)

      fib(100)   # instant — without cache this is exponential

  {cyan("Note:")} arguments must be hashable (tuples yes, lists no).""")

player.quiz(
    "Use  functools.lru_cache  on a recursive  fib(n)  and compute  fib(30).\n"
    "  Expected:  832040\n"
    "  Define  fib  at module level.",
    check_code(lambda ns: (
        ns["fib"](30) == 832040 and ns["fib"](0) == 0 and ns["fib"](1) == 1,
        "memoized fib works",
    )),
    hint="from functools import lru_cache\\n@lru_cache\\ndef fib(n): ...",
    multiline=True,
)

player.explain("Decorators with arguments", f"""\
  To accept arguments, write a decorator {cyan("factory")} — a function that
  returns a decorator:

      def repeat(times):
          def decorator(fn):
              def wrapper(*args, **kwargs):
                  result = None
                  for _ in range(times):
                      result = fn(*args, **kwargs)
                  return result
              return wrapper
          return decorator

      @repeat(3)
      def tick():
          print("tock")

  Three layers: factory(args) → decorator(fn) → wrapper(*args).""")

player.quiz(
    "Write a decorator factory  only_positive  such that the wrapped function\n"
    "  returns None if its first argument is ≤ 0, else calls the function.\n"
    "  Apply it to  double(x) = x * 2.\n"
    "  double(5) → 10;  double(-3) → None;  double(0) → None.",
    check_code(lambda ns: (
        ns["double"](5) == 10 and ns["double"](-3) is None and ns["double"](0) is None,
        "only_positive guards work",
    )),
    hint="def only_positive(fn):\\n    def w(x, *a, **k): return fn(x, *a, **k) if x > 0 else None\\n    return w",
    multiline=True,
)

player.play()
