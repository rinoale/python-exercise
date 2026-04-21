"""Lesson 06: Functions"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from quiz_player import QuizPlayer, check_eval, check_function, cyan

player = QuizPlayer("Lesson 06: Functions")

player.explain("def — basic function", f"""\
  Functions are defined with `def`. They return None if you don't
  explicitly `return`.

      def greet(name):
          return f"Hello, {{name}}!"

      greet("Alice")           → "Hello, Alice!"

  {cyan('Default arguments:')}
      def greet(name, prefix="Hi"):
          return f"{{prefix}}, {{name}}!"

      greet("Alice")            → "Hi, Alice!"
      greet("Alice", "Hey")     → "Hey, Alice!"
      greet("Alice", prefix="Hi there")   # keyword argument

  {cyan('Gotcha — mutable default:')}
      def bad(x, items=[]):   # DON'T — shared across calls!
          items.append(x)
          return items
      # Use None and build inside:
      def good(x, items=None):
          if items is None: items = []
          items.append(x)
          return items""")

player.quiz(
    "Define a function  square(x)  that returns x * x.",
    check_function("square", [((2,), 4), ((5,), 25), ((-3,), 9), ((0,), 0)]),
    hint="def square(x):\\n    return x * x",
    multiline=True,
)

player.quiz(
    'Define  greet(name, prefix="Hi")  returning  f"{prefix}, {name}!".\n'
    '  Must handle  greet("Alice")  →  "Hi, Alice!"  and  greet("Bob", "Hey")  →  "Hey, Bob!".',
    check_function("greet", [
        (("Alice",), "Hi, Alice!"),
        (("Bob", "Hey"), "Hey, Bob!"),
    ]),
    multiline=True,
)

player.explain("lambda — inline anonymous function", f"""\
  A lambda is a one-expression function. It's not special, just shorter:

      square = lambda x: x * x
      square(5)                → 25

  {cyan('Use lambdas for small callbacks:')}
      sorted(words, key=lambda s: len(s))
      list(map(lambda x: x*2, [1, 2, 3]))    → [2, 4, 6]
      list(filter(lambda x: x>0, [-1, 0, 1, 2]))   → [1, 2]

  {cyan("Don't over-use them.")} If your lambda needs multiple lines
  or a name, just use `def`.""")

player.quiz(
    "Apply  (lambda x: x * x)  to the number 7.  What's the expression?",
    check_eval(49),
    hint="(lambda x: x * x)(7)",
)

player.quiz(
    "Use map + lambda to double every item in [1, 2, 3, 4]. "
    "Return a list.  Expected: [2, 4, 6, 8]",
    check_eval([2, 4, 6, 8]),
    hint="list(map(lambda x: x*2, [1,2,3,4]))",
)

player.quiz(
    "Use filter + lambda to keep positive numbers from [-2, -1, 0, 1, 2]. "
    "Return a list.  Expected: [1, 2]",
    check_eval([1, 2]),
    hint="list(filter(lambda x: x > 0, [...]))",
)

player.explain("*args and **kwargs — flexible arguments", f"""\
      def total(*nums):                  # *nums collects positional args
          return sum(nums)
      total(1, 2, 3)          → 6

      def config(**opts):                # **opts collects keyword args
          return opts
      config(host="x", port=8080)   → {{'host': 'x', 'port': 8080}}

      def wrap(prefix, *args, **kwargs):
          return prefix, args, kwargs

  {cyan('Unpacking at the call site (mirror image):')}
      nums = [1, 2, 3]
      total(*nums)                      # same as total(1, 2, 3)
      opts = {{"host": "x", "port": 8080}}
      config(**opts)                    # same as config(host="x", port=8080)""")

player.quiz(
    "Define  total(*nums)  that returns the sum of all its arguments.",
    check_function("total", [
        ((), 0),
        ((1, 2, 3), 6),
        ((10, -5, 5), 10),
    ]),
    hint="def total(*nums):\\n    return sum(nums)",
    multiline=True,
)

player.quiz(
    "Unpack the list [1, 2, 3, 4] into sum() using * at the call site. "
    "Hmm — sum() actually takes an iterable, not *args. So instead, "
    "use max() with unpacked args. Expected result: 4",
    check_eval(4),
    hint="max(*[1, 2, 3, 4])   (same as max(1, 2, 3, 4))",
)

player.play()
