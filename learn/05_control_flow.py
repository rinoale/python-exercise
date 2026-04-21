"""Lesson 05: Control Flow — if, for, while, comprehensions"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from quiz_player import QuizPlayer, check_eval, cyan

player = QuizPlayer("Lesson 05: Control Flow")

player.explain("if / elif / else", f"""\
  Indentation defines blocks — no braces, no "then".

      if temp > 30:
          print("hot")
      elif temp > 20:
          print("warm")
      else:
          print("cool")

  {cyan('Conditional expression (ternary):')}
      status = "adult" if age >= 18 else "minor"

  Python uses lazy evaluation for `and`/`or`. This idiom is common:
      name = user_input or "anonymous"    # fallback if empty/None""")

player.quiz(
    'Type a conditional expression that gives "yes" if 10 > 5, else "no".',
    check_eval("yes"),
    hint='"yes" if 10 > 5 else "no"',
)

player.quiz(
    'Type an expression: the larger of 7 and 12 (use the ternary form, no max()).',
    check_eval(12),
    hint="a if a > b else b",
)

player.explain("for loops — iterate over anything iterable", f"""\
  Lists, tuples, strings, dicts, sets, ranges, files, generators…

      for x in [1, 2, 3]: ...
      for char in "abc": ...
      for i in range(5): ...              # 0,1,2,3,4
      for i in range(1, 11): ...          # 1..10
      for i in range(10, 0, -1): ...      # 10..1 (step -1)

  {cyan('Useful helpers:')}
      enumerate(xs)     → (index, value) pairs
      zip(xs, ys)       → pairs of (x, y)
      reversed(xs)      → iterate backward
      sum, min, max, len, sorted — built-ins that take iterables""")

player.quiz(
    "Compute 1 + 2 + ... + 10 using a built-in and range().",
    check_eval(55),
    hint="sum(range(1, 11))",
)

player.quiz(
    "Type an expression producing the list [5, 4, 3, 2, 1].",
    check_eval([5, 4, 3, 2, 1]),
    hint="list(range(5, 0, -1))",
)

player.explain("Comprehensions — concise loops as expressions", f"""\
  {cyan('List comprehension:')}
      [expr for x in iterable]                 # map
      [expr for x in iterable if cond]         # map + filter

      [x*x for x in range(5)]           → [0, 1, 4, 9, 16]
      [x for x in range(10) if x%2==0]  → [0, 2, 4, 6, 8]

  {cyan('Dict and set comprehensions:')}
      {{x: x*x for x in range(4)}}    → {{0:0, 1:1, 2:4, 3:9}}
      {{x%3 for x in range(10)}}     → {{0, 1, 2}}

  {cyan('Tip:')} Don't nest too deep. If it's hard to read, use a for loop.""")

player.quiz(
    "List comprehension: squares of 1..5  →  [1, 4, 9, 16, 25]",
    check_eval([1, 4, 9, 16, 25]),
    hint="[x*x for x in range(1, 6)]",
)

player.quiz(
    "List comprehension: even numbers from 0 to 9  →  [0, 2, 4, 6, 8]",
    check_eval([0, 2, 4, 6, 8]),
    hint="[x for x in range(10) if x % 2 == 0]",
)

player.quiz(
    "Dict comprehension: map each char of \"abc\" to its index (a→0, b→1, c→2).  "
    "Expected: {'a': 0, 'b': 1, 'c': 2}",
    check_eval({"a": 0, "b": 1, "c": 2}),
    hint='{c: i for i, c in enumerate("abc")}',
)

player.explain("while, break, continue", f"""\
      while cond:            # loop while cond is truthy
          ...
          if done: break     # exit the loop early
          if skip: continue  # jump to the next iteration

  {cyan('Loop-else:')} Python has an `else` on loops. It runs only if the
  loop completed WITHOUT hitting `break`. Useful for search loops:

      for x in items:
          if match(x):
              result = x
              break
      else:
          result = None      # no match found

  It's unusual but handy once you get used to it.""")

player.quiz(
    "Sum of all numbers in range(1, 100) that are divisible by 7. "
    "(Type an expression using sum() and a comprehension.)",
    check_eval(sum(x for x in range(1, 100) if x % 7 == 0)),
    hint="sum(x for x in range(1, 100) if x % 7 == 0)",
)

player.play()
