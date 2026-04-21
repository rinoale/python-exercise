"""Lesson 09: Comprehensions and Generators"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from quiz_player import QuizPlayer, check_eval, check_function, check_code, cyan

player = QuizPlayer("Lesson 09: Comprehensions and Generators")

player.explain("Comprehension recap — all four forms", f"""\
  list:  [x*x for x in range(5)]           → [0, 1, 4, 9, 16]
  dict:  {{x: x*x for x in range(4)}}        → {{0:0, 1:1, 2:4, 3:9}}
  set:   {{x % 3 for x in range(10)}}        → {{0, 1, 2}}
  gen:   (x*x for x in range(5))           → <generator> (lazy!)

  {cyan("Generator expression vs list comprehension:")}
      [x*x for x in range(10**8)]   # builds 400 MB list
      (x*x for x in range(10**8))   # yields one at a time — almost no memory

  Generators are great when you pipe through another consumer:
      sum(x*x for x in range(1000))      # no intermediate list
      any(is_prime(n) for n in big)      # stops at first True""")

player.quiz(
    "Sum of squares of 1..10 using a generator expression (no list literal).",
    check_eval(sum(x*x for x in range(1, 11))),
    hint="sum(x*x for x in range(1, 11))",
)

player.quiz(
    "Are any numbers in range(100) divisible by 97?  Use any() with a gen expr.",
    check_eval(True),
    hint="any(x % 97 == 0 for x in range(100))",
)

player.explain("yield — turn a function into a generator", f"""\
  Any `def` that contains {cyan("yield")} becomes a generator function.
  Calling it returns a generator (no code runs yet). Iteration runs the
  body, pausing at each yield:

      def squares(n):
          for i in range(n):
              yield i * i

      list(squares(4))         → [0, 1, 4, 9]
      for x in squares(4): ...  # lazy: one value at a time

  {cyan("Generators are iterators.")} They have .__next__() and raise
  StopIteration when exhausted. You can only iterate once.""")

player.quiz(
    "Define  evens(n)  as a generator yielding even numbers 0, 2, 4, ... up to (but not including) n.\n"
    "  The function must use  yield  (not return a list).",
    check_code(lambda ns: (
        list(ns["evens"](8)) == [0, 2, 4, 6] and list(ns["evens"](1)) == [0],
        "yields even numbers correctly",
    )),
    hint="def evens(n):\\n    for i in range(0, n, 2):\\n        yield i",
    multiline=True,
)

player.explain("Consuming generators — the key idea", f"""\
  Many built-ins accept any iterable and consume it lazily:

      sum(gen)          list(gen)       tuple(gen)
      max(gen)          set(gen)        dict(gen)
      any(gen)          all(gen)        next(gen)
      "".join(gen)      sorted(gen)     enumerate(gen)

  {cyan("Chaining generators is cheap:")}
      def lines(path):
          with open(path) as f:
              for line in f: yield line
      def stripped(lines): return (line.rstrip() for line in lines)
      def non_empty(lines): return (l for l in lines if l)

  You can pipe millions of rows through this without loading them all.""")

player.explain("yield from — delegate to another iterable", f"""\
      def flatten(lists):
          for sub in lists:
              yield from sub              # yield each item of sub

      list(flatten([[1, 2], [3, 4], [5]]))   → [1, 2, 3, 4, 5]

  Equivalent to  for x in sub: yield x  but faster and cleaner.""")

player.quiz(
    "Define  flatten(lists)  as a generator that yields every element\n"
    "  from every sub-list, in order.  Use  yield from.\n"
    "  flatten([[1,2],[3,4]]) consumed as list → [1,2,3,4].",
    check_code(lambda ns: (
        list(ns["flatten"]([[1, 2], [3, 4], [5]])) == [1, 2, 3, 4, 5],
        "flatten works",
    )),
    hint="def flatten(lists):\\n    for sub in lists:\\n        yield from sub",
    multiline=True,
)

player.play()
