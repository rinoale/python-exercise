"""Lesson 21: Iterators and Lazy Evaluation"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from quiz_player import QuizPlayer, check_eval, check_code, cyan

player = QuizPlayer("Lesson 21: Iterators and Lazy Evaluation")

player.explain("Iterable vs Iterator", f"""\
  {cyan("Iterable")} — has  __iter__  that returns an iterator.
      list, tuple, str, dict, set, file, generator — all iterable.

  {cyan("Iterator")} — has  __next__; calling it yields the next item,
  or raises StopIteration when done. Also is itself iterable
  (its  __iter__  returns self).

      lst = [1, 2, 3]                    # iterable, not iterator
      it  = iter(lst)                    # iterator
      next(it)                           → 1
      next(it)                           → 2

  {cyan("for x in obj:")} is sugar for:
      it = iter(obj)
      while True:
          try: x = next(it)
          except StopIteration: break
          ...""")

player.quiz(
    "Use  iter()  and  next()  on [10, 20, 30] to get the second value.\n"
    "  Single expression: iterate, advance once, then return next.\n"
    "  Expected: 20",
    check_eval(20),
    hint='(lambda it: (next(it), next(it))[1])(iter([10, 20, 30]))',
)

player.explain("Build your own iterator class", f"""\
      class Countdown:
          def __init__(self, n): self.n = n
          def __iter__(self):   return self            # iterator = self
          def __next__(self):
              if self.n <= 0: raise StopIteration
              self.n -= 1
              return self.n + 1

      list(Countdown(3))           → [3, 2, 1]

  {cyan("Separate iterable and iterator when you want multiple passes:")}
      class Range:
          def __init__(self, n): self.n = n
          def __iter__(self):
              return RangeIter(self.n)       # fresh iterator each time

      for x in r: ...       # iterates from start every time""")

player.quiz(
    "Build  Countdown(n)  as an iterator class.  list(Countdown(3)) → [3, 2, 1].\n"
    "  Must implement  __iter__  and  __next__  (no yield).",
    check_code(lambda ns: (
        list(ns["Countdown"](3)) == [3, 2, 1] and list(ns["Countdown"](1)) == [1],
        "iterator class works",
    )),
    hint="raise StopIteration when n hits 0; implement __iter__ and __next__",
    multiline=True,
)

player.explain("Infinite iterators", f"""\
  Because iteration is lazy, you can have infinite sequences safely:

      import itertools as it
      for i, x in zip(it.count(100, step=5), ["a", "b", "c"]):
          print(i, x)                    # 100 a, 105 b, 110 c

      def naturals():
          n = 1
          while True:
              yield n
              n += 1

      for n in naturals():
          if n > 5: break
          print(n)

  {cyan("Use  itertools.islice  to cap an infinite iterator:")}
      first_100_primes = list(itertools.islice(primes(), 100))""")

player.explain("itertools.tee — clone an iterator", f"""\
      import itertools as it
      a, b = it.tee(some_iter)           # two independent iterators
      list(a) == list(b)                 # each can be consumed separately

  {cyan("Internally:")} tee caches values the slower branch hasn't seen
  yet. Don't call tee on a huge iterator and then consume one branch
  entirely before the other — memory blows up.""")

player.quiz(
    "Use  itertools.islice  to take the first 5 squares from an infinite generator.\n"
    "  Start with:  def squares():\\n      n = 1\\n      while True:\\n          yield n*n; n += 1\n"
    "  Expected result:  [1, 4, 9, 16, 25]",
    check_code(lambda ns: (
        list(__import__("itertools").islice(ns["squares"](), 5)) == [1, 4, 9, 16, 25],
        "islice slices the infinite generator",
    )),
    hint="def squares(): ... with yield",
    multiline=True,
)

player.explain("Why lazy evaluation matters", f"""\
  {cyan("Memory:")} a generator of 100M items uses O(1) memory; a list uses O(n).

      # BAD for huge data:
      total = sum([line_length(line) for line in open(path)])
      # GOOD — no intermediate list:
      total = sum(line_length(line) for line in open(path))

  {cyan("Short-circuit:")} any/all stop as soon as they know the answer.

      any(is_prime(n) for n in big_range)    # stops at the first prime

  {cyan("Composition:")} pipelines read top-down without temp lists.

      records = (parse(line) for line in open(path))
      recent  = (r for r in records if r.ts > cutoff)
      for r in recent: ...

  The iterator protocol is Python's killer feature for data pipelines.""")

player.play()
