"""Lesson 14: Dunder (Magic) Methods — Deep Dive"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from quiz_player import QuizPlayer, check_eval, check_code, cyan

player = QuizPlayer("Lesson 14: Dunder Methods")

player.explain("What are dunders?", f"""\
  {cyan("Dunder")} = Double UNDERscore. Python protocols are implemented
  by methods like  __init__, __repr__, __eq__, __len__.

  The interpreter calls them for you on specific syntax:
      x + y          → type(x).__add__(x, y)
      len(x)         → type(x).__len__(x)
      x[i]           → type(x).__getitem__(x, i)
      x == y         → type(x).__eq__(x, y)
      str(x)         → type(x).__str__(x)

  Implement dunders to make YOUR types behave like built-ins.""")

player.explain("__repr__ vs __str__", f"""\
      class Point:
          def __init__(self, x, y):
              self.x, self.y = x, y
          def __repr__(self):             # for developers — unambiguous
              return f"Point({{self.x}}, {{self.y}})"
          def __str__(self):              # for end users — readable
              return f"({{self.x}}, {{self.y}})"

      p = Point(3, 4)
      repr(p)         → "Point(3, 4)"        # also: the REPL shows this
      str(p)          → "(3, 4)"
      print(p)        → (3, 4)               # print() uses __str__

  {cyan("Rule:")} if you write only one, write  __repr__. Python falls
  back to __repr__ when __str__ is missing, but not the other way.""")

player.quiz(
    "Define  Point  with  __init__(x, y)  and  __repr__  returning\n"
    "  'Point(x=3, y=4)' format.  repr(Point(3, 4)) should be 'Point(x=3, y=4)'.",
    check_code(lambda ns: (
        repr(ns["Point"](3, 4)) == "Point(x=3, y=4)" and
        repr(ns["Point"](-1, 0)) == "Point(x=-1, y=0)",
        "repr works",
    )),
    hint='return f"Point(x={self.x}, y={self.y})"',
    multiline=True,
)

player.explain("__eq__ and __hash__ — equality and hashability", f"""\
  By default, instances are equal only if they're the SAME object.
  Two Point(3, 4) values are considered different. Fix with __eq__:

      class Point:
          def __init__(self, x, y): self.x, self.y = x, y
          def __eq__(self, other):
              if not isinstance(other, Point): return NotImplemented
              return (self.x, self.y) == (other.x, other.y)
          def __hash__(self):
              return hash((self.x, self.y))   # must match __eq__

  {cyan("Rule:")} objects that compare equal MUST have the same hash.
  If you override  __eq__, you must either:
      1. Also override  __hash__  consistently, OR
      2. Set  __hash__ = None  (makes the class unhashable)

  Without hash, you can't put instances in sets or use as dict keys.""")

player.quiz(
    "Define  Point  with  __init__(x, y),  __eq__,  and  __hash__  so that:\n"
    "    Point(3, 4) == Point(3, 4)\n"
    "    {Point(3, 4), Point(3, 4)}  has length 1",
    check_code(lambda ns: (
        ns["Point"](3, 4) == ns["Point"](3, 4) and
        ns["Point"](3, 4) != ns["Point"](3, 5) and
        len({ns["Point"](3, 4), ns["Point"](3, 4), ns["Point"](1, 2)}) == 2,
        "equality and hash aligned",
    )),
    hint="hash((self.x, self.y))",
    multiline=True,
)

player.explain("Container protocols — __len__, __getitem__, __iter__", f"""\
  Implementing these makes your class act like a sequence/collection:

      class Deck:
          def __init__(self, cards): self.cards = list(cards)
          def __len__(self):         return len(self.cards)
          def __getitem__(self, i):  return self.cards[i]
          # __iter__ is auto-implied when __getitem__ takes integers

      d = Deck([1, 2, 3, 4])
      len(d)              → 4
      d[0]                → 1
      for c in d: ...     # works! because __getitem__ is indexed from 0
      2 in d              → True     (falls back to iteration)

  {cyan("For true custom iteration, implement __iter__ directly:")}
      def __iter__(self):
          return iter(self.cards)""")

player.quiz(
    "Define  Bag  wrapping a list. It should support:\n"
    "    len(bag)   via __len__\n"
    "    bag[i]    via __getitem__\n"
    "    for x in bag: ...   (works automatically via __getitem__)\n"
    "  Constructor:  Bag([1, 2, 3])",
    check_code(lambda ns: (
        (lambda b: len(b) == 3 and b[0] == 1 and b[2] == 3 and list(b) == [1, 2, 3]
        )(ns["Bag"]([1, 2, 3])),
        "container protocol works",
    )),
    multiline=True,
)

player.explain("__call__ — make instances callable", f"""\
      class Multiplier:
          def __init__(self, factor): self.factor = factor
          def __call__(self, x): return x * self.factor

      double = Multiplier(2)
      double(5)        → 10          # called like a function!
      callable(double)  → True

  {cyan("Why?")} Useful for configurable functions (strategy pattern),
  function-like objects that carry state, or writing decorators as classes.""")

player.explain("Arithmetic dunders", f"""\
      __add__, __sub__, __mul__, __truediv__, __mod__, __pow__
      __radd__ (reverse, for when the left side doesn't know how)
      __iadd__ (in-place: x += y)
      __neg__, __abs__
      __lt__, __le__, __gt__, __ge__  (use @functools.total_ordering)

  Implementing __add__ lets you write  a + b  for your type:
      class Vec:
          def __init__(self, x, y): self.x, self.y = x, y
          def __add__(self, other): return Vec(self.x+other.x, self.y+other.y)

      Vec(1, 2) + Vec(3, 4)     → Vec(4, 6)""")

player.play()
