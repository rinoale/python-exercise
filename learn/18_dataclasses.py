"""Lesson 18: Dataclasses"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from quiz_player import QuizPlayer, check_eval, check_code, cyan

player = QuizPlayer("Lesson 18: Dataclasses")

player.explain("@dataclass — auto-generate __init__, __repr__, __eq__", f"""\
      from dataclasses import dataclass

      @dataclass
      class Point:
          x: int
          y: int

  Expands to roughly:
      class Point:
          def __init__(self, x: int, y: int):
              self.x = x
              self.y = y
          def __repr__(self):
              return f"Point(x={{self.x}}, y={{self.y}})"
          def __eq__(self, other):
              if not isinstance(other, Point): return NotImplemented
              return (self.x, self.y) == (other.x, other.y)

  {cyan("This replaces 80% of the cases you'd use namedtuple or plain classes.")}""")

player.quiz(
    "Use  @dataclass  to define  Point(x: int, y: int).\n"
    "  Verify:  Point(3, 4) == Point(3, 4)   and   repr(Point(3, 4)) starts with 'Point('.",
    check_code(lambda ns: (
        ns["Point"](3, 4) == ns["Point"](3, 4) and
        ns["Point"](3, 4) != ns["Point"](1, 2) and
        repr(ns["Point"](3, 4)).startswith("Point("),
        "@dataclass works",
    )),
    hint="from dataclasses import dataclass",
    multiline=True,
)

player.explain("field() — defaults, factories, metadata", f"""\
      from dataclasses import dataclass, field

      @dataclass
      class Config:
          name: str
          tags: list[str] = field(default_factory=list)    # per-instance []
          timeout: float = 30.0
          secret: str = field(default="", repr=False)       # hide from repr

  {cyan("Why default_factory?")} A mutable default like  tags: list[str] = []
  would be {cyan("shared")} across every instance (same gotcha as in lesson 08).
  field(default_factory=list) constructs a fresh list per instance.

  {cyan("Other field options:")}
      default=...           simple default
      init=False            don't include in __init__
      repr=False            don't show in __repr__
      compare=False         don't participate in __eq__
      hash=False            excluded from __hash__""")

player.quiz(
    "Define  @dataclass  Cart  with:\n"
    "    owner: str\n"
    "    items: list[str]  (default empty list, per-instance)\n"
    "  Then verify  Cart('a').items  and  Cart('b').items  are different objects\n"
    "  so that appending to one doesn't affect the other.",
    check_code(lambda ns: _test_cart(ns)),
    hint="items: list[str] = field(default_factory=list)",
    multiline=True,
)


def _test_cart(ns):
    Cart = ns["Cart"]
    a = Cart("a")
    b = Cart("b")
    a.items.append("x")
    if b.items == ["x"]:
        return False, "items list is shared — use field(default_factory=list)"
    return True, "per-instance lists"


player.explain("frozen and slots", f"""\
  {cyan("frozen=True — immutable instances, hashable automatically:")}
      @dataclass(frozen=True)
      class Point:
          x: int
          y: int

      p = Point(3, 4)
      p.x = 5                # FrozenInstanceError
      {{p, Point(3, 4)}}      # usable in sets; hash is auto

  {cyan("slots=True (3.10+) — smaller memory, faster attribute access:")}
      @dataclass(slots=True)
      class Vec:
          x: float
          y: float

      v = Vec(1.0, 2.0)
      v.z = 3                # AttributeError — __slots__ prevents new attrs

  {cyan("Rule of thumb:")} use  frozen=True  for value objects
  (coordinates, currency, configs). Add  slots=True  when you'll
  instantiate millions.""")

player.quiz(
    "Define  @dataclass(frozen=True)  Coord(x: int, y: int).\n"
    "  Verify  it's hashable (use as set element) and raises on mutation.",
    check_code(lambda ns: _test_frozen(ns)),
    hint="@dataclass(frozen=True)",
    multiline=True,
)


def _test_frozen(ns):
    Coord = ns["Coord"]
    c = Coord(3, 4)
    s = {Coord(3, 4), Coord(3, 4), Coord(1, 2)}
    if len(s) != 2:
        return False, f"set contains {len(s)} items; expected 2"
    try:
        c.x = 99
    except Exception as e:
        if "frozen" in str(e).lower() or isinstance(e, AttributeError):
            return True, "frozen + hashable"
        return False, f"mutation raised unexpected: {e}"
    return False, "mutation should have raised"


player.explain("When to use NamedTuple instead", f"""\
      from typing import NamedTuple

      class Point(NamedTuple):
          x: int
          y: int

  {cyan("NamedTuple vs dataclass — both give you typed records.")}

      NamedTuple  — iterable, unpackable (x, y = point), immutable.
                     Compares positionally like a tuple.
      dataclass   — more flexible (frozen/slots/post-init/methods),
                     comparable as an object, more room to grow.

  {cyan("Rule of thumb:")} use dataclass by default. Pick NamedTuple only
  when you specifically want tuple-like behavior and unpacking.""")

player.play()
