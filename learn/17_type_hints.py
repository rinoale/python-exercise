"""Lesson 17: Type Hints"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from quiz_player import QuizPlayer, check_eval, check_code, cyan

player = QuizPlayer("Lesson 17: Type Hints")

player.explain("The basics — annotations are just metadata", f"""\
  Python doesn't enforce type hints at runtime. They're documentation
  and fuel for type checkers (mypy, pyright).

      def greet(name: str, times: int = 1) -> str:
          return (f"hi, {{name}} " * times).strip()

      count: int = 0
      names: list[str] = []

  {cyan("Built-in generics (Python 3.9+):")}
      list[int]           dict[str, int]
      tuple[int, ...]     set[str]
      tuple[int, str]     (fixed-length tuple with types)

  Don't import List, Dict from typing anymore — use the built-in lowercase.""")

player.quiz(
    "Write  add(a: int, b: int) -> int  that returns a + b.\n"
    "  The annotations must be literally present in  __annotations__.",
    check_code(lambda ns: (
        ns["add"](2, 3) == 5 and
        ns["add"].__annotations__.get("a") is int and
        ns["add"].__annotations__.get("b") is int and
        ns["add"].__annotations__.get("return") is int,
        "annotations present and function works",
    )),
    multiline=True,
)

player.explain("Optional, Union, and the pipe operator", f"""\
  {cyan("Optional[X]  means  X | None:")}
      def get_user(id: int) -> User | None:       # 3.10+ form
          ...
      def get_user(id: int) -> Optional[User]:    # old form, still valid

  {cyan("Union — any of several types:")}
      def parse(s: str) -> int | float | None:
          ...

  {cyan("Tip:")} If a parameter defaults to None, annotate with  | None:
      def greet(name: str | None = None) -> str:
          return f"hi, {{name or 'stranger'}}!"

  Old code uses  from typing import Optional, Union. New code uses  |.""")

player.explain("Callable, Iterable, Sequence — generic protocols", f"""\
      from collections.abc import Callable, Iterable, Sequence, Mapping

      def apply(fn: Callable[[int], int], xs: Iterable[int]) -> list[int]:
          return [fn(x) for x in xs]

      def first(items: Sequence[int]) -> int:
          return items[0]                   # needs indexing, not just iteration

  {cyan("Callable[[ArgTypes], ReturnType]:")}
      Callable[[int, str], bool]       # (int, str) -> bool
      Callable[..., str]               # any args -> str

  {cyan("Rule of thumb:")}
      Iterable   — can loop over (accepts generators, lists, sets)
      Sequence   — can index and len() (list, tuple; not set, not generator)
      Mapping    — dict-like (keys, values, items, [] lookup)""")

player.explain("TypeVar and generics — write polymorphic functions", f"""\
      from typing import TypeVar
      T = TypeVar("T")

      def first(xs: list[T]) -> T:
          return xs[0]

      first([1, 2, 3])       # mypy knows return type is int
      first(["a", "b"])      # mypy knows return type is str

  {cyan("3.12+ new generic syntax (no TypeVar needed):")}
      def first[T](xs: list[T]) -> T:
          return xs[0]

      class Box[T]:
          def __init__(self, value: T) -> None:
              self.value = value""")

player.explain("Protocol — structural (duck) typing", f"""\
      from typing import Protocol

      class SupportsLen(Protocol):
          def __len__(self) -> int: ...

      def two_or_more(obj: SupportsLen) -> bool:
          return len(obj) >= 2

  Any class with a  __len__ method satisfies SupportsLen — no need to
  inherit. {cyan("This is duck typing, with type checking.")}

  Pair with  @runtime_checkable  to use  isinstance  too:
      from typing import runtime_checkable
      @runtime_checkable
      class Closable(Protocol):
          def close(self) -> None: ...

      isinstance(fileobj, Closable)     # works at runtime""")

player.explain("TypedDict, Literal, Final, Self", f"""\
  {cyan("TypedDict — dict with known keys/types:")}
      from typing import TypedDict
      class User(TypedDict):
          name: str
          age: int
      u: User = {{"name": "Alice", "age": 30}}

  {cyan("Literal — exactly one of these values:")}
      from typing import Literal
      def move(dir: Literal["N", "S", "E", "W"]) -> None: ...

  {cyan("Final — don't rebind this name:")}
      from typing import Final
      MAX: Final = 100

  {cyan("Self (3.11+) — the enclosing class, for fluent APIs:")}
      from typing import Self
      class Builder:
          def add(self, x: int) -> Self: ...""")

player.quiz(
    "Define  sum_ints(xs: list[int]) -> int  (correct annotations; real summation).\n"
    "  Must have  list[int]  return type  int.",
    check_code(lambda ns: (
        ns["sum_ints"]([1, 2, 3]) == 6 and
        ns["sum_ints"].__annotations__.get("return") is int,
        "annotated and correct",
    )),
    multiline=True,
)

player.explain("Running a type checker", f"""\
      pip install mypy
      mypy learn/17_type_hints.py

      pip install pyright        # or pyright — often faster, strict by default

  {cyan("Gradual typing:")} annotate only what matters most. Legacy code
  can stay unannotated; new code should be typed. CI usually runs one
  of these tools as a lint step.""")

player.play()
