"""Lesson 19: Abstract Base Classes and Protocols"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from quiz_player import QuizPlayer, check_eval, check_code, cyan

player = QuizPlayer("Lesson 19: ABCs and Protocols")

player.explain("Why abstract classes?", f"""\
  A class that says {cyan("my subclasses MUST implement these methods")}.
  Forgetting one = TypeError at instantiation (not later, at call time).

      from abc import ABC, abstractmethod

      class Shape(ABC):
          @abstractmethod
          def area(self) -> float: ...

          @abstractmethod
          def perimeter(self) -> float: ...

          def describe(self) -> str:         # concrete method, inherited
              return f"area={{self.area()}}, perim={{self.perimeter()}}"

      Shape()                    # TypeError — can't instantiate ABC

      class Circle(Shape):
          def __init__(self, r): self.r = r
          def area(self):        return 3.14159 * self.r ** 2
          def perimeter(self):   return 2 * 3.14159 * self.r

      Circle(5)                  # fine — all abstract methods implemented""")

player.quiz(
    "Define abstract  Shape(ABC)  with abstract method  area().\n"
    "  Define  Square(Shape)  with  __init__(side)  and  area() -> side*side.\n"
    "  Verify  Shape() raises TypeError; Square(5).area() == 25.",
    check_code(lambda ns: _test_shape(ns)),
    hint="from abc import ABC, abstractmethod",
    multiline=True,
)


def _test_shape(ns):
    Shape = ns["Shape"]
    Square = ns["Square"]
    try:
        Shape()
    except TypeError:
        pass
    else:
        return False, "Shape should not be instantiable"
    if Square(5).area() != 25:
        return False, f"Square(5).area() = {Square(5).area()}"
    return True, "ABC enforced"


player.explain("Protocols — duck typing with type hints", f"""\
  Unlike ABCs, {cyan("Protocols don't require inheritance")}. Any class
  with the right methods satisfies the Protocol automatically.

      from typing import Protocol

      class SupportsArea(Protocol):
          def area(self) -> float: ...

      def total_area(shapes: list[SupportsArea]) -> float:
          return sum(s.area() for s in shapes)

      class Square:                      # NO inheritance needed!
          def __init__(self, s): self.s = s
          def area(self): return self.s * self.s

      total_area([Square(3), Square(4)])    # type-checks, works

  {cyan("Runtime checking with @runtime_checkable:")}
      from typing import runtime_checkable
      @runtime_checkable
      class HasName(Protocol):
          name: str

      isinstance(obj, HasName)           # works at runtime""")

player.quiz(
    "Define  @runtime_checkable Protocol  Quacks  with method  quack().\n"
    "  Define  Duck  with  quack()  (no inheritance from Quacks).\n"
    "  Verify  isinstance(Duck(), Quacks)  is True.",
    check_code(lambda ns: _test_protocol(ns)),
    hint="from typing import Protocol, runtime_checkable",
    multiline=True,
)


def _test_protocol(ns):
    Quacks = ns["Quacks"]
    Duck = ns["Duck"]
    if not isinstance(Duck(), Quacks):
        return False, "Duck is not recognized as Quacks — did you @runtime_checkable the protocol?"
    return True, "Duck is a Quacks (structurally)"


player.explain("ABC vs Protocol — when to pick which", f"""\
  {cyan("ABC")}
    - You control ALL the implementations.
    - You want to share concrete code in the base class.
    - You want an inheritance hierarchy (Animal → Dog → Poodle).

  {cyan("Protocol")}
    - You're accepting objects you don't own (3rd-party, user-supplied).
    - No inheritance needed — any structural match works.
    - Playing nice with duck typing that already exists in the codebase.

  In practice, Protocols fit modern Python better. Prefer them unless
  you specifically need a shared base class with concrete code.""")

player.explain("Virtual subclasses and register()", f"""\
  An ABC can {cyan("claim")} a class as its subclass without that class
  inheriting from it:

      class MyContainer(ABC):
          @abstractmethod
          def __contains__(self, x): ...

      MyContainer.register(list)         # list is now a virtual subclass
      issubclass(list, MyContainer)      → True
      isinstance([1,2,3], MyContainer)   → True

  {cyan("Used by stdlib:")} collections.abc.Iterable, Container, Sequence
  already register built-in types this way — which is how
  isinstance([], Iterable) works even though list doesn't inherit from Iterable.""")

player.play()
