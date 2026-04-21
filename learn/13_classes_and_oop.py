"""Lesson 13: Classes and Object-Oriented Programming"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from quiz_player import QuizPlayer, check_eval, check_code, cyan

player = QuizPlayer("Lesson 13: Classes and OOP")

player.explain("The basics — class, __init__, self", f"""\
      class Point:
          def __init__(self, x, y):       # constructor
              self.x = x                   # instance attribute
              self.y = y

          def distance(self, other):       # method — `self` is this instance
              return ((self.x - other.x)**2 + (self.y - other.y)**2) ** 0.5

      p = Point(3, 4)
      p.distance(Point(0, 0))     → 5.0

  {cyan("`self` is explicit in Python.")} Every method's first parameter
  is the instance. You don't type it when calling —  p.distance(q)  is
  secretly  Point.distance(p, q).""")

player.quiz(
    "Define a class  Point  with an  __init__(x, y)  and a method\n"
    "  distance_to_origin()  that returns  sqrt(x**2 + y**2).\n"
    "  Point(3, 4).distance_to_origin() should be 5.0.",
    check_code(lambda ns: (
        abs(ns["Point"](3, 4).distance_to_origin() - 5.0) < 1e-9 and
        abs(ns["Point"](0, 0).distance_to_origin()) < 1e-9,
        "Point works",
    )),
    hint="((self.x**2 + self.y**2) ** 0.5)",
    multiline=True,
)

player.explain("Class attributes vs instance attributes", f"""\
      class Dog:
          species = "Canis familiaris"       # class attribute — shared

          def __init__(self, name):
              self.name = name                # instance attribute — per-dog

      d1 = Dog("Rex")
      d2 = Dog("Max")
      d1.species is d2.species      → True    (shared)
      d1.name   == d2.name          → False   (separate)

  {cyan("Danger:")} class attributes that are mutable (like lists) are shared!
      class Box:
          items = []                  # shared across all Box instances
          def add(self, x): self.items.append(x)
      # Use  self.items = []  in __init__ instead.""")

player.explain("classmethod and staticmethod", f"""\
      class Temperature:
          def __init__(self, celsius):
              self.celsius = celsius

          @classmethod
          def from_fahrenheit(cls, f):
              return cls((f - 32) * 5 / 9)   # cls = the class itself

          @staticmethod
          def is_freezing(celsius):
              return celsius <= 0             # no self, no cls

      t = Temperature.from_fahrenheit(212)    # alternative constructor
      Temperature.is_freezing(-5)              → True

  {cyan("Rule of thumb:")}
      @classmethod   when you need the class (alternative constructors, subclass-aware)
      @staticmethod  when it's really just a utility function
      (instance method — default)  when you need `self`""")

player.quiz(
    "Define a class  Temperature  with  __init__(celsius)  and a\n"
    "  @classmethod  from_fahrenheit(f)  as alternative constructor.\n"
    "  Temperature.from_fahrenheit(32).celsius should be 0.0 (close enough).",
    check_code(lambda ns: (
        abs(ns["Temperature"].from_fahrenheit(32).celsius) < 1e-9 and
        abs(ns["Temperature"].from_fahrenheit(212).celsius - 100.0) < 1e-9,
        "alternative constructor works",
    )),
    hint="@classmethod\\ndef from_fahrenheit(cls, f): return cls((f - 32) * 5 / 9)",
    multiline=True,
)

player.explain("@property — computed attributes", f"""\
      class Circle:
          def __init__(self, radius):
              self.radius = radius

          @property
          def area(self):
              return 3.14159 * self.radius ** 2    # called as  c.area, no ()

          @area.setter          # optional — make it writable
          def area(self, value):
              self.radius = (value / 3.14159) ** 0.5

      c = Circle(10)
      c.area          → 314.159       # looks like an attribute
      c.area = 100    # calls the setter

  {cyan("Use @property when the API should LOOK like an attribute")} but
  you want logic behind it (validation, derived values, lazy compute).""")

player.explain("Inheritance and super()", f"""\
      class Animal:
          def __init__(self, name):
              self.name = name
          def speak(self):
              return "..."

      class Dog(Animal):
          def speak(self):           # override
              return "woof"

      class Puppy(Dog):
          def __init__(self, name, age):
              super().__init__(name)    # call parent __init__
              self.age = age

      Puppy("Rex", 1).speak()    → "woof"   (inherited from Dog)

  {cyan("super() returns a proxy to the next class in the MRO")}
  (method resolution order). For single inheritance, that's the parent.

  {cyan("Prefer composition over inheritance.")} Deep class trees get tangled.""")

player.quiz(
    "Define an  Animal  class with  __init__(name)  and method  speak() → '...'.\n"
    "  Define  Cat(Animal)  that overrides  speak()  to return 'meow'.\n"
    "  Cat('Whiskers').speak() == 'meow'; Cat('X').name == 'X'.",
    check_code(lambda ns: (
        ns["Cat"]("Whiskers").speak() == "meow" and
        ns["Cat"]("X").name == "X",
        "Cat inherits correctly",
    )),
    multiline=True,
)

player.play()
