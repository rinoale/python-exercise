"""Lesson 20: Advanced OOP Internals — descriptors, __slots__, metaclasses"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from quiz_player import QuizPlayer, check_eval, check_code, cyan

player = QuizPlayer("Lesson 20: Advanced OOP Internals")

player.explain("Descriptors — the magic behind @property", f"""\
  A {cyan("descriptor")} is any object that defines  __get__,  __set__,
  and/or  __delete__, and is stored as a CLASS attribute.

      class Positive:
          def __set_name__(self, owner, name):
              self.name = "_" + name           # e.g. "_price"
          def __get__(self, obj, objtype=None):
              if obj is None: return self
              return getattr(obj, self.name, 0)
          def __set__(self, obj, value):
              if value < 0:
                  raise ValueError(f"must be >= 0, got {{value}}")
              setattr(obj, self.name, value)

      class Product:
          price = Positive()              # descriptor instance on the CLASS

      p = Product()
      p.price = 10                        # calls Positive.__set__
      p.price = -1                        # ValueError
      p.price                             # calls Positive.__get__

  {cyan("@property is just a descriptor.")} Writing your own is the escape
  hatch when you want the same validation reused across many classes.""")

player.quiz(
    "Write a descriptor  Positive  that rejects negative assignments.\n"
    "  Use it in  class Product  for  price.\n"
    "  Verify  Product().price = 10 works, but  = -1  raises ValueError.",
    check_code(lambda ns: _test_descriptor(ns)),
    hint="implement __set_name__, __get__, __set__",
    multiline=True,
)


def _test_descriptor(ns):
    Product = ns["Product"]
    p = Product()
    p.price = 10
    if p.price != 10:
        return False, f"p.price = {p.price}, expected 10"
    try:
        p.price = -1
    except ValueError:
        return True, "descriptor validates"
    return False, "should have raised ValueError on negative"


player.explain("__slots__ — smaller, faster instances", f"""\
  Normally every instance has a  __dict__  for dynamic attributes.
  __slots__ replaces it with a fixed set of attributes:

      class Vec:
          __slots__ = ("x", "y")         # declared attributes only
          def __init__(self, x, y):
              self.x, self.y = x, y

      v = Vec(1, 2)
      v.z = 3                            # AttributeError — no __dict__

  {cyan("Pros:")}
      ~40-50% less memory per instance
      Faster attribute access
      Helps catch typos (obj.naem = 1 would normally silently succeed)

  {cyan("Cons:")}
      No dynamic attributes
      Doesn't mix well with multiple inheritance
      Not needed for most classes — use when you'll have millions

  Dataclasses can request slots via  @dataclass(slots=True).""")

player.quiz(
    "Define  class Frozen with __slots__ = ('a', 'b')  and  __init__(a, b).\n"
    "  Verify  assigning obj.c = 1 raises AttributeError.",
    check_code(lambda ns: _test_slots(ns)),
    multiline=True,
)


def _test_slots(ns):
    Frozen = ns["Frozen"]
    f = Frozen(1, 2)
    if f.a != 1 or f.b != 2:
        return False, "init didn't set attributes"
    try:
        f.c = 99
    except AttributeError:
        return True, "__slots__ prevents extra attrs"
    return False, "extra attribute should have been rejected"


player.explain("__init_subclass__ — a lighter alternative to metaclasses", f"""\
  Runs once for every class that inherits from yours. Useful for
  registering plugins or validating subclass structure.

      class Plugin:
          registry = {{}}
          def __init_subclass__(cls, **kwargs):
              super().__init_subclass__(**kwargs)
              Plugin.registry[cls.__name__] = cls

      class Slack(Plugin): ...
      class Email(Plugin): ...

      Plugin.registry         → {{"Slack": Slack, "Email": Email}}

  {cyan("Benefits over metaclasses:")} much simpler, only one hook, works
  through normal inheritance.""")

player.explain("Metaclasses — mostly, you don't need them", f"""\
  A metaclass creates classes the same way a class creates instances.
  The default metaclass is  type.

      class MyMeta(type):
          def __new__(mcs, name, bases, ns):
              ns["created_at"] = time.time()
              return super().__new__(mcs, name, bases, ns)

      class Logged(metaclass=MyMeta): ...

  {cyan("When you might reach for a metaclass:")}
      - Writing an ORM (Django models) or serialization library
      - Enforcing rules on class creation itself
      - Rewriting class bodies at definition time

  {cyan("In normal app code:")} class decorators,  __init_subclass__, and
  descriptors cover 99% of needs. Metaclasses are an escape hatch for
  framework authors — and they compose poorly with each other.""")

player.play()
