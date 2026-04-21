"""Lesson 01: Values and Variables

Python is dynamically typed — variables don't declare types.
Answer by typing Python expressions. `?` for hint, `s` to skip.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from quiz_player import QuizPlayer, check_eval, check_predicate, cyan, green

player = QuizPlayer("Lesson 01: Values and Variables")

player.explain("Intro — values have types, variables don't", f"""\
  No type declarations. Assignment binds a name to a value:

      x = 42             # int
      name = "Alice"     # str
      pi = 3.14          # float
      done = True        # bool
      nothing = None     # NoneType

  The value carries the type. Use {cyan('type(x)')} to inspect it:

      type(42)      → <class 'int'>
      type("hi")    → <class 'str'>
      type(3.14)    → <class 'float'>""")

player.quiz(
    "Type an expression that returns the type of 3.14.",
    check_eval(float),
    hint="type(value)",
)

player.quiz(
    "Type an expression whose value is the type of the string \"hello\".",
    check_eval(str),
    hint="type(\"hello\")",
)

player.explain("Type conversion (casting)", f"""\
  Convert between types explicitly:

      int("42")      → 42
      int(3.7)       → 3        (truncates, does NOT round)
      float("3.14")  → 3.14
      str(42)        → "42"
      bool(0)        → False
      bool("")       → False
      bool("False")  → True     ← non-empty string is truthy!

  {cyan('Rule of thumb:')} empty things are falsy (0, 0.0, "", [], {{}}, None).
  Everything else is truthy.""")

player.quiz(
    "Convert the string \"100\" to an integer.",
    check_eval(100),
    hint="int(...)",
)

player.quiz(
    "What is bool(0.0) ?",
    check_eval(False),
)

player.quiz(
    'What is bool("0") ?  (note: a non-empty string, even "0")',
    check_eval(True),
    hint='"0" is a 1-character string, not the number 0',
)

player.explain("isinstance — the preferred type check", f"""\
  {cyan('isinstance(value, Type)')} returns True if value is that type
  (or a subclass). Prefer it over type(x) == X when writing checks.

      isinstance(42, int)        → True
      isinstance(3.14, (int, float))  → True   (tuple = any of these)
      isinstance("hi", int)      → False

  {cyan('Surprise:')} bool is a subclass of int in Python.
      isinstance(True, int)      → True
      True + True                → 2""")

player.quiz(
    "Is the value True an instance of int?  (type the expression)",
    check_eval(True),
    hint="isinstance(True, int)",
)

player.quiz(
    "Type an expression that is True iff 3.14 is either an int or a float.",
    check_eval(True),
    hint="isinstance(x, (int, float))",
)

player.play()
