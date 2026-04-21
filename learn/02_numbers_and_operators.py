"""Lesson 02: Numbers and Operators"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from quiz_player import QuizPlayer, check_eval, cyan

player = QuizPlayer("Lesson 02: Numbers and Operators")

player.explain("Arithmetic operators", f"""\
  +   addition            3 + 2    → 5
  -   subtraction         3 - 2    → 1
  *   multiplication      3 * 2    → 6
  /   true division       7 / 2    → 3.5    (always returns float)
  //  floor division      7 // 2   → 3      (integer part, rounds down)
  %   modulo (remainder)  7 % 3    → 1
  **  power               2 ** 10  → 1024

  {cyan('Note:')} / always returns float, even for whole results:
      4 / 2  → 2.0   (not 2)""")

player.quiz(
    "What is 7 // 2 ?",
    check_eval(3),
    hint="// is floor division",
)

player.quiz(
    "What is 10 % 3 ?",
    check_eval(1),
    hint="% gives the remainder",
)

player.quiz(
    "What is 2 ** 10 ? (two to the tenth)",
    check_eval(1024),
)

player.quiz(
    "What is 10 / 4 ? (think about the type!)",
    check_eval(2.5),
    hint="/ always returns a float",
)

player.explain("Comparison & boolean", f"""\
  Comparisons return True or False:
      5 > 3       → True
      5 == 5.0    → True    (value compared, not type)
      5 is 5.0    → False   (`is` compares identity, not equality)

  Boolean operators:
      and    True if both are truthy
      or     True if either is truthy
      not    flips truthy ↔ falsy

  {cyan('Short-circuit:')} `and`/`or` return an operand, not a bool.
      0 or "hi"      → "hi"
      42 and "done"  → "done"
      [] or None     → None""")

player.quiz(
    "Is 5 equal to 5.0 ?  (type the expression with ==)",
    check_eval(True),
    hint="5 == 5.0",
)

player.quiz(
    "What does  True and not False  evaluate to?",
    check_eval(True),
)

player.quiz(
    "What does  0 or \"hello\"  evaluate to?",
    check_eval("hello"),
    hint="or returns the first truthy operand",
)

player.explain("Truthy and falsy — the full list", f"""\
  {cyan('Falsy values:')}   0, 0.0, "", [], {{}}, set(), None, False
  {cyan('Everything else is truthy.')}

  This lets you write concise checks:
      if not items: ...           # empty list/dict/string
      value = user_input or "default"

  But beware: 0 is falsy. If 0 is a valid value, use `is None`:
      if count is None: ...       # explicit None check""")

player.quiz(
    "What is bool([]) ?",
    check_eval(False),
)

player.quiz(
    "What is bool([0]) ?  (a list containing the number 0)",
    check_eval(True),
    hint="The list is non-empty, so it's truthy.",
)

player.play()
