"""Lesson 23: Pattern Matching (Python 3.10+)"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from quiz_player import QuizPlayer, check_eval, check_function, check_code, cyan

player = QuizPlayer("Lesson 23: Pattern Matching")

player.explain("match / case — structural pattern matching", f"""\
      def describe(p):
          match p:
              case 0:                    return "zero"
              case 1 | 2 | 3:            return "small"           # OR pattern
              case int() if p > 100:     return "big"             # guard
              case int():                return "medium"
              case str():                return "a string"
              case _:                    return "unknown"         # wildcard

  {cyan("This is NOT switch/case.")} It does structural destructuring and
  binding — closer to Haskell or Rust pattern matching.""")

player.quiz(
    'Define  classify(x)  using  match/case  returning:\n'
    '    "zero"    if x == 0\n'
    '    "small"   if x in 1..5\n'
    '    "big"     otherwise (assume int)\n'
    '  Tests:  classify(0), classify(3), classify(100)',
    check_function("classify", [
        ((0,), "zero"),
        ((3,), "small"),
        ((100,), "big"),
        ((5,), "small"),
        ((6,), "big"),
    ]),
    hint="case 0: ...; case n if 1 <= n <= 5: ...; case _: ...",
    multiline=True,
)

player.explain("Sequence patterns — destructure tuples and lists", f"""\
      match point:
          case (0, 0):            return "origin"
          case (x, 0):            return f"on x axis at {{x}}"
          case (0, y):            return f"on y axis at {{y}}"
          case (x, y):            return f"point ({{x}}, {{y}})"
          case [x, y, z]:         return f"3D point"
          case [first, *rest]:    return f"list starting with {{first}}"
          case []:                return "empty"

  {cyan("Binding:")} bare names  x, y  in a pattern BIND values. To match
  against an existing variable, use a dotted name or a literal:

      BLUE = "blue"
      match color:
          case BLUE:              # WRONG — this binds `BLUE` to whatever color is!
          case "blue":            # literal match
          case shapes.BLUE:       # dotted — looks up attribute, doesn't bind""")

player.quiz(
    "Define  shape(point)  using match:\n"
    '    (0, 0) → "origin"\n'
    '    (x, 0) → "on x axis"\n'
    '    (0, y) → "on y axis"\n'
    '    (x, y) → "elsewhere"',
    check_function("shape", [
        (((0, 0),), "origin"),
        (((5, 0),), "on x axis"),
        (((0, 3),), "on y axis"),
        (((2, 3),), "elsewhere"),
    ]),
    multiline=True,
)

player.explain("Class patterns and mapping patterns", f"""\
  {cyan("Class patterns")} destructure by attribute:

      from dataclasses import dataclass
      @dataclass
      class Point: x: int; y: int

      match p:
          case Point(x=0, y=0):           print("origin")
          case Point(x=0):                print("on y axis")
          case Point():                   print("somewhere")

  Positional works if the class declares __match_args__ (dataclass does
  so automatically in field order):

      case Point(0, 0): ...

  {cyan("Mapping patterns")} for dicts:
      match response:
          case {{"status": 200, "body": body}}:  return body
          case {{"status": code, **rest}}:        return f"error {{code}}"
          case _: return None

  {cyan("Keys match exactly, extra keys are allowed by default")} — use
  **rest to capture them, or omit to ignore.""")

player.quiz(
    'Define  handle(resp: dict)  that returns:\n'
    '    "ok <body>"   if resp is {"status": 200, "body": <body>}\n'
    '    "error <code>"  for any other status\n'
    '    "unknown"      if there is no "status" key',
    check_function("handle", [
        (({"status": 200, "body": "hi"},), "ok hi"),
        (({"status": 404},), "error 404"),
        (({"msg": "huh"},), "unknown"),
    ]),
    hint='case {"status": 200, "body": b}: return f"ok {b}"',
    multiline=True,
)

player.explain("Guards and wildcards", f"""\
  Add an  if  clause to a case:

      match user:
          case {{"role": role}} if role in ("admin", "root"):  return "elevated"
          case {{"role": _}}:                                   return "normal"
          case _:                                               return "no role"

  {cyan("The _ wildcard matches anything without binding.")} Use it for
  catch-all cases. Capture into a name only when you need the value.""")

player.play()
