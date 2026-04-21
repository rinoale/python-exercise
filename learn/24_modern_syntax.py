"""Lesson 24: Modern Python Syntax (3.8+)"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from quiz_player import QuizPlayer, check_eval, check_code, cyan

player = QuizPlayer("Lesson 24: Modern Python Syntax")

player.explain("Walrus operator  :=  (3.8+)", f"""\
  Assign inside an expression. Useful when you need the value twice:

      # BEFORE:
      data = read_chunk()
      while data:
          process(data)
          data = read_chunk()

      # AFTER:
      while (data := read_chunk()):
          process(data)

  Another common case — avoiding redundant work in comprehensions:

      # BEFORE:
      results = [f(x) for x in xs if f(x) > 0]   # computes f(x) twice!
      # AFTER:
      results = [y for x in xs if (y := f(x)) > 0]

  {cyan("Parentheses required around  (name := value)  in most contexts.")}""")

player.quiz(
    "Use the walrus operator to filter [1, 2, 3, 4, 5] to keep only those\n"
    "  items whose square is > 10, returning the SQUARES (not the originals).\n"
    "  Expected: [16, 25]",
    check_eval([16, 25]),
    hint="[y for x in [1,2,3,4,5] if (y := x*x) > 10]",
)

player.explain("Dict merge and update  |  and  |=  (3.9+)", f"""\
      a = {{"x": 1}}
      b = {{"y": 2}}
      a | b                         → {{"x": 1, "y": 2}}      (new dict)
      a |= b                        # in-place merge into a

      {{**a, **b}}                   # old spread-based approach (still works)

  {cyan("Duplicate keys:")} the right-hand side wins.
      {{"a": 1}} | {{"a": 2}}         → {{"a": 2}}""")

player.quiz(
    'What does  {"a": 1, "b": 2} | {"b": 20, "c": 3}  evaluate to?',
    check_eval({"a": 1, "b": 20, "c": 3}),
    hint="right side overwrites on duplicate keys",
)

player.explain("String methods  removeprefix  /  removesuffix  (3.9+)", f"""\
  Strip a KNOWN prefix or suffix. Safer than .replace():

      "unhappy".removeprefix("un")       → "happy"
      "report.txt".removesuffix(".txt")  → "report"
      "hello".removeprefix("hi")         → "hello"   (no match = unchanged)

  {cyan("Why not just slice or .replace?")}
      "readme.md".replace(".md", "")     → "readme"
      "redo.md.md".replace(".md", "")    → "redo"       # removes BOTH!
      "redo.md.md".removesuffix(".md")   → "redo.md"    # only the suffix""")

player.quiz(
    'Use  removesuffix  to turn "report.json" into "report".',
    check_eval("report"),
    hint='.removesuffix(".json")',
)

player.explain("f-string improvements (3.12+)", f"""\
  Older Python: you couldn't reuse quote styles inside f-strings:
      f"Hello, {{user['name']}}"         # OK in 3.11-
      f"Hello, {{user["name"]}}"         # SyntaxError in 3.11-

  3.12+ removes that restriction — any quote style works inside braces,
  and expressions can span multiple lines:

      f"{{obj.items["key"]}}"            # nested same quotes: fine
      f"{{'\\n'.join(parts)}}"            # backslash inside: fine

  You still need  !r  / !s  / !a  and format specs the same way.""")

player.explain("Exception groups  except*  (3.11+)", f"""\
  For concurrent code — one operation can raise MANY exceptions at once:

      try:
          await asyncio.gather(task1(), task2(), task3())
      except* TimeoutError as eg:
          for e in eg.exceptions: ...
      except* ValueError as eg:
          ...

  {cyan("ExceptionGroup")} packages multiple exceptions. The  except*  matches
  any of them, re-raising the rest.

  {cyan("You probably won't write these often")} unless you write async code.""")

player.explain("tomllib — TOML in the stdlib (3.11+)", f"""\
      import tomllib

      with open("pyproject.toml", "rb") as f:    # note: binary mode
          data = tomllib.load(f)

      name = data["project"]["name"]

  {cyan("Read-only.")} For writing TOML, you still need the  tomli-w  or
  tomlkit  third-party packages.""")

player.explain("Newer syntactic goodies — grab bag", f"""\
  {cyan("Generic type syntax (3.12+):")}
      def first[T](xs: list[T]) -> T: ...
      class Box[T]: ...

  {cyan("type statement (3.12+):")}
      type Vector = list[float]          # cleaner than Vector: TypeAlias

  {cyan("PEP 701 f-strings (3.12+):")} f-strings use the same parser as
  regular Python now. Most weird quoting/backslash issues are gone.

  {cyan("PEP 709 comprehension inlining (3.12+):")} list comps are faster
  (no implicit function call overhead).""")

player.play()
