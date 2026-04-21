"""Lesson 03: Strings"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from quiz_player import QuizPlayer, check_eval, check_literal, cyan

player = QuizPlayer("Lesson 03: Strings")

player.explain("Quotes and escapes", f"""\
  Strings use '...', "...", or triple quotes \"\"\"...\"\"\".
  Triple-quoted strings preserve newlines.

      s1 = 'hello'
      s2 = "can't"          # " avoids escaping the apostrophe
      s3 = \"\"\"line 1
      line 2\"\"\"             # multi-line

  {cyan('Common escapes:')}  \\n newline   \\t tab   \\\\ backslash   \\" double quote

  Strings are {cyan('immutable')} — you can't change characters in place:
      s = "abc"
      s[0] = "X"   # TypeError!
      s = "X" + s[1:]   # build a new string instead""")

player.quiz(
    'What is the first character of "hello" ?  (use indexing)',
    check_eval("h"),
    hint='"hello"[0]',
)

player.quiz(
    'What is "hello"[-1] ?',
    check_eval("o"),
    hint="negative indices count from the end",
)

player.explain("Slicing", f"""\
  {cyan('s[start:stop:step]')} — stop is exclusive.

      s = "hello"
      s[1:4]    → "ell"
      s[:2]     → "he"       (omitted start = 0)
      s[2:]     → "llo"      (omitted stop = end)
      s[::2]    → "hlo"      (every 2nd char)
      s[::-1]   → "olleh"    (reversed!)""")

player.quiz(
    'What does "hello"[1:4] return?',
    check_eval("ell"),
)

player.quiz(
    'Type an expression that reverses the string "python".',
    check_eval("nohtyp"),
    hint='slicing with step -1',
)

player.explain("Common string methods", f"""\
  Methods return NEW strings (original is unchanged — strings are immutable).

      "  hi  ".strip()          → "hi"
      "HELLO".lower()           → "hello"
      "hello".upper()           → "HELLO"
      "abc".replace("b", "X")   → "aXc"
      "a,b,c".split(",")        → ["a", "b", "c"]
      "-".join(["x", "y", "z"]) → "x-y-z"
      "hi".startswith("h")      → True
      "hello".find("l")         → 2   (first index, or -1)""")

player.quiz(
    'Type an expression that joins ["a", "b", "c"] into the string "a-b-c".',
    check_eval("a-b-c"),
    hint='"-".join([...])',
)

player.quiz(
    'Split the string "one,two,three" on commas into a list.',
    check_eval(["one", "two", "three"]),
    hint='.split(",")',
)

player.explain("f-strings — formatted string literals", f"""\
  Put an expression inside {{...}} in an f-string:

      name = "Alice"
      age = 30
      f"Hello, {{name}}! You are {{age}}."
      → "Hello, Alice! You are 30."

  {cyan('Format specifiers:')} put them after a colon:
      f"{{3.14159:.2f}}"       → "3.14"        (2 decimal places)
      f"{{42:>5}}"             → "   42"        (right-align, width 5)
      f"{{42:0>5}}"            → "00042"        (pad with zeros)
      f"{{42:b}}"              → "101010"       (binary)
      f"{{1234567:,}}"         → "1,234,567"    (thousands separator)""")

player.quiz(
    'Build an f-string for pi=3.14159 shown to 2 decimal places. '
    'Expected result: the string "pi = 3.14"',
    check_eval("pi = 3.14"),
    hint='f"pi = {3.14159:.2f}"',
)

player.quiz(
    'Type an f-string expression whose value is "007" (the number 7, zero-padded to width 3).',
    check_eval("007"),
    hint='f"{7:03d}" or f"{7:0>3}"',
)

player.play()
