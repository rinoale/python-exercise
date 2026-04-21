"""Lesson 07: I/O Basics — print, input, files"""
import os
import sys
import tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from quiz_player import QuizPlayer, check_eval, check_literal, check_code, cyan

player = QuizPlayer("Lesson 07: I/O Basics")

player.explain("print() — more flexible than it looks", f"""\
      print("a", "b", "c")              → a b c       (default sep=" ")
      print("a", "b", sep="-")          → a-b
      print("no newline", end="")       # suppress trailing \\n
      print("err", file=sys.stderr)     # print to stderr

  {cyan('Printing a single value:')}
      print(value)                      # calls str(value) under the hood
      print(f"{{value!r}}")              # !r uses repr() — useful for debug""")

player.quiz(
    'Type the EXACT string that this call prints:\n'
    '    print("a", "b", "c", sep="+")',
    check_literal("a+b+c"),
    hint='sep="+" is used BETWEEN values',
)

player.quiz(
    'Type the EXACT string that this call prints:\n'
    '    print("hello", end="!")',
    check_literal("hello!"),
    hint="end= replaces the default newline",
)

player.explain("input() — read a line from stdin", f"""\
      name = input("Your name: ")       # always returns a str
      age = int(input("Age: "))         # convert explicitly if needed

  {cyan('Type conversion is on you:')}
      n = input("n: ")     # even if user types "42", n is "42" (str)
      n = int(n)           # now 42 (int)

  For non-trivial input parsing, consider argparse or click instead.""")

# Create a temp file for the next quiz
_tmp = tempfile.NamedTemporaryFile("w", delete=False, suffix=".txt", encoding="utf-8")
_tmp.write("hello from a file!")
_tmp.close()
_tmp_path = _tmp.name

player.explain("Reading and writing files", f"""\
  Use {cyan('with open(path, mode)')} so the file closes automatically.

      with open("notes.txt", "r") as f:         # read text
          content = f.read()
      with open("notes.txt", "w") as f:         # write (truncates!)
          f.write("new content\\n")
      with open("notes.txt", "a") as f:         # append
          f.write("more\\n")

  {cyan('Modes:')}
      "r"   read (default)
      "w"   write — truncates the file to zero
      "a"   append
      "rb"/"wb"  binary mode (bytes instead of str)

  {cyan('Iterating a file yields lines (with their newlines):')}
      with open(path) as f:
          for line in f:
              print(line.rstrip())

  For this quiz we've created a file at:
      {_tmp_path}
  Its contents are the string:  "hello from a file!" """)

player.quiz(
    f"Read the full contents of the file at:\n"
    f"    {_tmp_path}\n"
    f"  Expected: the string \"hello from a file!\".\n"
    "  Type a single expression that returns the file's text.",
    check_eval("hello from a file!"),
    hint=f'open("{_tmp_path}").read()',
)

player.explain("pathlib — the modern way to handle paths", f"""\
  {cyan('from pathlib import Path')}
      p = Path("/tmp/notes.txt")
      p.exists()              # True/False
      p.read_text()           # read full file as str (Python 3.5+)
      p.write_text("hi!")     # overwrite text
      p.suffix                # ".txt"
      p.parent                # Path("/tmp")
      p / "sub" / "file.md"   # join with  /  operator

  Prefer pathlib over os.path for new code.""")

player.quiz(
    f"Use pathlib to read the same file. (Multi-line OK — import then read.)\n"
    f"    path: {_tmp_path}\n"
    f"  Assign the file's text to a variable named  content.\n"
    f"  Expected:  content == \"hello from a file!\"",
    check_code(lambda ns: (
        ns.get("content") == "hello from a file!",
        "read via pathlib"
        if ns.get("content") == "hello from a file!"
        else f"content is {ns.get('content')!r}",
    )),
    hint=f'from pathlib import Path\\ncontent = Path("{_tmp_path}").read_text()',
    multiline=True,
)

player.explain("Summary — end of Level 1", f"""\
  You've covered the beginner tier:
    01 Values and variables, types, isinstance
    02 Numbers, operators, truthy/falsy
    03 Strings, slicing, f-strings
    04 list / tuple / dict / set
    05 if, for, while, comprehensions
    06 def, lambda, *args, **kwargs
    07 print, input, files, pathlib

  {cyan('Next:')} Level 2 (intermediate) — advanced functions,
  classes, dunder methods, decorators, exceptions, stdlib tour.

  See ../learn/README.md for the full curriculum.""")

player.play()

try:
    os.unlink(_tmp_path)
except OSError:
    pass
