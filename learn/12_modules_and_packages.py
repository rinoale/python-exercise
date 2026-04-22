"""Lesson 12: Modules and Packages"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from quiz_player import QuizPlayer, check_eval, check_literal, check_code, cyan

player = QuizPlayer("Lesson 12: Modules and Packages")

player.explain("Importing — three forms", f"""\
      import math                       # math.sqrt, math.pi, ...
      from math import sqrt, pi         # sqrt(16), pi
      from math import sqrt as square_root      # rename

  {cyan("Guideline:")} prefer  `import module`  unless the name is used many
  times or the path is long. Wildcard  from x import *  is discouraged —
  it pollutes your namespace and breaks name tracking in tools.""")

player.quiz(
    "Import math and compute the square root of 16.\n"
    "  Assign the result to a variable named  result.\n"
    "  Expected:  result == 4.0",
    check_code(lambda ns: (
        ns.get("result") == 4.0 and type(ns.get("result")) is float,
        "got 4.0" if ns.get("result") == 4.0 else f"result is {ns.get('result')!r}",
    )),
    hint="import math\\nresult = math.sqrt(16)",
    multiline=True,
)

player.quiz(
    "Import math and format math.pi to 2 decimal places as a string.\n"
    "  Assign the result to a variable named  result.\n"
    "  Expected:  result == \"3.14\"",
    check_code(lambda ns: (
        ns.get("result") == "3.14",
        "got '3.14'" if ns.get("result") == "3.14" else f"result is {ns.get('result')!r}",
    )),
    hint='import math\\nresult = f"{math.pi:.2f}"',
    multiline=True,
)

player.explain("if __name__ == \"__main__\" — script vs module", f"""\
  When Python runs a file, it sets {cyan("__name__")} to:
      - the string  "__main__"   if run directly:  python myfile.py
      - the module name          if imported:  import myfile

  Use this to separate reusable functions from the script entry point:

      # file: greet.py
      def say_hi(name): return f"hi, {{name}}"

      if __name__ == "__main__":
          import sys
          print(say_hi(sys.argv[1]))

  Running  python greet.py Alice  prints  hi, Alice.
  But  import greet  defines  say_hi  only — the  if  block is SKIPPED,
  so nothing is printed on import.""")

player.explain("Package structure", f"""\
  A {cyan("package")} is a folder with an  __init__.py  (can be empty).

      myapp/
        __init__.py
        core.py
        utils/
          __init__.py
          strings.py
          numbers.py

  Import paths mirror the folder structure:
      from myapp.core import main
      from myapp.utils import strings
      from myapp.utils.strings import slugify

  {cyan("Relative imports")} (inside a package):
      from .core import main            # same package
      from ..utils import strings       # parent package

  Only use relative imports inside a package. Scripts run from the
  command line should use absolute imports.""")

player.explain("Installing and discovering packages", f"""\
      pip install requests                  # installs to your env
      pip install -e .                      # editable install of local project
      pip install -r requirements.txt       # batch install
      pip list                              # what's installed
      pip show requests                     # where is it installed

  {cyan("Where does Python look?")}
      sys.path   — list of directories searched for imports
      First match wins. Your own files can accidentally shadow
      stdlib modules — if you name a file  random.py  next to a script
      that  import random, your file gets imported.

  Check sys.path if imports behave mysteriously.""")

_cwd = os.getcwd()
player.quiz(
    "Import os and get your current working directory as a string.\n"
    "  Assign it to a variable named  result.\n"
    f"  Expected:  result == {_cwd!r}",
    check_code(lambda ns: (
        ns.get("result") == _cwd,
        f"got {_cwd!r}" if ns.get("result") == _cwd else f"result is {ns.get('result')!r}",
    )),
    hint="import os\\nresult = os.getcwd()",
    multiline=True,
)

player.play()
