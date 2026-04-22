"""Lesson 16: Standard Library Tour"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from quiz_player import QuizPlayer, check_code, cyan

player = QuizPlayer("Lesson 16: Standard Library Tour")

player.explain("collections — specialized containers", f"""\
  {cyan("Counter")} — frequency tables:
      from collections import Counter
      c = Counter("mississippi")
      c.most_common(2)          → [('i', 4), ('s', 4)]
      c["x"]                    → 0 (no KeyError)

  {cyan("defaultdict")} — auto-initialized values:
      from collections import defaultdict
      groups = defaultdict(list)
      for x in items:
          groups[x.category].append(x)       # no `if not in` needed

  {cyan("deque")} — fast appends and pops from both ends:
      from collections import deque
      q = deque([1, 2, 3])
      q.appendleft(0)     # O(1); on a list this is O(n)
      q.pop()             # from right; .popleft() from left

  {cyan("namedtuple")} — tuple with field names (for legacy; use dataclass today):
      from collections import namedtuple
      Point = namedtuple("Point", "x y")
      p = Point(3, 4); p.x, p.y""")

def _check_counter(ns):
    r = ns.get("result")
    if r == "a":
        return True, f"got {r!r}"
    return False, f"got {r!r}; expected 'a' — assign to `result`"

player.quiz(
    'Use Counter to find the most common character in "abracadabra".\n'
    '  Assign the character (a string) to `result`.',
    check_code(_check_counter),
    hint='from collections import Counter\nresult = Counter("abracadabra").most_common(1)[0][0]',
    multiline=True,
)

player.explain("itertools — iterator building blocks", f"""\
      import itertools as it

      it.chain([1,2], [3,4])              → 1 2 3 4
      it.count(10, 2)                     → 10, 12, 14, ...    (infinite)
      it.cycle("ab")                      → a, b, a, b, ...    (infinite)
      it.repeat("x", 3)                   → x, x, x
      it.islice(big, 5)                   → first 5 items
      it.combinations("ABC", 2)           → AB, AC, BC
      it.permutations("ABC", 2)           → AB, AC, BA, BC, CA, CB
      it.product([1,2], [3,4])            → (1,3) (1,4) (2,3) (2,4)
      it.groupby(sorted_data, key=...)    → grouped consecutive runs

  {cyan("All lazy.")} They return iterators — wrap with list() to see output.""")

def _check_chain(ns):
    r = ns.get("result")
    if r == [1, 2, 3, 4, 5]:
        return True, f"got {r!r}"
    return False, f"got {r!r}; expected [1, 2, 3, 4, 5] — assign to `result`"

player.quiz(
    "Use itertools.chain to flatten [[1,2],[3,4],[5]] into a list.\n"
    "  Assign the list to `result`.  Expected: [1, 2, 3, 4, 5]",
    check_code(_check_chain),
    hint="import itertools\nresult = list(itertools.chain([1,2],[3,4],[5]))",
    multiline=True,
)

def _check_combinations(ns):
    r = ns.get("result")
    if r == 6:
        return True, f"got {r!r}"
    return False, f"got {r!r}; expected 6 — assign to `result`"

player.quiz(
    "How many 2-element combinations are in 'ABCD'?\n"
    "  Use itertools.combinations.  Assign the count to `result`.  Expected: 6",
    check_code(_check_combinations),
    hint="import itertools\nresult = len(list(itertools.combinations('ABCD', 2)))",
    multiline=True,
)

player.explain("functools — function tools", f"""\
  {cyan("partial")} — pre-fill some arguments:
      from functools import partial
      int16 = partial(int, base=16)
      int16("ff")                → 255

  {cyan("reduce")} — fold a sequence to a single value:
      from functools import reduce
      reduce(lambda a, b: a * b, [1, 2, 3, 4])    → 24

  {cyan("cache / lru_cache")} — memoization:
      from functools import cache
      @cache
      def fib(n): return n if n < 2 else fib(n-1)+fib(n-2)

  {cyan("total_ordering")} — fill in missing comparison ops:
      @total_ordering
      class Version:
          def __eq__(self, o): ...
          def __lt__(self, o): ...
          # Now <=, >, >=, != are auto-generated.""")

def _check_reduce(ns):
    r = ns.get("result")
    if r == 120:
        return True, f"got {r!r}"
    return False, f"got {r!r}; expected 120 — assign to `result`"

player.quiz(
    "Use functools.reduce to compute 1*2*3*4*5 (factorial of 5).\n"
    "  Assign the number to `result`.  Expected: 120",
    check_code(_check_reduce),
    hint="from functools import reduce\nresult = reduce(lambda a,b: a*b, range(1, 6))",
    multiline=True,
)

player.explain("pathlib — modern file paths", f"""\
      from pathlib import Path
      p = Path("/home/user/notes.txt")
      p.exists(), p.is_file(), p.is_dir()
      p.suffix                   → ".txt"
      p.stem                     → "notes"
      p.parent                   → Path("/home/user")
      p.name                     → "notes.txt"
      Path.home() / "docs" / "a.md"     # path join with /
      p.read_text(), p.write_text("...")
      list(Path.cwd().glob("*.py"))      # glob matching""")

player.explain("datetime, json, re — the daily three", f"""\
  {cyan("datetime:")}
      from datetime import datetime, timedelta, timezone
      now = datetime.now()
      now + timedelta(days=7)
      datetime.fromisoformat("2026-04-20T12:00:00")
      now.strftime("%Y-%m-%d %H:%M")

  {cyan("json:")}
      import json
      json.dumps({{"a": 1, "b": [1, 2]}})      # obj → str
      json.loads('{{"a": 1}}')                  # str → obj
      json.dump(obj, open("f.json", "w"))     # to file
      json.load(open("f.json"))                # from file

  {cyan("re (regex):")}
      import re
      re.search(r"\\d+", "hello 42")          # Match or None
      re.findall(r"\\w+", "one two three")    # list of all matches
      re.sub(r"\\s+", " ", messy)             # replace whitespace runs
      re.compile(r"...")                       # compile once, reuse""")

def _check_regex(ns):
    r = ns.get("result")
    if r == ["123", "45"]:
        return True, f"got {r!r}"
    return False, f"got {r!r}; expected [\"123\", \"45\"] — assign to `result`"

player.quiz(
    'Extract all digit groups from "abc123def45" as a list of strings.\n'
    '  Assign the list to `result`.  Expected: ["123", "45"]',
    check_code(_check_regex),
    hint='import re\nresult = re.findall(r"\\d+", "abc123def45")',
    multiline=True,
)

player.play()
