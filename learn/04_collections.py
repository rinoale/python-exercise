"""Lesson 04: Collections — list, tuple, dict, set"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from quiz_player import QuizPlayer, check_eval, cyan

player = QuizPlayer("Lesson 04: Collections")

player.explain("list — ordered, mutable", f"""\
  Lists hold ordered items. Indexable, sliceable, mutable.

      nums = [1, 2, 3]
      nums.append(4)         # → [1, 2, 3, 4]
      nums[0] = 99           # → [99, 2, 3, 4]
      nums.pop()             # → 4, list becomes [99, 2, 3]
      nums + [10, 20]        # → [99, 2, 3, 10, 20]  (new list)
      len(nums)              # → 3

  {cyan('Watch out:')} methods like .append() and .sort() mutate in place
  and return None.  [1,2,3].append(4) is None!""")

player.quiz(
    "Type an expression: take [1, 2, 3, 4, 5] and get the last 3 elements as a list.",
    check_eval([3, 4, 5]),
    hint="slicing: [start:stop]",
)

player.quiz(
    "Type an expression producing the list [1, 2, 3, 4] built from [1, 2, 3] plus the number 4. "
    "(Remember: .append() returns None!)",
    check_eval([1, 2, 3, 4]),
    hint="use + with [4], not .append()",
)

player.quiz(
    "Type an expression that sorts [3, 1, 4, 1, 5] ascending. "
    "(Hint: use the built-in sorted(), not .sort())",
    check_eval([1, 1, 3, 4, 5]),
)

player.explain("tuple — ordered, immutable", f"""\
  Tuples look like lists but use (). They can't be changed.

      point = (3, 4)
      x, y = point          # unpacking
      point[0]              # → 3
      point[0] = 99         # TypeError — immutable

  {cyan('When to use a tuple vs a list:')}
      list   — items that will change, items of same kind (e.g. users)
      tuple  — fixed groupings, record-like (e.g. (x, y) coordinate)

  Single-element tuple needs the trailing comma:
      (5,)    is a tuple       (5)    is just 5 in parens""")

player.quiz(
    "Unpack the tuple (10, 20) into variables and return their sum. "
    "Type a single expression using the tuple.",
    check_eval(30),
    hint="sum((10, 20))   or  10 + 20",
)

player.explain("dict — key → value mapping", f"""\
  Unordered (in concept — Python 3.7+ preserves insertion order) lookup table.

      person = {{"name": "Alice", "age": 30}}
      person["name"]            → "Alice"
      person["email"] = "..."   # add/update
      person.get("phone", "n/a") → "n/a"    (no KeyError on missing)
      "age" in person            → True
      list(person.keys())        → ["name", "age", "email"]
      list(person.items())       → [("name","Alice"), ...]

  {cyan('Merge two dicts (Python 3.9+):')}
      {{"a": 1}} | {{"b": 2}}   → {{"a": 1, "b": 2}}""")

player.quiz(
    'What does  {{"a": 1, "b": 2}}.get("c", 0)  return?',
    check_eval(0),
    hint="get(key, default) returns default if key is missing",
)

player.quiz(
    'Merge {{"a": 1}} and {{"b": 2}} into a single dict using the | operator.',
    check_eval({"a": 1, "b": 2}),
    hint='{"a": 1} | {"b": 2}',
)

player.explain("set — unique, unordered", f"""\
  Sets hold unique items. Fast membership test. Support math operations.

      s = {{1, 2, 3, 2, 1}}      # duplicates dropped → {{1, 2, 3}}
      s.add(4)                  # → {{1, 2, 3, 4}}
      3 in s                    # → True

  {cyan('Set operations:')}
      a | b   union          (in a or b)
      a & b   intersection   (in both)
      a - b   difference     (in a, not in b)
      a ^ b   symmetric diff (in a or b, not both)

  Empty set is set(), not {{}} — {{}} is an empty dict!""")

player.quiz(
    "How many unique elements are in [1, 2, 2, 3, 3, 3] ?  "
    "(type an expression that returns the count)",
    check_eval(3),
    hint="len(set([...]))",
)

player.quiz(
    "Compute the intersection of {1, 2, 3} and {2, 3, 4}.",
    check_eval({2, 3}),
    hint="& for intersection",
)

player.play()
