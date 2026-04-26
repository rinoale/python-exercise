"""
Advanced Quiz 01 — Metaclasses, Descriptors, Concurrency, Advanced Patterns, Hard Algorithms
Topics: dunder methods, descriptors, metaclasses, async, DP, graphs, advanced patterns

Instructions:
  - Each question is a function that returns your answer.
  - Run this file:  python quiz/advanced_01.py
  - Fill in the body of each function where you see `pass`.
"""


# ============================================================
# SECTION 1: Advanced Python Syntax
# ============================================================

def q01_custom_descriptor():
    """
    Write a descriptor class `Validated` that only accepts positive integers.
    Raise ValueError on set if the value is not a positive int.

    class Stats:
        score = Validated()
        level = Validated()

    s = Stats()
    s.score = 10     # OK
    s.level = 5      # OK

    Test: try setting s.score = -1, catch ValueError, return True.
    Then return (s.score, s.level, caught_error).

    Expected: (10, 5, True)
    """
    # YOUR CODE HERE
    pass


def q02_metaclass_registry():
    """
    Write a metaclass `RegistryMeta` that automatically registers
    every class created with it into a dict called `registry`,
    keyed by class name.

    class Base(metaclass=RegistryMeta): pass
    class Foo(Base): pass
    class Bar(Base): pass

    Return sorted(RegistryMeta.registry.keys())
    Expected: ["Bar", "Base", "Foo"]
    """
    # YOUR CODE HERE
    pass


def q03_slots_vs_dict():
    """
    Create two classes:
      - PointDict: regular class with x, y attributes
      - PointSlots: class with __slots__ = ('x', 'y')

    Create 1000 instances of each (x=1, y=2).
    Use sys.getsizeof on one instance of each.

    Return True if PointSlots instance is smaller than PointDict instance.
    Expected: True
    """
    import sys
    # YOUR CODE HERE
    pass


def q04_async_gather():
    """
    Write an async function that launches 3 coroutines concurrently
    using asyncio.gather. Each coroutine `delayed_value(val, delay)`
    should await asyncio.sleep(delay) then return val.

    Calls:
      delayed_value("a", 0.03)
      delayed_value("b", 0.01)
      delayed_value("c", 0.02)

    Gather them and return the list of results.
    Expected: ["a", "b", "c"]  (gather preserves call order, not completion order)
    """
    import asyncio
    # YOUR CODE HERE
    pass


# ============================================================
# SECTION 2: Good Code vs Bad Code
# ============================================================

def q05_dataclass_vs_manual():
    """
    BAD version (given — do NOT change):
    """
    class PersonBad:
        def __init__(self, name, age, email):
            self.name = name
            self.age = age
            self.email = email
        def __repr__(self):
            return f"PersonBad(name={self.name!r}, age={self.age!r}, email={self.email!r})"
        def __eq__(self, other):
            return isinstance(other, PersonBad) and (self.name, self.age, self.email) == (other.name, other.age, other.email)

    """
    Rewrite using @dataclass. The class should be called PersonGood.
    Create PersonGood("Alice", 30, "alice@example.com").
    Return (person.name, person.age, person.email).
    Expected: ("Alice", 30, "alice@example.com")
    """
    # YOUR CODE HERE
    pass


def q06_property_vs_getter():
    """
    BAD version (given — do NOT change):
    """
    class RectangleBad:
        def __init__(self, w, h):
            self._w = w
            self._h = h
        def get_area(self):
            return self._w * self._h
        def set_width(self, w):
            if w <= 0: raise ValueError
            self._w = w

    """
    Rewrite using @property for `width` (with validation) and `area` (read-only).
    Create RectangleGood(4, 5).
    Return (rect.area, rect.width).
    Then set rect.width = 10 and return rect.area again.
    Return (first_area, width, second_area).
    Expected: (20, 4, 50)
    """
    # YOUR CODE HERE
    pass


# ============================================================
# SECTION 3: Advanced Design Patterns
# ============================================================

def q07_chain_of_responsibility():
    """
    Implement Chain of Responsibility for processing purchase requests.

    Each handler has an approval limit:
      Manager:  up to 1000
      Director: up to 5000
      VP:       up to 50000

    If a handler can't approve, it passes to the next.
    If nobody can approve, return "rejected".

    Test:
      chain = Manager -> Director -> VP

      chain.handle(500)    -> "Manager approved 500"
      chain.handle(3000)   -> "Director approved 3000"
      chain.handle(30000)  -> "VP approved 30000"
      chain.handle(100000) -> "rejected"

    Return the tuple of four results.
    """
    # YOUR CODE HERE
    pass


def q08_abstract_factory():
    """
    Implement an Abstract Factory for UI components.

    Two families: "light" and "dark".

    Each family produces:
      - create_button()  -> "<theme> button"
      - create_checkbox() -> "<theme> checkbox"

    Write a function `build_ui(factory)` that returns:
      (factory.create_button(), factory.create_checkbox())

    Return (build_ui(LightFactory()), build_ui(DarkFactory()))
    Expected: (("light button", "light checkbox"), ("dark button", "dark checkbox"))
    """
    # YOUR CODE HERE
    pass


# ============================================================
# SECTION 4: Hard Algorithms
# ============================================================

def q09_lru_cache_manual():
    """
    Implement an LRU Cache with capacity 3 using OrderedDict.

    class LRUCache:
        __init__(self, capacity)
        get(self, key) -> value or -1
        put(self, key, value) -> None

    Operations:
      cache = LRUCache(3)
      cache.put(1, "a")
      cache.put(2, "b")
      cache.put(3, "c")
      r1 = cache.get(1)       # "a" — moves 1 to most recent
      cache.put(4, "d")       # evicts key 2 (least recently used)
      r2 = cache.get(2)       # -1 (evicted)
      r3 = cache.get(3)       # "c"

    Return (r1, r2, r3)
    Expected: ("a", -1, "c")
    """
    # YOUR CODE HERE
    pass


def q10_longest_increasing_subsequence():
    """
    Find the LENGTH of the longest increasing subsequence (LIS).
    Use dynamic programming (O(n^2) is fine).

    Input: [10, 22, 9, 33, 21, 50, 41, 60, 80]
    Expected: 6  (the LIS is [10, 22, 33, 50, 60, 80])
    """
    nums = [10, 22, 9, 33, 21, 50, 41, 60, 80]
    # YOUR CODE HERE
    pass


def q11_graph_shortest_path():
    """
    Find the shortest path in an unweighted graph using BFS.

    Graph (adjacency list):
      A -> B, C
      B -> A, D, E
      C -> A, F
      D -> B
      E -> B, F
      F -> C, E

    Return the shortest path from "A" to "F" as a list of nodes.
    Expected: ["A", "C", "F"]
    """
    # YOUR CODE HERE
    pass


def q12_knapsack():
    """
    0/1 Knapsack Problem using dynamic programming.

    Items: [(weight, value), ...]
      (2, 3), (3, 4), (4, 5), (5, 6)

    Knapsack capacity: 8

    Return the maximum total value achievable.
    Expected: 10  (items with weight 3,5 -> value 4+6 or weight 2,3 -> 3+4=7... work it out!)
    """
    items = [(2, 3), (3, 4), (4, 5), (5, 6)]
    capacity = 8
    # YOUR CODE HERE
    pass


# ============================================================
# AUTO-GRADER
# ============================================================

def _grade():
    results = []

    def check(name, got, expected):
        passed = got == expected
        mark = "PASS" if passed else "FAIL"
        results.append(passed)
        print(f"  [{mark}] {name}")
        if not passed:
            print(f"         expected: {expected}")
            print(f"         got:      {got}")

    print("\n=== Advanced Quiz 01 ===\n")

    check("q01 custom descriptor", q01_custom_descriptor(), (10, 5, True))
    check("q02 metaclass registry", q02_metaclass_registry(), ["Bar", "Base", "Foo"])
    check("q03 slots vs dict", q03_slots_vs_dict(), True)
    check("q04 async gather", q04_async_gather(), ["a", "b", "c"])
    check("q05 dataclass vs manual", q05_dataclass_vs_manual(), ("Alice", 30, "alice@example.com"))
    check("q06 property vs getter", q06_property_vs_getter(), (20, 4, 50))
    check("q07 chain of responsibility", q07_chain_of_responsibility(), (
        "Manager approved 500",
        "Director approved 3000",
        "VP approved 30000",
        "rejected",
    ))
    check("q08 abstract factory", q08_abstract_factory(), (
        ("light button", "light checkbox"),
        ("dark button", "dark checkbox"),
    ))
    check("q09 LRU cache", q09_lru_cache_manual(), ("a", -1, "c"))
    check("q10 longest increasing subseq", q10_longest_increasing_subsequence(), 6)
    check("q11 graph shortest path", q11_graph_shortest_path(), ["A", "C", "F"])
    check("q12 knapsack", q12_knapsack(), 10)

    total = len(results)
    passed = sum(results)
    print(f"\nScore: {passed}/{total}")
    if passed == total:
        print("You're a Python master!\n")
    else:
        print("Review advanced OOP, concurrency, and algorithm lessons.\n")


if __name__ == "__main__":
    _grade()
