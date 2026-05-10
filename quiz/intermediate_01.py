"""
Intermediate Quiz 01 — Decorators, Generators, OOP, Design Patterns, Algorithms
Topics: closures, decorators, iterators, classes, dunder methods, patterns, sorting

Instructions:
  - Each question is a function that returns your answer.
  - Run this file:  python quiz/intermediate_01.py
  - Fill in the body of each function where you see `pass`.
"""


# ============================================================
# SECTION 1: Python Syntax — Closures, Decorators, Generators
# ============================================================

def q01_closure_counter():
    """
    Write a function `make_counter()` that returns a function.
    Each time the returned function is called, it returns the next integer
    starting from 0.

    Example:
      c = make_counter()
      c()  -> 0
      c()  -> 1
      c()  -> 2

    Return the results of calling c() three times as a tuple: (0, 1, 2).
    Hint: use `nonlocal`.
    """
    # YOUR CODE HERE — define make_counter, then use it
    def make_counter():
        count = 0
        def counter():
            nonlocal count
            result = count
            count += 1
            return result
        return counter

    c = make_counter()
    return (c(), c(), c())
    pass


def q02_decorator_timer():
    """
    Write a decorator `@track_calls` that counts how many times
    a function has been called. Store the count as an attribute
    `call_count` on the function object itself.

    Apply it to a dummy function `greet()` that returns "hello".
    Call greet() 5 times, then return greet.call_count.

    Expected return: 5
    """
    def track_calls(fn):
        def wrapper(*args, **kwargs):
            wrapper.call_count += 1
            return fn(*args, **kwargs)
        wrapper.call_count = 0

        return wrapper

    @track_calls
    def greet():
        return "hello"
    greet()
    greet()
    greet()
    greet()
    greet()

    return greet.call_count


def q03_fibonacci_generator():
    """
    Write a generator function `fib()` that yields Fibonacci numbers
    indefinitely: 0, 1, 1, 2, 3, 5, 8, 13, ...

    Use it to collect the first 10 Fibonacci numbers into a list.
    Return that list.

    Expected: [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
    """
    # user's original — recursive approach (correct but O(2^n) per call)
    def fib_recursive():
        n = 0
        def fib_internal(i):
            if i + 1 == 1:
                return 0
            elif i + 1 == 2:
                return 1
            return fib_internal(i - 1) + fib_internal(i - 2)

        while True:
            yield fib_internal(n)
            n += 1

    # correct answer — iterative generator (O(1) per call)
    def fib():
        a, b = 0, 1
        while True:
            yield a
            a, b = b, a + b

    g = fib()
    return [next(g) for _ in range(10)]


def q04_context_manager():
    """
    Write a context manager class `Indenter` that tracks indentation level.
    Each `with` block increases indent by 1, and it decreases when exiting.

    The class should have a method `print_line(text)` that returns
    the text prefixed with (level * 2) spaces.

    Example usage:
      ind = Indenter()
      r1 = ind.print_line("top")             # "top"
      with ind:
          r2 = ind.print_line("child")        # "  child"
          with ind:
              r3 = ind.print_line("grandchild")  # "    grandchild"
          r4 = ind.print_line("child again")  # "  child again"
      r5 = ind.print_line("top again")       # "top again"

    Return (r1, r2, r3, r4, r5).
    """
    class Indenter():
        def __init__(self):
            self.current_cursor = 0
        def __enter__(self):
            self.current_cursor += 1
        def __exit__(self, exc_type, exc_val, exc_tb):
            self.current_cursor -= 1
        def print_line(self, line):
            return f"{' ' * (self.current_cursor * 2)}{line}"

    ind = Indenter()
    r1 = ind.print_line("top")
    with ind:
        r2 = ind.print_line("child")
        with ind:
            r3 = ind.print_line("grandchild")
        r4 = ind.print_line("child again")
    r5 = ind.print_line("top again")

    return (r1, r2, r3, r4, r5)


# ============================================================
# SECTION 2: Good Code vs Bad Code
# ============================================================

def q05_flatten_bad():
    """
    BAD version (given — do NOT change):
    Flattens a nested list one level deep.
    """
    nested = [[1, 2], [3, 4], [5, 6, 7]]
    result = []
    for sublist in nested:
        for item in sublist:
            result.append(item)
    return result

def q05_flatten_good():
    """
    Rewrite using itertools.chain.from_iterable (one line for the return).
    Return a list with the same contents.
    """
    nested = [[1, 2], [3, 4], [5, 6, 7]]
    def flatten(iter):
        for arr in iter:
            yield from arr

    return list(flatten(nested))

def q05_flatten_answer():
    from itertools import chain
    nested = [[1, 2], [3, 4], [5, 6, 7]]
    return list(chain.from_iterable(nested))


def q06_group_by_bad():
    """
    BAD version (given — do NOT change):
    Groups words by their first letter.
    """
    words = ["apple", "avocado", "banana", "blueberry", "cherry"]
    groups = {}
    for w in words:
        key = w[0]
        if key not in groups:
            groups[key] = []
        groups[key].append(w)
    return groups

def q06_group_by_good():
    """
    Rewrite using collections.defaultdict. Return a regular dict at the end.
    """
    words = ["apple", "avocado", "banana", "blueberry", "cherry"]
    from collections import defaultdict
    groups = defaultdict(list)
    for word in words: groups[word[0]].append(word)
    return groups


# ============================================================
# SECTION 3: Design Patterns
# ============================================================

def q07_observer_pattern():
    """
    Implement a simple Observer (pub/sub) pattern:

    class EventBus:
        subscribe(event_name, callback)  — register a callback
        emit(event_name, data)           — call all callbacks for that event

    Usage:
      bus = EventBus()
      log = []
      bus.subscribe("login", lambda data: log.append(f"user {data} logged in"))
      bus.subscribe("login", lambda data: log.append(f"welcome {data}"))
      bus.emit("login", "Alice")

    Return log  ->  ["user Alice logged in", "welcome Alice"]
    """
    from collections import defaultdict

    # user's original — works but defaultdict(dict) + `or []` was fragile
    class EventBusOriginal:
        def __init__(self):
            self.event_map = defaultdict(dict)

        def subscribe(self, event_name, callback):
            event_callbacks = self.event_map[event_name] or []
            event_callbacks.append(callback)
            self.event_map[event_name] = event_callbacks

        def emit(self, event_name, name):
            for callback in self.event_map[event_name]:
                callback(name)

    # correct answer — defaultdict(list) eliminates the workaround
    class EventBus:
        def __init__(self):
            self.event_map = defaultdict(list)

        def subscribe(self, event_name, callback):
            self.event_map[event_name].append(callback)

        def emit(self, event_name, data):
            for callback in self.event_map[event_name]:
                callback(data)

    bus = EventBus()
    log = []
    bus.subscribe("login", lambda data: log.append(f"user {data} logged in"))
    bus.subscribe("login", lambda data: log.append(f"welcome {data}"))
    bus.emit("login", "Alice")

    return log


def q08_decorator_pattern():
    """
    Implement the Decorator pattern (not Python decorators — the OOP pattern).

    Base class:  TextSource with method get_text() -> str
    Concrete:    PlainText("hello") where get_text() returns "hello"
    Decorator:   UpperCase(source) where get_text() returns source.get_text().upper()
    Decorator:   Brackets(source) where get_text() returns f"[{source.get_text()}]"

    Chain them:  Brackets(UpperCase(PlainText("hello")))
    Return the result of get_text()  ->  "[HELLO]"
    """
    # user's original — two bugs: missing self. and returns object instead of text
    class TextSourceOriginal:
        def get_text(self) -> str:
            pass

    class PlainTextOriginal(TextSourceOriginal):
        def __init__(self, text):
            self.text = text

        def get_text(self):
            return self.text

    class UpperCaseOriginal(TextSourceOriginal):
        def __init__(self, source):
            self.source = source

        def get_text(self):
            return source.get_text().upper()       # bug: source → self.source

    class BracketsOriginal(TextSourceOriginal):
        def __init__(self, source):
            self.source = source

        def get_text(self):
            return f"[{source.get_text()}]"        # bug: source → self.source

    # return Brackets(UpperCase(PlainText("hello")))  # bug: missing .get_text()

    # correct answer — proper Decorator pattern with shared TextDecorator base
    class TextSource:
        def get_text(self) -> str:
            pass

    class PlainText(TextSource):
        def __init__(self, text):
            self.text = text

        def get_text(self):
            return self.text

    class TextDecorator(TextSource):
        def __init__(self, source):
            self.source = source

        def get_text(self):
            return self.source.get_text()

    class UpperCase(TextDecorator):
        def get_text(self):
            return self.source.get_text().upper()

    class Brackets(TextDecorator):
        def get_text(self):
            return f"[{self.source.get_text()}]"

    return Brackets(UpperCase(PlainText("hello"))).get_text()


# ============================================================
# SECTION 4: Algorithms
# ============================================================

def q09_valid_parentheses():
    """
    Given a string containing only '(', ')', '{', '}', '[', ']',
    determine if the input string is valid.

    A string is valid if:
      - Open brackets are closed by the same type.
      - Open brackets are closed in the correct order.

    Return a tuple of booleans for these test cases:
      "()"         -> True
      "()[]{}"     -> True
      "(]"         -> False
      "([)]"       -> False
      "{[]}"       -> True

    Expected return: (True, True, False, False, True)
    """
    # user's original — works for given cases but crashes on leading close bracket
    def parentheses_validator_original(input_str):
        paren_pair = { ")": "(", "}": "{", "]": "[" }
        open_arr = []
        for ch in input_str:
            try:
                if ch in paren_pair.values():
                    open_arr.append(ch)
                else:
                    if open_arr[len(open_arr) - 1] != paren_pair[ch]:
                        raise ValueError
                    open_arr.pop()
            except ValueError:
                return False

        return len(open_arr) == 0

    # correct answer — handles empty-stack edge case, no try/except needed
    def parentheses_validator(s):
        pair = {")": "(", "}": "{", "]": "["}
        stack = []
        for ch in s:
            if ch in pair.values():
                stack.append(ch)
            elif not stack or stack.pop() != pair[ch]:
                return False
        return len(stack) == 0

    return (parentheses_validator("()"), parentheses_validator("()[]{}"), parentheses_validator("(]"), parentheses_validator("([)]"), parentheses_validator("{[]}"))


def q10_merge_sort():
    """
    Implement merge sort. Return the sorted list.

    Input: [38, 27, 43, 3, 9, 82, 10]
    Expected: [3, 9, 10, 27, 38, 43, 82]
    """
    nums = [38, 27, 43, 3, 9, 82, 10]
    # YOUR CODE HERE
    pass


def q11_binary_search():
    """
    Implement binary search on a sorted list.
    Return the INDEX of the target, or -1 if not found.

    Test cases (return as a tuple):
      search([1, 3, 5, 7, 9, 11], 7)   -> 3
      search([1, 3, 5, 7, 9, 11], 4)   -> -1
      search([1, 3, 5, 7, 9, 11], 1)   -> 0
      search([1, 3, 5, 7, 9, 11], 11)  -> 5

    Expected return: (3, -1, 0, 5)
    """
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

    print("\n=== Intermediate Quiz 01 ===\n")

    check("q01 closure counter", q01_closure_counter(), (0, 1, 2))
    check("q02 decorator track_calls", q02_decorator_timer(), 5)
    check("q03 fibonacci generator", q03_fibonacci_generator(), [0, 1, 1, 2, 3, 5, 8, 13, 21, 34])
    check("q04 context manager", q04_context_manager(), ("top", "  child", "    grandchild", "  child again", "top again"))
    check("q05 flatten (good)", q05_flatten_good(), q05_flatten_bad())
    check("q06 group by (good)", q06_group_by_good(), q06_group_by_bad())
    check("q07 observer pattern", q07_observer_pattern(), ["user Alice logged in", "welcome Alice"])
    check("q08 decorator pattern", q08_decorator_pattern(), "[HELLO]")
    check("q09 valid parentheses", q09_valid_parentheses(), (True, True, False, False, True))
    check("q10 merge sort", q10_merge_sort(), [3, 9, 10, 27, 38, 43, 82])
    check("q11 binary search", q11_binary_search(), (3, -1, 0, 5))

    total = len(results)
    passed = sum(results)
    print(f"\nScore: {passed}/{total}")
    if passed == total:
        print("Perfect! You're ready for advanced.\n")
    else:
        print("Review decorators, generators, and OOP lessons.\n")


if __name__ == "__main__":
    _grade()
