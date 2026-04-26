"""
Beginner Quiz 01 — Python Syntax, Good vs Bad Code, Basic Patterns
Topics: variables, collections, loops, functions, comprehensions, generators

Instructions:
  - Each question is a function that returns your answer.
  - Run this file:  python quiz/beginner_01.py
  - It will grade your answers automatically.
  - Fill in the body of each function where you see `pass`.
"""


# ============================================================
# SECTION 1: Python Syntax
# ============================================================

def q01_swap_variables():
    """
    Swap the values of a and b using Pythonic syntax (one line).
    Return the tuple (a, b) after swapping.
    """
    a = 10
    b = 20
    a, b = b, a
    pass
    return (a, b)


def q02_unpack_and_ignore():
    """
    Given this tuple, unpack ONLY the first and last values.
    Use _ for the values you want to ignore.
    Return (first, last).
    """
    data = ("alpha", "beta", "gamma", "delta")
    first, _, _, last = data
    return (first, last)
    pass


def q03_string_formatting():
    """
    Using an f-string, return the string:
      "name=Alice, age=30, score=95.50"
    Variables are provided below. The score must show exactly 2 decimal places.
    """
    name = "Alice"
    age = 30
    score = 95.5
    return f"name={name}, age={age}, score={score:.2f}"
    pass


def q04_truthy_falsy():
    """
    Without using `if`, return a list containing ONLY the truthy values
    from the input list below. Use the built-in `filter` function.
    Return a list (not a filter object).
    """
    values = [0, "", None, "hello", 42, [], {}, (1,), False, " "]
    return list(filter(lambda v: bool(v) and v, values))
    pass


def q05_dict_get_default():
    """
    Given the dictionary below, safely get the value for key "z".
    If "z" doesn't exist, return -1.
    Do NOT use try/except or `in` — use a single dict method call.
    """
    d = {"a": 1, "b": 2, "c": 3}
    return getattr(d,'z', -1)
    pass


# ============================================================
# SECTION 2: Good Code vs Bad Code
# ============================================================

def q06_sum_of_squares_bad():
    """
    BAD version (given — do NOT change):
    """
    result = []
    for i in range(10):
        result.append(i * i)
    return sum(result)

def q06_sum_of_squares_good():
    """
    Rewrite q06 in a more Pythonic and efficient way.
    Hint: you don't need a list at all — use a generator expression.
    Return the same result as q06_sum_of_squares_bad().
    """
    pass
    return sum([i**2 for i in range(10)])


def q07_count_words_bad():
    """
    BAD version (given — do NOT change):
    """
    text = "the cat sat on the mat the cat"
    words = text.split()
    counts = {}
    for w in words:
        if w in counts:
            counts[w] = counts[w] + 1
        else:
            counts[w] = 1
    return counts

def q07_count_words_good():
    """
    Rewrite q07 using a standard-library class that handles counting.
    Return the same result type (a dict-like mapping of word -> count).
    """
    text = "the cat sat on the mat the cat"
    pass
    from collections import Counter
    return Counter(text.split())


def q08_check_all_positive_bad():
    """
    BAD version (given — do NOT change):
    """
    nums = [3, 7, 1, 9, 4]
    flag = True
    for n in nums:
        if n <= 0:
            flag = False
            break
    return flag

def q08_check_all_positive_good():
    """
    Rewrite q08 in one line using a built-in function.
    """
    nums = [3, 7, 1, 9, 4]
    pass
    return all(i > 0 for i in nums)


# ============================================================
# SECTION 3: Basic Design Patterns
# ============================================================

def q09_singleton():
    """
    Implement a simple Singleton class.
    Calling MyClass() multiple times should return the SAME instance.

    Return True if `MyClass() is MyClass()` (identity check).
    """
    class MyClass():
        _instance = None

        def __new__(cls):
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance
    pass
    return MyClass() is MyClass()


def q10_strategy_pattern():
    """
    Implement a tiny strategy pattern:

    Write a function `apply_operation(a, b, operation)` where
    `operation` is a callable (function) that takes two args.

    Then call it three times:
      r1 = apply_operation(10, 3, <add>)      -> 13
      r2 = apply_operation(10, 3, <subtract>)  -> 7
      r3 = apply_operation(10, 3, <multiply>)  -> 30

    You can use lambdas or regular functions for the operations.
    Return (r1, r2, r3).
    """
    def apply_operation(l, r, fn):
        return fn(l,r)

    r1 = apply_operation(10, 3, lambda l,r: l + r)
    r2 = apply_operation(10, 3, lambda l,r: l - r)
    r3 = apply_operation(10, 3, lambda l,r: l * r)
    pass
    return (r1, r2, r3)


# ============================================================
# SECTION 4: Basic Algorithm
# ============================================================

def q11_two_sum():
    """
    Given a list of integers and a target, return the INDICES of the
    two numbers that add up to the target.

    You may assume exactly one solution exists.
    Try to solve it in O(n) time (use a dict/hash map).

    Example: nums=[2, 7, 11, 15], target=9  ->  (0, 1)
    """
    nums = [2, 7, 11, 15]
    target = 9
    dict_pair = {i: target - i for i in nums}
    pass
    result = None
    for i in nums:
        r = dict_pair.get(i)
        if r and dict_pair.get(r):
            result = (nums.index(i), nums.index(r))
            break
    return result

def q11_two_sum_good():
    nums = [2, 7, 11, 15]
    target = 9
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return (seen[complement], i)
        seen[num] = i


def q12_reverse_string_in_place():
    """
    Given a list of characters, reverse it IN-PLACE (do not create a new list).
    Return the same list object.

    Hint: use two pointers (left, right) swapping towards the center.
    """
    chars = list("hello")
    for i, char in enumerate(chars):
        if i > len(chars) / 2:
            break
        j = len(chars) - 1 - i
        chars[i], chars[j] = chars[j], chars[i]
    return chars
    pass

def q12_reverse_string_in_place_good():
    chars = list("hello")
    left, right = 0, len(chars) - 1
    while left < right:
        chars[left], chars[right] = chars[right], chars[left]
        left += 1
        right -= 1
    return chars


# ============================================================
# AUTO-GRADER
# ============================================================

def _grade():
    from collections import Counter
    results = []

    def check(name, got, expected):
        passed = got == expected
        mark = "PASS" if passed else "FAIL"
        results.append(passed)
        print(f"  [{mark}] {name}")
        if not passed:
            print(f"         expected: {expected}")
            print(f"         got:      {got}")

    print("\n=== Beginner Quiz 01 ===\n")

    check("q01 swap variables", q01_swap_variables(), (20, 10))
    check("q02 unpack and ignore", q02_unpack_and_ignore(), ("alpha", "delta"))
    check("q03 string formatting", q03_string_formatting(), "name=Alice, age=30, score=95.50")
    check("q04 truthy/falsy filter", q04_truthy_falsy(), ["hello", 42, (1,), " "])
    check("q05 dict get default", q05_dict_get_default(), -1)
    check("q06 sum of squares (good)", q06_sum_of_squares_good(), q06_sum_of_squares_bad())
    check("q07 count words (good)", dict(q07_count_words_good()) if q07_count_words_good() else None, q07_count_words_bad())
    check("q08 all positive (good)", q08_check_all_positive_good(), q08_check_all_positive_bad())
    check("q09 singleton", q09_singleton(), True)
    check("q10 strategy pattern", q10_strategy_pattern(), (13, 7, 30))
    check("q11 two sum", q11_two_sum(), (0, 1))

    r12 = q12_reverse_string_in_place()
    check("q12 reverse in place", r12, list("olleh"))

    total = len(results)
    passed = sum(results)
    print(f"\nScore: {passed}/{total}")
    if passed == total:
        print("Perfect! You're ready for intermediate.\n")
    else:
        print("Keep going — review the lessons for topics you missed.\n")


if __name__ == "__main__":
    _grade()
