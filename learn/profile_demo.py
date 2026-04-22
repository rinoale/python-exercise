"""
Profile demo — run this to see clear differences between good and bad code.

Usage:
    python learn/profile_demo.py                    # basic timing
    python -m cProfile learn/profile_demo.py        # function-level profile
    python -m cProfile -o out.prof learn/profile_demo.py && snakeviz out.prof  # graphical
"""

import time

N = 30_000


# ── Example 1: list vs set membership ────────────────────
#    "Is x in this collection?" — list scans one by one, set hashes directly.

def find_duplicates_bad(data):
    seen = []
    dupes = []
    for x in data:
        if x in seen:          # O(n) scan every time
            dupes.append(x)
        seen.append(x)
    return dupes


def find_duplicates_good(data):
    seen = set()
    dupes = []
    for x in data:
        if x in seen:          # O(1) hash lookup
            dupes.append(x)
        seen.add(x)
    return dupes


# ── Example 2: nested loop vs dict lookup ────────────────
#    Match users to orders — O(n*m) brute force vs O(n+m) with a dict.

def match_orders_bad(users, orders):
    result = []
    for order in orders:
        for user in users:                # scan all users for each order
            if user["id"] == order["uid"]:
                result.append((user["name"], order["item"]))
                break
    return result


def match_orders_good(users, orders):
    user_map = {u["id"]: u["name"] for u in users}   # build index once
    result = []
    for order in orders:
        name = user_map.get(order["uid"])             # O(1) lookup
        if name:
            result.append((name, order["item"]))
    return result


# ── Run and compare ──────────────────────────────────────

def bench(label, func, *args):
    start = time.perf_counter()
    func(*args)
    elapsed = time.perf_counter() - start
    print(f"  {label}  {elapsed:.3f}s")

data = list(range(N)) + list(range(N // 2))

print(f"── 1. Duplicate finder (N={N:,}) ──")
print(f"       list scan vs set lookup\n")
bench("list  (bad) ", find_duplicates_bad, data)
bench("set   (good)", find_duplicates_good, data)

users  = [{"id": i, "name": f"user_{i}"} for i in range(N)]
orders = [{"uid": i % N, "item": f"item_{i}"} for i in range(N)]

print(f"\n── 2. Match orders (N={N:,}) ──")
print(f"       nested loop vs dict index\n")
bench("loop  (bad) ", match_orders_bad, users, orders)
bench("dict  (good)", match_orders_good, users, orders)
