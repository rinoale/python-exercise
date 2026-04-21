"""Lesson 26: Performance and Profiling"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from quiz_player import QuizPlayer, check_eval, check_code, cyan

player = QuizPlayer("Lesson 26: Performance and Profiling")

player.explain("Rule #1 — measure, don't guess", f"""\
  {cyan("Premature optimization is the root of all evil.")}
  Write clear code first; optimize only where proven slow.

  Typical order of operations when something is slow:
      1. Profile.  Find where time is ACTUALLY spent.
      2. Algorithm. Wrong Big-O beats any micro-optimization.
      3. Data structures. list vs set vs dict membership is 10-1000x.
      4. Avoid work. Cache, memoize, short-circuit.
      5. Push loops to C. NumPy, built-ins, stdlib C modules.
      6. Parallelize or native extension (Cython, Rust).

  Steps 1–4 solve >90% of real problems.""")

player.explain("timeit — microbenchmarks done right", f"""\
      # From the shell:
      python -m timeit -s "x = list(range(1000))" "sum(x)"
      # 1000 loops, best of 5: 12.3 usec per loop

      # In code:
      from timeit import timeit
      t = timeit("sum(range(1000))", number=100_000)

  {cyan("Why timeit, not time.time()?")} timeit disables GC, runs many
  trials, and reports the best time. Manual timing includes noise from
  other processes, JIT warm-up, etc.""")

player.explain("cProfile — find slow functions", f"""\
      # Profile a whole script:
      python -m cProfile -o out.prof myscript.py

      # Look at results:
      python -m pstats out.prof
      (Pdb) sort cumulative
      (Pdb) stats 20                   # top 20 by cumulative time

      # Or graphical:
      pip install snakeviz
      snakeviz out.prof                # interactive flamegraph

  {cyan("What to look for:")} functions with HIGH cumtime (total time
  including calls) but FEW ncalls = algorithm issue. Many calls with
  small tottime = consider caching or batching.""")

player.explain("Data structure gotchas — list vs set vs dict", f"""\
      x in list           → O(n)   slow for big lists
      x in set            → O(1)   use a set for membership checks
      x in dict           → O(1)   same as set

      list.append(x)      → O(1) amortized
      list.insert(0, x)   → O(n)   prepending is slow; use collections.deque
      list.pop()          → O(1)
      list.pop(0)         → O(n)   same — deque.popleft() is O(1)

  {cyan("Classic speedup:")} turn a list into a set when you do many
  "in" checks. 10k-element list membership: ms. set membership: ns.""")

player.quiz(
    "A list vs a set: which gives O(1) membership checks?  Type 'list' or 'set'.",
    check_eval("set"),
    hint="dicts and sets are hash-based",
)

player.explain("Quick Python wins", f"""\
  {cyan("Local variable access is faster than global:")}
      def hot():
          local_len = len            # 1.5-2x in tight loops
          for x in items: local_len(x)

  {cyan("Built-ins in C beat Python loops:")}
      sum(xs)                  # >> explicit loop
      "".join(parts)           # >> s = s + x repeated
      any(pred(x) for x in xs) # short-circuits, unlike a list comp

  {cyan("__slots__")} for millions of small objects (Lesson 20).

  {cyan("Use sets for dedup and membership.")}
  {cyan("Use dict.get(k, default) instead of in-checks + lookup.")}""")

player.explain("When Python isn't fast enough", f"""\
  {cyan("NumPy / Pandas")} — for numeric/array/table work. Vectorized
  ops run in C; a loop rewritten as a NumPy expression is often 50-500x.

  {cyan("Cython / C extensions")} — rewrite the hot function in typed
  Cython or C, keep the rest in Python. The  profile says what to rewrite.

  {cyan("PyPy")} — alternative Python interpreter with JIT. Sometimes 10x
  faster for pure-Python workloads. Drawback: some C extensions lag.

  {cyan("Rust (via pyo3 / maturin)")} — modern choice for critical-path
  libraries (ruff, pydantic v2, polars). Seamless Python bindings.

  {cyan("Don't start here.")} Most "Python is slow" turns out to be
  accidentally-quadratic code, a missing index, or chatty I/O.""")

player.play()
