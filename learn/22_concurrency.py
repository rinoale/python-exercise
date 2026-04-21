"""Lesson 22: Concurrency — threads, processes, async"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from quiz_player import QuizPlayer, check_eval, check_code, cyan

player = QuizPlayer("Lesson 22: Concurrency")

player.explain("Three models in Python", f"""\
  Choose based on what bottlenecks you:

  {cyan("Threads")}         — multiple OS threads, shared memory.
                    Good for:  I/O-bound work (API calls, disk, DB).
                    Bad for:   CPU-bound work (the GIL serializes bytecode).

  {cyan("Processes")}       — separate Python interpreters, no shared state.
                    Good for:  CPU-bound work (true parallelism).
                    Cost:      slow startup, IPC overhead to pass data.

  {cyan("asyncio")}         — one thread, cooperative multitasking via await.
                    Good for:  thousands of concurrent I/O operations.
                    Cost:      everything you call must be async-aware.""")

player.explain("The GIL — what it really blocks", f"""\
  The {cyan("Global Interpreter Lock")} means only ONE thread executes
  Python bytecode at a time. So adding threads doesn't speed up pure
  Python CPU work.

  But the GIL is RELEASED during:
    - I/O system calls (read, write, send, recv)
    - C extension calls (NumPy, Pillow, etc., when well-written)
    - time.sleep()

  So threads DO help for:
    - HTTP requests (waiting on network)
    - File I/O (waiting on disk)
    - Database queries
    - Calling NumPy on large arrays

  {cyan("Note (3.13+):")} a "no-GIL" build exists but is experimental and
  not the default. Assume GIL semantics for now.""")

player.explain("concurrent.futures — the easiest pool API", f"""\
      from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

      def fetch(url): ...

      with ThreadPoolExecutor(max_workers=10) as pool:
          results = list(pool.map(fetch, urls))     # parallel I/O

      with ProcessPoolExecutor() as pool:
          squares = list(pool.map(cpu_heavy, numbers))   # true parallel

  {cyan(".submit()")} returns a Future you can compose:
      futures = [pool.submit(fetch, u) for u in urls]
      for fut in as_completed(futures):
          result = fut.result()                    # first-done wins

  {cyan("Defaults:")} ThreadPoolExecutor() uses min(32, cpu_count() + 4).
  ProcessPoolExecutor() uses cpu_count().""")

player.explain("asyncio — async/await", f"""\
      import asyncio
      import aiohttp

      async def fetch(session, url):
          async with session.get(url) as resp:
              return await resp.text()

      async def main():
          async with aiohttp.ClientSession() as s:
              results = await asyncio.gather(*[fetch(s, u) for u in urls])

      asyncio.run(main())

  {cyan("Mental model:")} await gives up control until the awaited thing
  is ready. The event loop runs OTHER tasks in the meantime. No threads,
  no race conditions, but everything in your stack must be async-aware
  (e.g., use  aiohttp, not  requests).

  {cyan("Key primitives:")}
      asyncio.run(coro)                run an async entry point
      asyncio.gather(c1, c2, ...)      await many at once
      asyncio.create_task(coro)         schedule a coroutine as a task
      asyncio.sleep(seconds)           non-blocking sleep
      async for x in async_iter: ...
      async with ctx: ...""")

player.quiz(
    "Define an async function  wait_then(value)  that  await asyncio.sleep(0)\n"
    "  and returns value.\n"
    "  Also, set  result  at module level to the output of running\n"
    "  asyncio.run(wait_then(42)).  Expected: result == 42.",
    check_code(lambda ns: (
        ns.get("result") == 42,
        "async function runs and returns 42",
    )),
    hint="import asyncio; async def wait_then(v): await asyncio.sleep(0); return v\\nresult = asyncio.run(wait_then(42))",
    multiline=True,
)

player.explain("Picking the right tool — a cheat sheet", f"""\
  Task                             Best tool
  ──────────────────────────────   ─────────────────────────────
  1 HTTP request                   requests / urllib (just run it)
  100 HTTP requests                ThreadPoolExecutor  or  asyncio
  10,000 HTTP requests             asyncio  (threads run out of RAM)
  CPU-heavy loop (pure Python)     ProcessPoolExecutor
  CPU-heavy NumPy                  Already parallel in C; don't bother
  Shared state between tasks       Threads with Lock, or asyncio (single-thread)
  Simple fan-out / fan-in          ThreadPoolExecutor + pool.map

  {cyan("Don't start with concurrency.")} Profile first — often the bottleneck
  is a bad algorithm, an N+1 query, or a missing cache, not serial code.""")

player.play()
