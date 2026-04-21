"""Lesson 15: Context Managers"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from quiz_player import QuizPlayer, check_eval, check_code, cyan

player = QuizPlayer("Lesson 15: Context Managers")

player.explain("The with statement — guaranteed cleanup", f"""\
  The idiom for acquiring a resource and releasing it reliably:

      with open("file.txt") as f:
          content = f.read()
      # f is closed here — even if .read() raised

  Equivalent to:
      f = open("file.txt")
      try:
          content = f.read()
      finally:
          f.close()

  {cyan("Built-in examples:")}
      with open(path): ...                      # file — auto-close
      with threading.Lock(): ...                 # lock — auto-release
      with tempfile.TemporaryDirectory(): ...    # temp dir — auto-delete""")

player.explain("Multiple contexts in one with", f"""\
      with open("in.txt") as fin, open("out.txt", "w") as fout:
          fout.write(fin.read().upper())

  Or across lines (3.10+ parenthesized form):
      with (
          open("a") as a,
          open("b") as b,
          open("c") as c,
      ):
          ...""")

player.explain("Writing one — class form with __enter__ / __exit__", f"""\
      class Timer:
          def __enter__(self):
              import time
              self.start = time.time()
              return self                    # bound to `as x`

          def __exit__(self, exc_type, exc, tb):
              self.elapsed = time.time() - self.start
              # Return True to SUPPRESS exceptions; False/None to propagate.
              # Usually return False (or nothing).

      with Timer() as t:
          do_work()
      print(f"took {{t.elapsed}}s")

  {cyan("__exit__ signature:")} (exc_type, exc_value, traceback).
  All three are None if the block finished normally.""")

player.quiz(
    "Define a class  Counter  used as a context manager.\n"
    "  Usage:\n"
    "      with Counter() as c:\n"
    "          c.tick()\n"
    "          c.tick()\n"
    "      c.count  should be 2 AFTER the with block.\n"
    "  Also:  c.entered is True and c.exited is True.",
    check_code(lambda ns: _test_counter(ns)),
    hint="__enter__ returns self; __exit__ sets exited = True",
    multiline=True,
)


def _test_counter(ns):
    Counter = ns["Counter"]
    with Counter() as c:
        c.tick()
        c.tick()
    ok = (c.count == 2 and getattr(c, "entered", False) and getattr(c, "exited", False))
    return ok, ("works" if ok else f"count={c.count} entered={getattr(c,'entered',None)} exited={getattr(c,'exited',None)}")


player.explain("contextlib — the easy way", f"""\
  For a context manager that's really just setup + teardown around a
  block, use {cyan("@contextmanager")}. Write it as a generator: one yield.

      from contextlib import contextmanager

      @contextmanager
      def cd(path):
          old = os.getcwd()
          os.chdir(path)
          try:
              yield                      # the `with` body runs here
          finally:
              os.chdir(old)              # always restore, even on exceptions

      with cd("/tmp"):
          do_stuff()                      # now inside /tmp
      # back in the original directory

  The {cyan("try/finally")} is load-bearing — without it, an exception in
  the block skips the cleanup.""")

player.quiz(
    "Use  @contextmanager  to write  scoped_value(obj, attr, new).\n"
    "  It temporarily sets  obj.attr = new  and restores the old value on exit.\n"
    "  Yield nothing.\n"
    "  Setup:  obj = type('O', (), {})(); obj.x = 1.\n"
    "  After:  with scoped_value(obj, 'x', 99):  assert obj.x == 99\n"
    "  Outside: obj.x == 1 again.",
    check_code(lambda ns: _test_scoped(ns)),
    hint="save old, set new, try: yield, finally: restore",
    multiline=True,
)


def _test_scoped(ns):
    scoped = ns["scoped_value"]
    obj = type("O", (), {})()
    obj.x = 1
    with scoped(obj, "x", 99):
        if obj.x != 99:
            return False, f"inside: obj.x = {obj.x}"
    if obj.x != 1:
        return False, f"after: obj.x = {obj.x} (not restored)"
    # also verify restoration even if exception raised
    try:
        with scoped(obj, "x", 7):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    if obj.x != 1:
        return False, "not restored after exception"
    return True, "works, even across exceptions"


player.explain("contextlib.suppress — silence specific exceptions", f"""\
      from contextlib import suppress

      with suppress(FileNotFoundError):
          os.remove("maybe_exists.txt")
      # removed if it was there; no error if it wasn't

  Cleaner than  try / except FileNotFoundError: pass.""")

player.play()
