"""Lesson 11: Exceptions"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from quiz_player import QuizPlayer, check_eval, check_code, cyan

player = QuizPlayer("Lesson 11: Exceptions")

player.explain("try / except / else / finally", f"""\
      try:
          result = risky()
      except ValueError as e:
          print(f"bad value: {{e}}")
      except (KeyError, IndexError):
          print("lookup failed")
      else:
          print("no exception raised")       # runs only if try succeeded
      finally:
          cleanup()                          # always runs

  {cyan("Catch narrowly.")} Bare  except:  catches even KeyboardInterrupt
  and SystemExit — almost never what you want. Catch the specific
  exception, or at least  except Exception.""")

player.quiz(
    "Fill in a function  safe_div(a, b)  that returns  a / b,\n"
    "  but returns the string \"infinity\" instead of raising on b == 0.",
    check_code(lambda ns: (
        ns["safe_div"](10, 2) == 5.0 and ns["safe_div"](1, 0) == "infinity" and ns["safe_div"](-6, 3) == -2.0,
        "safe_div handles zero denominator",
    )),
    hint="try: return a/b\\nexcept ZeroDivisionError: return 'infinity'",
    multiline=True,
)

player.explain("raise — signal an error yourself", f"""\
      def withdraw(balance, amount):
          if amount > balance:
              raise ValueError(f"insufficient funds: {{balance}} < {{amount}}")
          return balance - amount

  {cyan("Built-in exception types to know:")}
      ValueError       bad argument value      int("abc")
      TypeError        wrong type              1 + "x"
      KeyError         missing dict key        d["missing"]
      IndexError       list index out of range [1,2][5]
      AttributeError   missing attribute       obj.missing
      ZeroDivisionError                        1 / 0
      FileNotFoundError                        open("nope")
      StopIteration    generator exhausted     next(empty)
      RuntimeError     generic runtime problem""")

player.quiz(
    "Define  withdraw(balance, amount)  that returns the new balance if\n"
    "  there is enough, else  raises ValueError.  Test:\n"
    "    withdraw(100, 30) → 70\n"
    "    withdraw(50, 100) → ValueError",
    check_code(lambda ns: (
        _test_withdraw(ns["withdraw"]),
        "withdraw behaves correctly",
    )),
    hint="if amount > balance: raise ValueError(...)",
    multiline=True,
)


def _test_withdraw(fn):
    if fn(100, 30) != 70 or fn(50, 50) != 0:
        return False
    try:
        fn(50, 100)
    except ValueError:
        return True
    return False


player.explain("Custom exceptions", f"""\
  Define your own by subclassing Exception:

      class InsufficientFunds(Exception):
          pass

      class BankError(Exception):
          def __init__(self, code, msg):
              super().__init__(msg)
              self.code = code

  {cyan("Catch with  except InsufficientFunds:  — same syntax as built-ins.")}

  {cyan("Group related errors under a base class:")}
      class AppError(Exception): pass
      class NotFoundError(AppError): pass
      class AuthError(AppError): pass
      # Now  except AppError:  catches either subclass.""")

player.quiz(
    "Define a custom exception  InsufficientFunds  (subclass of Exception).\n"
    "  Then define  withdraw2(balance, amount)  that raises it (not ValueError)\n"
    "  when amount > balance.",
    check_code(lambda ns: (
        _test_custom_exc(ns),
        "custom exception raised",
    )),
    hint="class InsufficientFunds(Exception): pass",
    multiline=True,
)


def _test_custom_exc(ns):
    cls = ns.get("InsufficientFunds")
    if not (cls and issubclass(cls, Exception)):
        return False
    try:
        ns["withdraw2"](10, 20)
    except Exception as e:
        return isinstance(e, cls)
    return False


player.explain("Exception chaining — raise X from Y", f"""\
  When re-raising a different exception type, preserve the original cause:

      try:
          data = json.loads(text)
      except json.JSONDecodeError as e:
          raise ConfigError("bad config file") from e

  The traceback shows BOTH exceptions — the root cause isn't lost.

  {cyan("EAFP vs LBYL — the Pythonic choice:")}
      LBYL (Look Before You Leap):
          if key in d: return d[key]
      EAFP (Easier to Ask Forgiveness than Permission):
          try: return d[key]
          except KeyError: return default

  Python style prefers EAFP for most cases — it's usually faster and
  avoids race conditions (e.g., file existing then disappearing).""")

player.play()
