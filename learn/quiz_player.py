"""Interactive quiz-style lesson player.

A lesson builds up steps by calling .explain() and .quiz(), then .play().
Quizzes ask the user for a Python expression; check_eval/check_function
evaluate the answer and provide feedback.

    from quiz_player import QuizPlayer, check_eval
    p = QuizPlayer("Title")
    p.explain("Intro", "...")
    p.quiz("What is 2 + 2?", check_eval(4))
    p.play()
"""
import sys


def bold(t): return f"\033[1m{t}\033[0m"
def cyan(t): return f"\033[36m{t}\033[0m"
def green(t): return f"\033[32m{t}\033[0m"
def red(t): return f"\033[31m{t}\033[0m"
def yellow(t): return f"\033[33m{t}\033[0m"
def gray(t): return f"\033[90m{t}\033[0m"


class QuizPlayer:
    def __init__(self, title):
        self.title = title
        self.items = []
        self.score = 0
        self.total = 0

    def explain(self, title, content):
        self.items.append(("explain", title, content))

    def quiz(self, prompt, check, hint=None, multiline=False):
        self.items.append(("quiz", prompt, check, hint, multiline))

    def _safe_input(self, prompt):
        try:
            return input(prompt)
        except (EOFError, KeyboardInterrupt):
            print()
            print(gray("  bye!"))
            sys.exit(0)

    def _read_line(self, hint):
        while True:
            ans = self._safe_input(cyan("> "))
            cmd = ans.strip().lower()
            if cmd == "?" and hint:
                print(gray(f"  hint: {hint}"))
                continue
            if cmd in ("s", "skip"):
                return None
            if cmd == "":
                continue
            return ans

    def _read_multiline(self, hint):
        print(gray("  (multi-line — '.' on its own line submits; Ctrl-D too; ? hint, s skip)"))
        lines = []
        while True:
            try:
                line = input(cyan("... " if lines else "> "))
            except KeyboardInterrupt:
                print()
                sys.exit(0)
            except EOFError:
                print()
                return "\n".join(lines) if lines else None
            stripped = line.strip()
            if not lines:
                if stripped == "?" and hint:
                    print(gray(f"  hint: {hint}"))
                    continue
                if stripped in ("s", "skip"):
                    return None
                if stripped == "":
                    continue
            if stripped == ".":
                return "\n".join(lines)
            lines.append(line)

    def _do_quiz(self, prompt, check, hint, multiline):
        self.total += 1
        print(yellow(f"Quiz {self.total}:"))
        print(prompt)
        tail = "  (? hint, s skip)" if hint else "  (s skip)"
        print(gray(tail))
        while True:
            ans = self._read_multiline(hint) if multiline else self._read_line(hint)
            if ans is None:
                print(gray("  skipped"))
                return
            ok, feedback = check(ans)
            if ok:
                msg = f"  ✓ {feedback}" if feedback else "  ✓ correct"
                print(green(msg))
                self.score += 1
                return
            print(red(f"  ✗ {feedback}"))
            again = self._safe_input(gray("  retry? [Y/n] ")).strip().lower()
            if again in ("n", "no"):
                return

    def play(self):
        print(bold(self.title))
        print("─" * max(len(self.title), 20))
        print()
        for item in self.items:
            if item[0] == "explain":
                _, title, content = item
                print(yellow(title))
                print(content)
                print()
                self._safe_input(gray("  [Enter to continue]"))
                print()
            else:
                _, prompt, check, hint, multiline = item
                self._do_quiz(prompt, check, hint, multiline)
                print()
        if self.total == 0:
            return
        print("─" * 40)
        pct = 100 * self.score // self.total
        line = f"Final score: {self.score}/{self.total}  ({pct}%)"
        color = green if self.score == self.total else (yellow if pct >= 60 else red)
        print(bold(color(line)))


# ── Check factories ────────────────────────────────────────


def _format_error(e):
    return f"{e.__class__.__name__}: {e}"


def check_eval(expected, strict_type=True):
    """User types an expression; eval and compare with expected.
    strict_type=True also compares the type (so 1 != True, 1 != 1.0).
    """
    def check(s):
        try:
            result = eval(s)
        except Exception as e:
            return False, _format_error(e)
        if strict_type and type(result) is not type(expected):
            return False, (f"got {result!r} (type {type(result).__name__}); "
                           f"expected {expected!r} (type {type(expected).__name__})")
        if result != expected:
            return False, f"got {result!r}; expected {expected!r}"
        return True, f"got {result!r}"
    return check


def check_predicate(pred, description):
    """pred: callable(eval_result) -> bool. description: shown on failure."""
    def check(s):
        try:
            result = eval(s)
        except Exception as e:
            return False, _format_error(e)
        if pred(result):
            return True, f"got {result!r}"
        return False, f"got {result!r} — expected {description}"
    return check


def check_literal(expected):
    """Exact string match (after strip)."""
    def check(s):
        if s.strip() == expected.strip():
            return True, ""
        return False, f"expected exactly: {expected}"
    return check


def check_function(name, tests, must_use=None, must_not_use=None):
    """User submits code defining `name`; we run it against tests.

    tests: list of either
        (args_tuple, expected_return)                 — no kwargs
        (args_tuple, kwargs_dict, expected_return)    — with kwargs
    must_use / must_not_use: optional substrings the code must or must not contain.
    """
    def check(code):
        if must_use and must_use not in code:
            return False, f"your code should use `{must_use}`"
        if must_not_use and must_not_use in code:
            return False, f"your code should not use `{must_not_use}`"
        ns = {}
        try:
            exec(code, ns)
        except Exception as e:
            return False, _format_error(e)
        if name not in ns:
            return False, f"no function `{name}` defined"
        fn = ns[name]
        if not callable(fn):
            return False, f"`{name}` is not callable"
        for test in tests:
            if len(test) == 3:
                args, kwargs, expected = test
            else:
                args, expected = test
                kwargs = {}
            try:
                result = fn(*args, **kwargs)
            except Exception as e:
                call_str = f"{name}{args}"
                if kwargs:
                    call_str += f" with {kwargs}"
                return False, f"{call_str} raised {_format_error(e)}"
            if result != expected:
                call_str = f"{name}{args}"
                if kwargs:
                    call_str += f" with {kwargs}"
                return False, f"{call_str} → {result!r}, expected {expected!r}"
        return True, f"all {len(tests)} test(s) pass"
    return check


def check_code(predicate):
    """User submits code; we exec it, then call predicate(namespace).

    predicate(ns) must return (ok: bool, feedback: str).
    Useful for class definitions, multi-step answers, etc.
    """
    def check(code):
        ns = {}
        try:
            exec(code, ns)
        except Exception as e:
            return False, _format_error(e)
        try:
            return predicate(ns)
        except Exception as e:
            return False, f"test raised {_format_error(e)}"
    return check
