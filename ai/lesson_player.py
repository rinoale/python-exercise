# Reusable step-by-step lesson player
# Records steps, then lets you navigate with arrow keys
#
# Usage:
#   from lesson_player import LessonPlayer
#   player = LessonPlayer("Lesson Title")
#   player.add_step("Step Title", "content here")
#   player.play()

import sys
import tty
import termios


# ANSI color helpers
def bold(text):
    return f"\033[1m{text}\033[0m"


def yellow(text):
    return f"\033[33m{text}\033[0m"


def cyan(text):
    return f"\033[36m{text}\033[0m"


def green(text):
    return f"\033[32m{text}\033[0m"


def gray(text):
    return f"\033[90m{text}\033[0m"


def dim(text):
    return f"\033[2m{text}\033[0m"


class LessonPlayer:
    def __init__(self, title=""):
        self.title = title
        self.steps = []

    def add_step(self, title, content, detail=None):
        self.steps.append({"title": title, "content": content, "detail": detail})

    def play(self):
        if not self.steps:
            return
        if not sys.stdin.isatty():
            self._print_all()
            return
        idx = 0
        show_detail = False
        self._render(idx, show_detail)
        while True:
            key = self._read_key()
            if key == "right":
                idx = min(idx + 1, len(self.steps) - 1)
                show_detail = False
            elif key == "left":
                idx = max(idx - 1, 0)
                show_detail = False
            elif key == "home":
                idx = 0
                show_detail = False
            elif key == "end":
                idx = len(self.steps) - 1
                show_detail = False
            elif key == "detail":
                if self.steps[idx]["detail"]:
                    show_detail = not show_detail
            elif key == "quit":
                print("\033[0m")
                break
            self._render(idx, show_detail)

    def _print_all(self):
        """Fallback: print all steps at once when not in a terminal."""
        for i, step in enumerate(self.steps):
            print(bold(self.title))
            print(yellow(step["title"]))
            print()
            print(step["content"])
            if step["detail"]:
                print()
                print(step["detail"])
            print()
            print(gray(f"  [{i + 1}/{len(self.steps)}]"))
            print()
            print("-" * 60)
            print()

    def _render(self, idx, show_detail=False):
        print("\033[2J\033[H", end="")  # clear screen, cursor to top
        step = self.steps[idx]
        print(bold(self.title))
        print(yellow(step["title"]))
        print()
        print(step["content"])
        if step["detail"]:
            if show_detail:
                print()
                print(step["detail"])
            else:
                print()
                print(cyan("  [d] show more"))
        print()
        nav = f"  [{idx + 1}/{len(self.steps)}] \u2190 \u2192 navigate"
        if step["detail"]:
            nav += " | d detail"
        nav += " | Home/End jump | q quit"
        print(gray(nav))

    def _read_key(self):
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
            if ch == "\x1b":
                seq = sys.stdin.read(2)
                if seq == "[D":
                    return "left"
                if seq == "[C":
                    return "right"
                if seq == "[H":
                    return "home"
                if seq == "[F":
                    return "end"
                return "unknown"
            elif ch in ("q", "\x03"):  # q or Ctrl-C
                return "quit"
            elif ch == " ":
                return "right"
            elif ch == "d":
                return "detail"
            return "unknown"
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
