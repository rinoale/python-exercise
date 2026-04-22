"""Lesson 25: Packaging and Tooling"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from quiz_player import QuizPlayer, check_eval, check_literal, cyan

player = QuizPlayer("Lesson 25: Packaging and Tooling")

player.explain("Virtual environments — isolate your project's deps", f"""\
  {cyan("Why?")} System Python is shared. Two projects can need different
  versions of the same library. Solution: a per-project env.

      python -m venv .venv              # create one
      source .venv/bin/activate         # Linux/macOS
      .venv\\Scripts\\activate           # Windows
      pip install requests               # now isolated
      deactivate                         # exit the env

  {cyan("Alternatives:")}
      pyenv    — manage Python VERSIONS (3.11 vs 3.12 vs 3.13 ...)
      pipx     — install Python CLIs in isolated envs (black, ruff, poetry)
      uv       — Rust-based, drop-in pip replacement — very fast (2024~)
      poetry   — combines virtualenv + lock file + build tool
      conda    — popular in data science; manages non-Python deps too""")

player.explain("pyproject.toml — modern project config", f"""\
  A single file replaces  setup.py, setup.cfg, requirements.txt, and more:

      [project]
      name = "myapp"
      version = "0.1.0"
      description = "my cool app"
      requires-python = ">=3.11"
      dependencies = [
          "requests>=2.31",
          "pydantic",
      ]

      [project.optional-dependencies]
      dev = ["pytest", "mypy", "ruff"]

      [project.scripts]
      myapp = "myapp.cli:main"             # creates  myapp  executable

      [build-system]
      requires = ["hatchling"]
      build-backend = "hatchling.build"

  Install YOUR project as a package:
      pip install -e .                     # editable: edits reflect live
      pip install ".[dev]"                 # with dev extras""")

player.quiz(
    "Type an expression that returns the string 'pyproject.toml' — the\n"
    "  modern single-file config that replaces setup.py.",
    check_eval("pyproject.toml"),
    hint='"pyproject.toml"  (just the string literal)',
)

player.explain("Linters and formatters", f"""\
  {cyan("ruff")} — Rust-based, extremely fast linter AND formatter
              (essentially covers Flake8 + isort + Black + pyupgrade)
      ruff check .                       # lint
      ruff check --fix .                 # auto-fix
      ruff format .                      # format

  {cyan("black")} — opinionated formatter; zero config
      black .

  {cyan("mypy  /  pyright")} — static type checkers
      mypy src/
      pyright src/                       # often faster, stricter

  {cyan("Recommended modern stack:")}  ruff (lint+format) + mypy or pyright.
  Set them up once in pyproject.toml, run in pre-commit / CI.""")

player.explain("pytest — the standard testing framework", f"""\
      # test_math.py
      def test_add():
          assert 1 + 1 == 2

      def test_raises():
          import pytest
          with pytest.raises(ValueError):
              int("not a number")

      @pytest.mark.parametrize("a,b,expected", [
          (1, 1, 2),
          (2, 3, 5),
          (0, 0, 0),
      ])
      def test_add_many(a, b, expected):
          assert a + b == expected

  Run:
      pytest                              # all tests
      pytest tests/                       # a directory
      pytest -k add                       # tests matching "add"
      pytest -x                           # stop at first failure
      pytest -v                           # verbose
      pytest --cov=myapp                  # coverage (needs pytest-cov)

  {cyan("Fixtures")} for reusable setup:
      @pytest.fixture
      def db():
          conn = connect()
          yield conn                      # what the test gets
          conn.close()                    # cleanup after the test""")

player.explain("Pre-commit — run checks before every git commit", f"""\
  Install once, catch problems before they reach CI.

      # .pre-commit-config.yaml
      repos:
        - repo: https://github.com/astral-sh/ruff-pre-commit
          rev: v0.5.0
          hooks:
            - id: ruff
              args: [--fix]
            - id: ruff-format

      pip install pre-commit
      pre-commit install                  # sets up git hooks

  Now  git commit  runs ruff and blocks if there are issues.""")

player.explain("Publishing a package", f"""\
  Once pyproject.toml is set up:
      pip install build twine
      python -m build                     # creates dist/*.tar.gz and *.whl
      twine upload dist/*                 # upload to PyPI

  Use  https://test.pypi.org  first to rehearse. Tag releases with git
  and keep a CHANGELOG.md. {cyan("Most projects use GitHub Actions to")}
  {cyan("auto-publish on tag push.")}""")

player.play()
