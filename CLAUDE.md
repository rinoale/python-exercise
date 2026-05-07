# CLAUDE.md

## Creating Jupyter Notebooks (.ipynb)

- Write the entire notebook as a single `Write` call with complete JSON, not cell-by-cell with `NotebookEdit`.
- `NotebookEdit` inserts one cell at a time, which costs ~10x more tokens. Only use it for small edits to existing notebooks.
- Structure: alternate markdown cells (explanations) and code cells (runnable examples).
- Code output should be text-based (print statements, ASCII visualization). Avoid matplotlib unless explicitly requested.
- Each notebook should be self-contained: imports at the top, no dependencies on other notebooks.
- NEVER delegate notebook (.ipynb) creation to subagents. Always write notebooks directly in the main conversation.
- Write only one file at a time. After finishing a file, ask the user whether to proceed to the next one before continuing.

## Creating Terminal Animations (.py)

### Rendering
- Use ONLY plain ASCII characters. No Unicode box-drawing (║, ═, ╔, ╲, ▶, ╮, ╞, etc.) — they have unpredictable display widths across terminals and cause misaligned output.
- Use `\r\n` (not `\n`) for line endings when in raw terminal mode. In raw mode, `\n` only moves down without returning to column 0, creating a staircase effect.
- Lines are plain indented text with `  ` prefix. Separators are `----` dashes. No border characters.
- Clear screen with ANSI escape `\033[2J\033[H`.

### Terminal Input (arrow keys, interactive controls)
- Enter raw mode once at startup with `tty.setraw()`, restore on exit with `termios.tcsetattr()`. Do NOT toggle raw mode per keypress — it causes race conditions reading multi-byte escape sequences.
- Read keys with `os.read(fd, 1)` (not `sys.stdin.read(1)`).
- Arrow keys send 3 bytes: `\x1b`, `[`, `A/B/C/D`. After reading `\x1b`, use `select.select()` with 0.1s timeout to read the remaining bytes. Shorter timeouts (0.05s) can miss bytes and break arrow detection.

### Slide-based Architecture
- Pre-compute ALL slides into a list at startup. This makes backward navigation instant and decouples computation from rendering.
- Each slide is a list of strings (lines to print). Store as `[(epoch, slide_index, lines), ...]`.
- The render function replaces placeholder lines (e.g., mode status) dynamically at display time.

### Controls
- SPACE toggles auto-play on/off. Arrow keys pause auto-play and navigate manually.
- Auto-play mode: `get_key(timeout=speed)` — returns None on timeout to advance, or the key if pressed.
- Manual mode: `get_key(timeout=None)` — blocks until a key is pressed.
- Show current mode clearly: `>> AUTO-PLAY (SPACE=pause, arrows=browse)` vs `== PAUSED (SPACE=play, arrows=browse)`.

### Content Structure: Overview + Detail
- Animated auto-play shows the big picture: how the entire process flows across many epochs, with speed accelerating over time (e.g., 2.5s -> 0.02s per slide).
- Each slide explains one specific step with full math, showing every intermediate value. The user can pause and arrow back/forward to compare numbers between steps.
- This combination lets the user first see the forest (auto-play), then examine individual trees (manual navigation).
