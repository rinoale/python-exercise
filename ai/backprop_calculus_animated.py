"""
Backpropagation Calculus -- Step-by-step animation.

2 inputs (x1, x2) -> 2 hidden neurons (h1, h2) -> 1 output
Total: 9 parameters (6 weights + 3 biases)

Controls:
  Left/Right arrow   Navigate slides
  SPACE              Play / Pause
  q                  Quit

Run:  python ai/backprop_calculus_animated.py
"""
import os
import sys
import math
import random
import termios
import tty
import select

random.seed(42)

W_INIT = {
    "w1": random.uniform(-1, 1),
    "w2": random.uniform(-1, 1),
    "w3": random.uniform(-1, 1),
    "w4": random.uniform(-1, 1),
    "w5": random.uniform(-1, 1),
    "w6": random.uniform(-1, 1),
    "b1": random.uniform(-0.5, 0.5),
    "b2": random.uniform(-0.5, 0.5),
    "b3": random.uniform(-0.5, 0.5),
}

LR = 2.0

DATA = [
    ([0.0, 0.0], 0.0),
    ([0.0, 1.0], 1.0),
    ([1.0, 0.0], 1.0),
    ([1.0, 1.0], 0.0),
]


def sigmoid(x):
    x = max(-500, min(500, x))
    return 1.0 / (1.0 + math.exp(-x))


def forward(w, x1, x2):
    z1 = w["w1"] * x1 + w["w3"] * x2 + w["b1"]
    h1 = sigmoid(z1)
    z2 = w["w2"] * x1 + w["w4"] * x2 + w["b2"]
    h2 = sigmoid(z2)
    z3 = w["w5"] * h1 + w["w6"] * h2 + w["b3"]
    out = sigmoid(z3)
    return {"x1": x1, "x2": x2, "z1": z1, "h1": h1, "z2": z2, "h2": h2,
            "z3": z3, "out": out}


def total_loss(w):
    return sum(0.5 * (t - forward(w, x[0], x[1])["out"]) ** 2
               for x, t in DATA)


# ==================================================================
# RENDERING -- plain text lines, no box characters
# ==================================================================

FRAME = []


def L(text=""):
    FRAME.append("  " + text)


def SEP():
    FRAME.append("  " + "-" * 68)


def gradient_bar(value, label, width=20):
    center = width // 2
    bar = list("-" * width)
    bar[center] = "+"
    mag = min(abs(value) * 8, center - 1)
    if value > 0.001:
        for i in range(center + 1, center + 1 + int(mag)):
            if i < width:
                bar[i] = ">"
    elif value < -0.001:
        for i in range(center - 1, center - 1 - int(mag), -1):
            if i >= 0:
                bar[i] = "<"
    return f"{label:<8}{''.join(bar)} {value:+.6f}"


def show_network(w, v):
    L()
    L(f"  x1=[{v['x1']:.0f}] ---w1={w['w1']:+.3f}---> [H1]={v['h1']:.3f} ---w5={w['w5']:+.3f}---\\")
    L(f"      \\                   b1={w['b1']:+.3f}                      \\")
    L(f"       \\ w3={w['w3']:+.3f}                                       >-- [OUT]={v['out']:.3f}")
    L(f"        X                                                  /    b3={w['b3']:+.3f}")
    L(f"       / w2={w['w2']:+.3f}                                       /")
    L(f"      /                   b2={w['b2']:+.3f}                      /")
    L(f"  x2=[{v['x2']:.0f}] ---w4={w['w4']:+.3f}---> [H2]={v['h2']:.3f} ---w6={w['w6']:+.3f}---/")
    L()


def show_header(epoch, step_name, t_loss):
    FRAME.append("")
    L(f"BACKPROPAGATION  (mode-placeholder)")
    SEP()
    L(f"Step: {step_name}    Loss: {t_loss:.6f}")
    SEP()


def show_nav():
    SEP()
    L("  <-/->  prev/next    SPACE  play/pause    q  quit")
    FRAME.append("")


# ==================================================================
# BUILD SLIDES
# ==================================================================

def build_epoch_slides(epoch, w):
    slides = []
    x, target = DATA[epoch % 4]
    x1, x2 = x
    v = forward(w, x1, x2)
    t_loss = total_loss(w)

    def header(step):
        show_header(epoch, step, t_loss)

    # --- Slide 1: FORWARD PASS ---
    FRAME.clear()
    header("1/5  FORWARD PASS")
    show_network(w, v)
    SEP()
    L(f"Input: [{x1:.0f}, {x2:.0f}]    Target: {target:.0f}")
    L()
    L(f"Hidden neuron H1:                Hidden neuron H2:")
    L(f"  z1 = w1*x1 + w3*x2 + b1         z2 = w2*x1 + w4*x2 + b2")
    z1e = f"{w['w1']:+.3f}*{x1:.0f} + {w['w3']:+.3f}*{x2:.0f} + {w['b1']:+.3f}"
    z2e = f"{w['w2']:+.3f}*{x1:.0f} + {w['w4']:+.3f}*{x2:.0f} + {w['b2']:+.3f}"
    L(f"     = {z1e}")
    L(f"     = {v['z1']:+.4f}")
    L(f"  h1 = sigmoid({v['z1']:+.4f}) = {v['h1']:.4f}")
    L()
    L(f"     = {z2e}")
    L(f"     = {v['z2']:+.4f}")
    L(f"  h2 = sigmoid({v['z2']:+.4f}) = {v['h2']:.4f}")
    L()
    L(f"Output neuron:")
    L(f"  z3 = w5*h1 + w6*h2 + b3")
    L(f"     = {w['w5']:+.3f}*{v['h1']:.3f} + {w['w6']:+.3f}*{v['h2']:.3f} + {w['b3']:+.3f}")
    L(f"     = {v['z3']:+.4f}")
    L(f"  out = sigmoid({v['z3']:+.4f}) = {v['out']:.4f}")
    show_nav()
    slides.append(list(FRAME))

    # --- Slide 2: COMPUTE LOSS ---
    loss = 0.5 * (target - v["out"]) ** 2
    error = v["out"] - target

    FRAME.clear()
    header("2/5  COMPUTE LOSS")
    show_network(w, v)
    SEP()
    L(f"Target: {target:.0f}    Output: {v['out']:.4f}")
    L()
    L(f"  Loss = 0.5 * (target - out)^2")
    L(f"       = 0.5 * ({target:.0f} - {v['out']:.4f})^2")
    L(f"       = {loss:.6f}")
    L()
    L(f"  error = out - target = {error:+.4f}")
    if abs(error) > 0.01:
        direction = "too HIGH -> push DOWN" if error > 0 else "too LOW -> push UP"
        L(f"  Output is {direction}")
    else:
        L(f"  Very close! Error is only {error:+.4f}")
    L()
    L("  Question: which of the 9 weights caused this, and how much?")
    L("  Chain rule traces blame backward through the network.")
    show_nav()
    slides.append(list(FRAME))

    # --- Slide 3: BACKWARD -- output layer ---
    dL_dout = error
    dout_dz3 = v["out"] * (1 - v["out"])
    dL_dz3 = dL_dout * dout_dz3

    dL_dw5 = dL_dz3 * v["h1"]
    dL_dw6 = dL_dz3 * v["h2"]
    dL_db3 = dL_dz3

    FRAME.clear()
    header("3/5  BACKWARD -- output layer")
    show_network(w, v)
    SEP()
    L("Chain:  Loss <- out <- z3 <- {w5, w6, b3}")
    L()
    L(f"  dLoss/dout = error = {dL_dout:+.4f}")
    L()
    L(f"  dout/dz3 = sigmoid'(z3) = out*(1-out)")
    L(f"           = {v['out']:.4f} * {1-v['out']:.4f} = {dout_dz3:.4f}")
    L()
    L(f"  dLoss/dz3 = dLoss/dout * dout/dz3       << CHAIN RULE")
    L(f"            = {dL_dout:+.4f} * {dout_dz3:.4f} = {dL_dz3:+.6f}")
    L()
    L(f"  dz3/dw5 = h1 = {v['h1']:.4f}")
    L(f"  dz3/dw6 = h2 = {v['h2']:.4f}")
    L(f"  dz3/db3 = 1")
    L()
    L(f"  dL/dw5 = {dL_dz3:+.6f} * {v['h1']:.4f} = {dL_dw5:+.6f}")
    L(f"  dL/dw6 = {dL_dz3:+.6f} * {v['h2']:.4f} = {dL_dw6:+.6f}")
    L(f"  dL/db3 = {dL_dz3:+.6f} * 1       = {dL_db3:+.6f}")
    show_nav()
    slides.append(list(FRAME))

    # --- Slide 4: BACKWARD -- hidden layer ---
    dL_dh1 = dL_dz3 * w["w5"]
    dh1_dz1 = v["h1"] * (1 - v["h1"])
    dL_dz1 = dL_dh1 * dh1_dz1
    dL_dw1 = dL_dz1 * x1
    dL_dw3 = dL_dz1 * x2
    dL_db1 = dL_dz1

    dL_dh2 = dL_dz3 * w["w6"]
    dh2_dz2 = v["h2"] * (1 - v["h2"])
    dL_dz2 = dL_dh2 * dh2_dz2
    dL_dw2 = dL_dz2 * x1
    dL_dw4 = dL_dz2 * x2
    dL_db2 = dL_dz2

    FRAME.clear()
    header("4/5  BACKWARD -- hidden layer")
    show_network(w, v)
    SEP()
    L("Chain:  Loss <- out <- z3 <- h1 <- z1 <- {w1, w3, b1}")
    L("        Loss <- out <- z3 <- h2 <- z2 <- {w2, w4, b2}")
    L()
    L(f"  Path through H1:")
    L(f"    dL/dh1 = dL/dz3 * w5 = {dL_dz3:+.4f} * {w['w5']:+.3f} = {dL_dh1:+.6f}")
    L(f"    dh1/dz1 = h1*(1-h1) = {dh1_dz1:.4f}")
    L(f"    dL/dz1 = {dL_dh1:+.6f} * {dh1_dz1:.4f} = {dL_dz1:+.6f}")
    L()
    L(f"  Path through H2:")
    L(f"    dL/dh2 = dL/dz3 * w6 = {dL_dz3:+.4f} * {w['w6']:+.3f} = {dL_dh2:+.6f}")
    L(f"    dh2/dz2 = h2*(1-h2) = {dh2_dz2:.4f}")
    L(f"    dL/dz2 = {dL_dh2:+.6f} * {dh2_dz2:.4f} = {dL_dz2:+.6f}")
    L()
    L(f"  Final gradients:")
    L(f"    dL/dw1 = {dL_dw1:+.6f} (dL/dz1 * x1)")
    L(f"    dL/dw3 = {dL_dw3:+.6f} (dL/dz1 * x2)")
    L(f"    dL/db1 = {dL_db1:+.6f}")
    L(f"    dL/dw2 = {dL_dw2:+.6f} (dL/dz2 * x1)")
    L(f"    dL/dw4 = {dL_dw4:+.6f} (dL/dz2 * x2)")
    L(f"    dL/db2 = {dL_db2:+.6f}")
    show_nav()
    slides.append(list(FRAME))

    # --- Slide 5: GRADIENT MAP + UPDATE ---
    grads = {"w1": dL_dw1, "w2": dL_dw2, "w3": dL_dw3, "w4": dL_dw4,
             "w5": dL_dw5, "w6": dL_dw6, "b1": dL_db1, "b2": dL_db2,
             "b3": dL_db3}

    FRAME.clear()
    header("5/5  GRADIENTS + UPDATE")
    L("Gradients (<<< = increase weight   >>> = decrease weight):")
    L()
    L(f"  Input->H1: {gradient_bar(dL_dw1, 'dL/dw1')}")
    L(f"             {gradient_bar(dL_dw3, 'dL/dw3')}")
    L(f"             {gradient_bar(dL_db1, 'dL/db1')}")
    L(f"  Input->H2: {gradient_bar(dL_dw2, 'dL/dw2')}")
    L(f"             {gradient_bar(dL_dw4, 'dL/dw4')}")
    L(f"             {gradient_bar(dL_db2, 'dL/db2')}")
    L(f"  H->Output: {gradient_bar(dL_dw5, 'dL/dw5')}")
    L(f"             {gradient_bar(dL_dw6, 'dL/dw6')}")
    L(f"             {gradient_bar(dL_db3, 'dL/db3')}")
    SEP()
    L(f"Update: w_new = w_old - {LR} * gradient")
    L()

    new_w = {}
    for name in ["w1", "w2", "w3", "w4", "w5", "w6", "b1", "b2", "b3"]:
        old = w[name]
        nv = old - LR * grads[name]
        new_w[name] = nv
        if grads[name] < -0.001:
            arrow = "UP"
        elif grads[name] > 0.001:
            arrow = "DN"
        else:
            arrow = "  "
        L(f"  {name}: {old:+.4f} -> {nv:+.4f}  {arrow}")

    L()
    preds = [forward(new_w, d[0][0], d[0][1])["out"] for d in DATA]
    new_loss = total_loss(new_w)
    L(f"After update:  Loss: {t_loss:.6f} -> {new_loss:.6f}")
    for i, (inp, tgt) in enumerate(DATA):
        mark = "OK" if abs(preds[i] - tgt) < 0.1 else "  "
        L(f"  [{inp[0]:.0f},{inp[1]:.0f}] -> {preds[i]:.3f}  (want {tgt:.0f})  {mark}")

    converged = all(abs(preds[i] - DATA[i][1]) < 0.05 for i in range(4))
    if converged:
        L()
        L("***  CONVERGED!  All predictions within 0.05 of target  ***")
    show_nav()
    slides.append(list(FRAME))

    return slides, new_w, converged


# ==================================================================
# INPUT HANDLING
# ==================================================================

_old_term = None


def enter_raw_mode():
    global _old_term
    fd = sys.stdin.fileno()
    _old_term = termios.tcgetattr(fd)
    tty.setraw(fd)


def exit_raw_mode():
    if _old_term is not None:
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, _old_term)


def get_key(timeout=None):
    fd = sys.stdin.fileno()
    if timeout is not None:
        ready, _, _ = select.select([fd], [], [], timeout)
        if not ready:
            return None
    else:
        select.select([fd], [], [])

    ch = os.read(fd, 1)
    if ch == b"\x1b":
        ready, _, _ = select.select([fd], [], [], 0.1)
        if ready:
            ch2 = os.read(fd, 1)
            if ch2 == b"[":
                ready2, _, _ = select.select([fd], [], [], 0.1)
                if ready2:
                    ch3 = os.read(fd, 1)
                    if ch3 == b"D":
                        return "LEFT"
                    elif ch3 == b"C":
                        return "RIGHT"
                    elif ch3 == b"A":
                        return "UP"
                    elif ch3 == b"B":
                        return "DOWN"
        return "ESC"
    elif ch in (b"q", b"Q"):
        return "QUIT"
    elif ch == b" ":
        return "SPACE"
    elif ch in (b"\r", b"\n"):
        return "ENTER"
    return ch.decode("utf-8", errors="replace")


def clear_screen():
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()


def render_frame(frame, epoch, slide_idx, total, auto):
    clear_screen()
    if auto:
        mode_line = ">> AUTO-PLAY  (SPACE=pause, arrows=browse)"
    else:
        mode_line = "== PAUSED     (SPACE=play,  arrows=browse)"
    for line in frame:
        if "mode-placeholder" in line:
            sys.stdout.write(f"  Epoch {epoch}   Slide {slide_idx+1}/{total}   {mode_line}\r\n")
        else:
            sys.stdout.write(line + "\r\n")
    sys.stdout.flush()


# ==================================================================
# MAIN
# ==================================================================

AUTO_SPEEDS = {
    1: 2.5, 2: 2.0, 3: 1.5, 5: 0.8, 10: 0.3, 30: 0.1, 100: 0.02
}


def get_auto_speed(epoch):
    speed = 2.5
    for threshold, s in sorted(AUTO_SPEEDS.items()):
        if epoch >= threshold:
            speed = s
    return speed


def main():
    all_slides = []
    w = dict(W_INIT)
    converged = False

    print("Pre-computing slides... ", end="", flush=True)
    for epoch in range(1, 5001):
        slides, w, converged = build_epoch_slides(epoch, w)
        all_slides.extend([(epoch, i, s) for i, s in enumerate(slides)])
        if converged:
            break
    print(f"done! {len(all_slides)} slides across {epoch} epochs.")
    print()
    print("Controls:")
    print("  Left/Right   Navigate slides")
    print("  SPACE        Play / Pause")
    print("  q            Quit")
    print()
    input("Press ENTER to start...")

    auto = True
    pos = 0
    total = len(all_slides)

    enter_raw_mode()
    try:
        while True:
            ep, sl_idx, frame = all_slides[pos]
            render_frame(frame, ep, sl_idx, 5, auto)

            if auto:
                speed = get_auto_speed(ep)
                key = get_key(timeout=speed)
            else:
                key = get_key(timeout=None)

            if key == "QUIT":
                break
            elif key == "SPACE":
                auto = not auto
            elif key == "LEFT":
                auto = False
                pos = max(0, pos - 1)
            elif key == "RIGHT":
                auto = False
                pos = min(total - 1, pos + 1)
            elif key is None and auto:
                pos += 1
                if pos >= total:
                    break

    except KeyboardInterrupt:
        pass
    finally:
        exit_raw_mode()
        clear_screen()
        ep, _, _ = all_slides[min(pos, total - 1)]
        w_final = dict(W_INIT)
        for e in range(1, ep + 1):
            _, w_final, _ = build_epoch_slides(e, w_final)
        preds = [forward(w_final, d[0][0], d[0][1])["out"] for d in DATA]
        print(f"\nStopped at epoch {ep}.")
        print(f"Final predictions:")
        for i, (inp, tgt) in enumerate(DATA):
            print(f"  [{inp[0]:.0f},{inp[1]:.0f}] -> {preds[i]:.3f}  (target {tgt:.0f})")
        if converged:
            print(f"\nThe network learned XOR in {ep} epochs!")


if __name__ == "__main__":
    main()
