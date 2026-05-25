"""Animated ASCII wizard scene for long-running operations.

A small narrative animation: a wizard idles at a desk on the left, walks
to a bookshelf on the right, picks a book, walks back, opens it on the
desk, reads for a moment, then closes it and returns to the shelf for
another. Loops while the LLM is still grinding.

Frames are rendered programmatically from a state machine (phase →
wizard position + pose + book state), which makes the animation easy to
tune (frame count, walking speed, dwell times) without re-typing dozens
of multi-line strings.

Usage:
    with WizardLive(console, status="Searching for citations", total=20) as wiz:
        wiz.update(status="Loading reranker model...")
        do_setup()
        for i in range(1, 21):
            wiz.update(status=f"Reading passage {i}/20", done=i)
            do_work(i)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable

from rich.console import Console, ConsoleOptions, RenderResult
from rich.live import Live
from rich.text import Text


# ---------- scene geometry ----------

SCENE_W = 60
SCENE_H = 12  # taller than v1 to fit the 7-row Mix 1 wizard

# Coordinate system: (x = column, y = row), origin top-left.
# Floor runs along row FLOOR_Y; everything stands on it.
FLOOR_Y = 10

# Desk occupies the left side. Top edge of the desk is at DESK_TOP_Y;
# the desk surface is where books rest.
DESK_LEFT = 4
DESK_RIGHT = 13      # exclusive
DESK_TOP_Y = 7

# Bookshelf on the right. Five rows of books, each row holds 6 books
# separated by | dividers.
SHELF_LEFT = 44
SHELF_RIGHT = 58     # exclusive
SHELF_TOP_Y = 3
SHELF_ROWS = 5
BOOKS_PER_ROW = 6

# Wizard sprite dimensions — Mix 1 (chunky robe + compact face).
# 6 rows tall: hat top + hat body sit directly above the eyes, no brim row.
WIZARD_W = 7
WIZARD_H = 6

# Walking range: the wizard's left edge moves between these x positions.
# Just-in-front-of-desk → just-in-front-of-shelf.
WALK_X_MIN = DESK_RIGHT + 1       # 14
WALK_X_MAX = SHELF_LEFT - WIZARD_W  # 37


# ---------- wizard sprites ----------
#
# Mix 1 wizard: 7 rows tall × 7 cols wide. Pointy hat with star and brim,
# o.o eyes, arms hanging at the sides, chunky triangular robe that flares
# at the bottom. Walking is animated by rippling the robe hem (legs are
# hidden under the robe).
#
# Whitespace in the sprite is treated as transparent when stamping onto
# the canvas, so the wizard doesn't visually erase ground or background.

# All body sprites are 7 cols wide with the wizard centered in the box,
# so symmetric parts remain in the same position under mirroring. reach_right
# is 8 cols wide because the extended arm needs an extra column to its right.
#
# Centering for width-7 sprites:
#   col:  0 1 2 3 4 5 6
#   hat:        _
#   brim:     / * \
#   eyes:     o . o
#   arms:   / |   | \
#   robe:  / /     \ \
#   hem:   \ _ _ _ _ _ /
WIZARD_SPRITES = {
    "idle": [
        r"   _   ",
        r"  /*\  ",
        r"  o.o  ",
        r" /| |\ ",
        r"//   \\",
        r"\_____/",
    ],
    # Walking poses — robe hem ripples one way then the other. Each is the
    # mirror image of the other, so walking-left animation uses these
    # automatically via _mirror_sprite (walk_a mirror → walk_b).
    "walk_a": [
        r"   _   ",
        r"  /*\  ",
        r"  o.o  ",
        r" /| |\ ",
        r"//   \\",
        r"\__\__/",
    ],
    "walk_b": [
        r"   _   ",
        r"  /*\  ",
        r"  o.o  ",
        r" /| |\ ",
        r"//   \\",
        r"\__/__/",
    ],
    # At the shelf, reaching out to the right. The right arm extends one
    # column past the body — sprite is 8 cols wide for this pose so the
    # > can land on the shelf edge. Only used facing right (never mirrored).
    "reach_right": [
        r"   _    ",
        r"  /*\   ",
        r"  o.o   ",
        r" /| |\_>",
        r"//   \\ ",
        r"\_____/ ",
    ],
    # Carrying a closed book in front of the body — book sits between
    # the arms as |=|.
    "carry_idle": [
        r"   _   ",
        r"  /*\  ",
        r"  o.o  ",
        r" /|=|\ ",
        r"//   \\",
        r"\_____/",
    ],
    # Carrying + walking: book stays put, robe hem ripples.
    "carry_walk_a": [
        r"   _   ",
        r"  /*\  ",
        r"  o.o  ",
        r" /|=|\ ",
        r"//   \\",
        r"\__\__/",
    ],
    "carry_walk_b": [
        r"   _   ",
        r"  /*\  ",
        r"  o.o  ",
        r" /|=|\ ",
        r"//   \\",
        r"\__/__/",
    ],
    # Placing book on the desk — eyes look down, arms come together.
    "place": [
        r"   _   ",
        r"  /*\  ",
        r"  _._  ",
        r" /|_|\ ",
        r"//   \\",
        r"\_____/",
    ],
    # Reading at desk: looking down (v.v), hands held over the book.
    "reading_a": [
        r"   _   ",
        r"  /*\  ",
        r"  v.v  ",
        r" /|_|\ ",
        r"//   \\",
        r"\_____/",
    ],
    # Reading_b: eyes briefly closed for a subtle blink during the
    # reading phase.
    "reading_b": [
        r"   _   ",
        r"  /*\  ",
        r"  -.-  ",
        r" /|_|\ ",
        r"//   \\",
        r"\_____/",
    ],
}


# ---------- canvas ----------

def _new_canvas() -> list[list[str]]:
    return [[" "] * SCENE_W for _ in range(SCENE_H)]


def _stamp(canvas: list[list[str]], x: int, y: int, sprite: list[str]) -> None:
    """Stamp a sprite onto the canvas, treating spaces as transparent."""
    for dy, line in enumerate(sprite):
        cy = y + dy
        if not (0 <= cy < SCENE_H):
            continue
        for dx, ch in enumerate(line):
            if ch == " ":
                continue
            cx = x + dx
            if 0 <= cx < SCENE_W:
                canvas[cy][cx] = ch


# Sprites are designed facing right. To face left we reverse each line
# and swap direction-bearing characters (/ ↔ \, < ↔ >, etc.) so the
# wizard's robe ripple, arm position, etc. flip consistently.
_MIRROR_TRANS = str.maketrans({
    "/": "\\",
    "\\": "/",
    "<": ">",
    ">": "<",
    "(": ")",
    ")": "(",
    "[": "]",
    "]": "[",
})


def _mirror_sprite(sprite: list[str]) -> list[str]:
    """Flip a sprite horizontally for the opposite-facing pose."""
    return [line.translate(_MIRROR_TRANS)[::-1] for line in sprite]


def _draw_floor(canvas: list[list[str]]) -> None:
    for x in range(SCENE_W):
        canvas[FLOOR_Y][x] = "═"


def _draw_desk(canvas: list[list[str]], book_state: str) -> None:
    """Draw the desk. book_state: 'none' | 'closed' | 'open'.

    The desk is a simple table — a horizontal top surface, two legs
    descending to the floor. A book on the desk sits on top of the
    surface.
    """
    # Surface top edge — characters
    width = DESK_RIGHT - DESK_LEFT
    surface_top = "_" * width
    surface_bot = "|" + "_" * (width - 2) + "|"
    # Stamp
    for x in range(DESK_LEFT, DESK_RIGHT):
        canvas[DESK_TOP_Y][x] = "_"
    canvas[DESK_TOP_Y + 1][DESK_LEFT] = "|"
    canvas[DESK_TOP_Y + 1][DESK_RIGHT - 1] = "|"
    for x in range(DESK_LEFT + 1, DESK_RIGHT - 1):
        canvas[DESK_TOP_Y + 1][x] = "_"
    # Legs
    canvas[DESK_TOP_Y + 2][DESK_LEFT + 1] = "|"
    canvas[DESK_TOP_Y + 2][DESK_RIGHT - 2] = "|"

    # Book on the desk — positioned at the RIGHT edge of the desk so it
    # sits right next to where the wizard stands (the wizard walks to
    # WALK_X_MIN = DESK_RIGHT + 1 when at the desk).
    if book_state == "closed":
        bx = DESK_RIGHT - 4
        canvas[DESK_TOP_Y - 1][bx] = "_"
        canvas[DESK_TOP_Y - 1][bx + 1] = "_"
        canvas[DESK_TOP_Y - 1][bx + 2] = "_"
    elif book_state == "open":
        # An open book — solid top edge over two pages with wavy text.
        bx = DESK_RIGHT - 6
        for i in range(5):
            canvas[DESK_TOP_Y - 2][bx + i] = "_"
        canvas[DESK_TOP_Y - 1][bx] = "|"
        canvas[DESK_TOP_Y - 1][bx + 1] = "~"
        canvas[DESK_TOP_Y - 1][bx + 2] = "|"
        canvas[DESK_TOP_Y - 1][bx + 3] = "~"
        canvas[DESK_TOP_Y - 1][bx + 4] = "|"


def _draw_shelf(canvas: list[list[str]], missing: tuple[int, int] | None) -> None:
    """Draw the bookshelf. `missing` is an (row, col) book to leave empty,
    or None for a full shelf."""
    width = SHELF_RIGHT - SHELF_LEFT
    # Top and bottom edges of the shelf
    for x in range(SHELF_LEFT, SHELF_RIGHT):
        canvas[SHELF_TOP_Y][x] = "_"
    canvas[SHELF_TOP_Y + SHELF_ROWS + 1][SHELF_LEFT] = "|"
    canvas[SHELF_TOP_Y + SHELF_ROWS + 1][SHELF_RIGHT - 1] = "|"
    for x in range(SHELF_LEFT + 1, SHELF_RIGHT - 1):
        canvas[SHELF_TOP_Y + SHELF_ROWS + 1][x] = "_"

    # Each shelf row: |█|█|█|█|█|█|
    # SHELF_RIGHT - SHELF_LEFT = 14; we want 6 books between 2 outer walls
    # with | separators: |B|B|B|B|B|B| = 13 chars. Add 1 char padding right.
    for row in range(SHELF_ROWS):
        y = SHELF_TOP_Y + 1 + row
        canvas[y][SHELF_LEFT] = "|"
        for b in range(BOOKS_PER_ROW):
            book_x = SHELF_LEFT + 1 + b * 2
            sep_x = book_x + 1
            if missing == (row, b):
                canvas[y][book_x] = " "
            else:
                canvas[y][book_x] = "█"
            canvas[y][sep_x] = "|"


# ---------- state machine ----------

@dataclass(frozen=True)
class _Phase:
    wizard_x: int
    wizard_pose: str
    book_state: str            # 'none' | 'closed' | 'open' (book on desk)
    missing_book: tuple[int, int] | None  # which shelf book is gone
    facing: str = "right"      # 'right' (sprite as-designed) or 'left' (mirrored)


def _phase_for(frame: int) -> _Phase:
    """Map a frame counter to a scene state. Total loop ≈ 50 frames."""
    # Phase boundaries (frames are cumulative)
    f_idle_start = 4               # initial idle at desk
    f_walk_to_shelf = 16           # walking right
    f_reach = 4                    # reaching for book
    f_at_shelf_with_book = 2       # held it, brief pause
    f_walk_to_desk = 16            # walking left, carrying
    f_place = 2                    # placing book on desk
    f_reading = 8                  # reading
    LOOP = (
        f_idle_start + f_walk_to_shelf + f_reach + f_at_shelf_with_book
        + f_walk_to_desk + f_place + f_reading
    )
    f = frame % LOOP

    # Which book the wizard pulls — always from the bottom row of the
    # shelf (that's where the wizard's hand reaches, see WIZARD_SPRITES
    # 'reach_right'). The column rotates per loop so each round-trip
    # picks a different book and the gap visibly moves.
    loop_idx = frame // LOOP
    target = (SHELF_ROWS - 1, loop_idx % BOOKS_PER_ROW)

    cur = 0
    if f < cur + f_idle_start:
        # Idle at desk before heading out — already faces right toward the
        # shelf so the transition into walk_right doesn't include a spin.
        return _Phase(
            wizard_x=WALK_X_MIN,
            wizard_pose="idle",
            book_state="none",
            missing_book=None,
            facing="right",
        )
    cur += f_idle_start

    if f < cur + f_walk_to_shelf:
        # progress 0 → 1 across the walk
        progress = (f - cur) / max(1, f_walk_to_shelf - 1)
        x = int(WALK_X_MIN + progress * (WALK_X_MAX - WALK_X_MIN))
        pose = "walk_a" if (f - cur) % 2 == 0 else "walk_b"
        return _Phase(
            wizard_x=x, wizard_pose=pose,
            book_state="none", missing_book=None,
            facing="right",
        )
    cur += f_walk_to_shelf

    if f < cur + f_reach:
        # At the shelf — pulse the reach pose a few frames so it reads as
        # an active gesture rather than a single still. The reach_right
        # sprite has the arm extending right; rendered as-designed.
        return _Phase(
            wizard_x=WALK_X_MAX,
            wizard_pose="reach_right",
            book_state="none",
            missing_book=None,
            facing="right",
        )
    cur += f_reach

    if f < cur + f_at_shelf_with_book:
        # Just pulled the book — now turning to walk back. carry_idle is
        # symmetric so the mirror is invisible, but setting facing="left"
        # here keeps the state machine internally consistent.
        return _Phase(
            wizard_x=WALK_X_MAX,
            wizard_pose="carry_idle",
            book_state="none",
            missing_book=target,
            facing="left",
        )
    cur += f_at_shelf_with_book

    if f < cur + f_walk_to_desk:
        progress = (f - cur) / max(1, f_walk_to_desk - 1)
        x = int(WALK_X_MAX - progress * (WALK_X_MAX - WALK_X_MIN))
        pose = "carry_walk_a" if (f - cur) % 2 == 0 else "carry_walk_b"
        return _Phase(
            wizard_x=x, wizard_pose=pose,
            book_state="none", missing_book=target,
            facing="left",
        )
    cur += f_walk_to_desk

    if f < cur + f_place:
        # Place book on the desk, still facing the desk (left).
        return _Phase(
            wizard_x=WALK_X_MIN,
            wizard_pose="place",
            book_state="closed",
            missing_book=target,
            facing="left",
        )
    cur += f_place

    # Reading at the desk — still facing left toward the desk.
    pose = "reading_a" if (f - cur) % 2 == 0 else "reading_b"
    return _Phase(
        wizard_x=WALK_X_MIN,
        wizard_pose=pose,
        book_state="open",
        missing_book=target,
        facing="left",
    )


def _render_frame(frame: int) -> list[str]:
    canvas = _new_canvas()
    phase = _phase_for(frame)
    _draw_floor(canvas)
    _draw_desk(canvas, phase.book_state)
    _draw_shelf(canvas, phase.missing_book)
    # Stamp wizard so its feet rest on the floor. Mirror the sprite if
    # the phase has him facing left so the walking ripple, reaching arm,
    # etc. all run in the correct direction.
    wizard_top_y = FLOOR_Y - WIZARD_H
    sprite = WIZARD_SPRITES[phase.wizard_pose]
    if phase.facing == "left":
        sprite = _mirror_sprite(sprite)
    _stamp(canvas, phase.wizard_x, wizard_top_y, sprite)
    return ["".join(row) for row in canvas]


# ---------- live display ----------

@dataclass
class _State:
    """Mutable display state shared between the Live thread and the caller."""

    status: str = "Thinking..."
    done: int = 0
    total: int | None = None
    start: float = 0.0


class _SceneRenderable:
    """Rich renderable: picks an animation frame from elapsed time and
    yields the scene + status + progress lines."""

    def __init__(self, state: _State, frame_seconds: float = 0.18):
        self.state = state
        # 0.18s per frame = ~5.5 FPS. Smooth enough for walking, not jittery.
        self.frame_seconds = frame_seconds

    def __rich_console__(
        self, console: Console, options: ConsoleOptions,
    ) -> RenderResult:
        elapsed = time.monotonic() - self.state.start
        frame_idx = int(elapsed / self.frame_seconds)
        scene_lines = _render_frame(frame_idx)
        # Retro green-on-black terminal styling — the whole loading display
        # uses green tones reminiscent of classic CRT terminals.
        for line in scene_lines:
            yield Text(line, style="green")
        yield Text("")
        yield Text("  " + self.state.status, style="bold green")
        if self.state.total:
            bar_w = 32
            filled = max(0, min(bar_w, int(bar_w * self.state.done / self.state.total)))
            bar = "█" * filled + "░" * (bar_w - filled)
            pct = int(100 * self.state.done / self.state.total)
            yield Text(
                f"  [{bar}] {self.state.done}/{self.state.total}  "
                f"({pct}%)  {elapsed:0.0f}s",
                style="dim green",
            )
        else:
            yield Text(f"  {elapsed:0.0f}s elapsed", style="dim green")


class WizardLive:
    """Context manager wrapping Rich's Live with the wizard scene animation."""

    def __init__(
        self,
        console: Console,
        *,
        status: str = "Thinking...",
        total: int | None = None,
    ) -> None:
        self.state = _State(status=status, total=total)
        self._renderable = _SceneRenderable(self.state)
        self._live = Live(
            self._renderable,
            console=console,
            # 8 FPS — comfortably faster than the frame cadence so we never
            # miss a sub-frame change.
            refresh_per_second=8,
            transient=True,
        )

    def __enter__(self) -> "WizardLive":
        self.state.start = time.monotonic()
        self._live.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        return self._live.__exit__(exc_type, exc, tb)

    def update(
        self,
        *,
        status: str | None = None,
        done: int | None = None,
        total: int | None = None,
    ) -> None:
        """Update the visible state. Any field left None is unchanged."""
        if status is not None:
            self.state.status = status
        if done is not None:
            self.state.done = done
        if total is not None:
            self.state.total = total
