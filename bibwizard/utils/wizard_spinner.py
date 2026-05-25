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
    sparkle: bool = False      # draw "aha!" sparkles above the wizard's hat


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
        # here keeps the state machine internally consistent. The first
        # frame of this phase gets a sparkle ("got it!" moment).
        return _Phase(
            wizard_x=WALK_X_MAX,
            wizard_pose="carry_idle",
            book_state="none",
            missing_book=target,
            facing="left",
            sparkle=(f - cur == 0),
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

    # Reading at the desk — still facing left toward the desk. The
    # midpoint of the reading phase gets a sparkle ("aha!" moment — the
    # wizard found something interesting in the book).
    pose = "reading_a" if (f - cur) % 2 == 0 else "reading_b"
    return _Phase(
        wizard_x=WALK_X_MIN,
        wizard_pose=pose,
        book_state="open",
        missing_book=target,
        facing="left",
        sparkle=(f - cur == f_reading // 2),
    )


def _draw_sparkles(canvas: list[list[str]], wizard_x: int, wizard_top_y: int) -> None:
    """Stamp 'aha!' sparkles above the wizard's hat. Used for moments where
    the wizard has just found or recognized something — picked the right
    book off the shelf, finished a satisfying passage."""
    sparkles = [
        (1, -2, "*"),
        (5, -2, "*"),
        (3, -3, "✦"),
    ]
    for dx, dy, ch in sparkles:
        sx = wizard_x + dx
        sy = wizard_top_y + dy
        if 0 <= sy < SCENE_H and 0 <= sx < SCENE_W:
            canvas[sy][sx] = ch


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
    if phase.sparkle:
        _draw_sparkles(canvas, phase.wizard_x, wizard_top_y)
    return ["".join(row) for row in canvas]


# ---------- alternative scene: wizard reading at his desk ----------
#
# Used by long-running "read each candidate passage" operations like
# `bibwizard cite`. The wizard never leaves the desk — he sits with an
# open book, scans the page, blinks, and occasionally has an "aha!"
# moment with a sparkle over his hat. Thematically maps to the LLM
# entailment loop, where each candidate passage is a page being read.

# Eye states. All are palindromic under mirroring (so we can mirror the
# wizard to face the book without flipping his expression).
_READING_EYES = {
    "read": "v.v",   # default: looking down at book
    "blink": "-.-",  # eyes closed for a beat
    "aha": "^.^",    # raised eyebrows, eureka moment
    "squint": ">.<", # focused / interested
}

# 12-frame cycle — at ~5 fps that's ~2.4s per cycle.
# Pattern: mostly reading, occasional blink, occasional aha! with sparkle.
_READING_FRAMES = [
    ("read",   False),
    ("read",   False),
    ("read",   False),
    ("blink",  False),
    ("read",   False),
    ("read",   False),
    ("squint", False),
    ("read",   False),
    ("read",   False),
    ("aha",    True),   # sparkle frame!
    ("read",   False),
    ("read",   False),
]


# ---------- pixel-art scene ----------
#
# Higher-resolution "16-bit game sprite" style of the wizard. Each
# character cell shows two stacked pixels via the upper-half block (▀)
# trick — the foreground color is the upper pixel, the background color
# the lower pixel. With 24-bit ANSI color support that's a 9-color
# palette per scene.
#
# The wizard sprite is 20 px wide × 32 px tall; combined with sparkles
# above the hat for the "aha!" moment we render into a 28 × 36 pixel
# canvas (= 28 chars wide × 18 char-rows tall). Modest terminal real
# estate, clear sprite silhouette.

PIXEL_PALETTE: dict[str, str | None] = {
    # Wizard
    "H": "#5e2a99",   # hat (deep purple)
    "K": "#3d1a66",   # hat brim (darker purple)
    "S": "#f5d742",   # star on hat (gold)
    "F": "#f0c69a",   # skin
    "E": "#1a1a1a",   # eye (near-black)
    "W": "#ececec",   # beard / hair (off-white)
    "R": "#2a5fbe",   # robe (royal blue)
    "T": "#d4a017",   # gold trim
    "L": "#2c2c2c",   # boots (very dark grey)
    # Scene
    "B": "#7a3a2a",   # brick (warm reddish-brown)
    "M": "#3a2018",   # mortar (dark grout between bricks)
    "D": "#5a3a1e",   # wood dark (desk / shelf frame)
    "d": "#8a5e2e",   # wood light (highlight)
    "C": "#a02c2c",   # book spine (red)
    "c": "#5e7fa0",   # book spine (steel blue alt)
    "O": "#1c1c1c",   # lantern housing (iron)
    "Y": "#ffd166",   # lantern glow (warm yellow)
    "y": "#ffeaa3",   # lantern flicker (lighter glow)
    "G": "#1f3a26",   # leaves / floor moss
    "g": "#2f5036",   # lighter floor accent
    ".": None,         # transparent
}

# Wizard sprite — 12 px wide × 16 px tall.
# Hat: narrow cone above a wider brim, so the brim "winks" out clearly.
# Same upper rows for every pose; only the last row (legs) changes.
_PX_BODY = [
    ".....HH.....",  #  0  hat point
    "....HSSH....",  #  1  star
    "...HHHHHH...",  #  2  cone widens
    "...HHHHHH...",  #  3  cone holds
    "..HHHHHHHH..",  #  4  lower cone (narrower than brim)
    "KKKKKKKKKKKK",  #  5  brim — sticks out beyond cone
    "....FFFF....",  #  6  face
    "...FEFFEF...",  #  7  eyes
    "...FFFFFF...",  #  8
    "....WWWW....",  #  9  beard
    "...WWWWWW...",  # 10
    "..RRRRRRRR..",  # 11  robe shoulders
    ".RRRRRRRRRR.",  # 12
    ".RTTTTTTTTR.",  # 13  gold clasp / belt
    ".RRRRRRRRRR.",  # 14
]

_PX_LEGS_IDLE = [
    "..LL....LL..",  # 15  legs square (standing still)
]

_PX_LEGS_WALK_A = [
    "..LL.....LL.",  # right leg shifted right (mid-stride)
]

_PX_LEGS_WALK_B = [
    ".LL.....LL..",  # left leg shifted left (mid-stride)
]

# Reading body — the wizard is looking *down* at the open book on his
# desk, so the eyes shift one row lower (row 8 instead of row 7).
# Otherwise identical to _PX_BODY. The _BLINK variant has the eyes
# briefly closed for a subtle breathing rhythm during the reading phase.
_PX_BODY_READING = [
    ".....HH.....",
    "....HSSH....",
    "...HHHHHH...",
    "...HHHHHH...",
    "..HHHHHHHH..",
    "KKKKKKKKKKKK",
    "....FFFF....",
    "...FFFFFF...",   # no eyes here — looking down
    "...FEFFEF...",   # eyes one row lower
    "....WWWW....",
    "...WWWWWW...",
    "..RRRRRRRR..",
    ".RRRRRRRRRR.",
    ".RTTTTTTTTR.",
    ".RRRRRRRRRR.",
]

_PX_BODY_READING_BLINK = [
    ".....HH.....",
    "....HSSH....",
    "...HHHHHH...",
    "...HHHHHH...",
    "..HHHHHHHH..",
    "KKKKKKKKKKKK",
    "....FFFF....",
    "...FFFFFF...",
    "...FFFFFF...",   # eyes closed (no E pixels)
    "....WWWW....",
    "...WWWWWW...",
    "..RRRRRRRR..",
    ".RRRRRRRRRR.",
    ".RTTTTTTTTR.",
    ".RRRRRRRRRR.",
]

# Sparkle pixel positions relative to the wizard's top-left.
_PX_SPARKLES = [
    (2, -3, "S"),
    (9, -3, "S"),
    (5, -5, "S"),
    (6, -5, "S"),
]


def _build_pixel_sprite(
    legs: list[str], body: list[str] | None = None,
) -> list[str]:
    return (body if body is not None else _PX_BODY) + legs


_PX_WIZARD_IDLE = _build_pixel_sprite(_PX_LEGS_IDLE)
_PX_WIZARD_WALK_A = _build_pixel_sprite(_PX_LEGS_WALK_A)
_PX_WIZARD_WALK_B = _build_pixel_sprite(_PX_LEGS_WALK_B)
_PX_WIZARD_READING = _build_pixel_sprite(_PX_LEGS_IDLE, _PX_BODY_READING)
_PX_WIZARD_READING_BLINK = _build_pixel_sprite(
    _PX_LEGS_IDLE, _PX_BODY_READING_BLINK,
)

WIZARD_PX_W = 12
WIZARD_PX_H = 16


# ---------- scene components ----------

# Wall lantern, 5 px wide × 7 px tall. Two variants for flicker.
_PX_LANTERN_BRIGHT = [
    "..O..",
    ".OOO.",
    "OyYyO",
    "OYYyO",
    "OYYYO",
    "OYyYO",
    ".OOO.",
]
_PX_LANTERN_DIM = [
    "..O..",
    ".OOO.",
    "OyyyO",
    "OYyyO",
    "OyYyO",
    "OYyyO",
    ".OOO.",
]

# Bookshelf, 14 px wide × 18 px tall. Three shelves, each with several
# book spines in alternating colors.
_PX_BOOKSHELF = [
    "DDDDDDDDDDDDDD",   # top
    "D...........d",
    "DC.c.C.c.C.cD",
    "DC.c.C.c.C.cD",
    "DC.c.C.c.C.cD",
    "DDDDDDDDDDDDDD",   # shelf divider
    "DC.c.C.c.C.cD",
    "DC.c.C.c.C.cD",
    "DC.c.C.c.C.cD",
    "DDDDDDDDDDDDDD",   # shelf divider
    "Dc.C.c.C.c.CD",
    "Dc.C.c.C.c.CD",
    "Dc.C.c.C.c.CD",
    "DDDDDDDDDDDDDD",   # shelf divider
    "Dd.........dD",
    "Dd.........dD",
    "DDDDDDDDDDDDDD",   # base
    "DDDDDDDDDDDDDD",
]

# Desk, 14 px wide × 6 px tall. The book that sits on top is stamped
# separately so it can switch between closed (compact) and open (wide,
# pages visible) depending on what the wizard is doing.
_PX_DESK = [
    "ddddddddddddDD",   # 0  desk top surface
    "DDDDDDDDDDDDDD",   # 1  top edge
    ".D..........D.",   # 2
    ".D..........D.",   # 3
    ".D..........D.",   # 4
    "DDD........DDD",   # 5  legs
]

# Books that sit on the desk. Closed = nobody's reading right now (the
# wizard's away at the shelf, or has just set the book back down). Open
# = he's actively reading the page — wider sprite, pages visible with
# text lines and a central binding.
_PX_BOOK_CLOSED = [
    "CCCC",   # 0  red cover, top edge
    "CcCC",   # 1  spine + base
]

_PX_BOOK_OPEN = [
    "WWWWWWWWWW",   # 0  pages spread (off-white)
    "WccCcccCcW",   # 1  text lines either side of central binding
    "WccCcccCcW",   # 2  more text
    "CCCCCCCCCC",   # 3  covers wrap around the bottom
]


def _stamp_pixel(
    canvas: list[list[str]], x: int, y: int, sprite: list[str],
) -> None:
    """Stamp a pixel sprite onto the canvas, '.' = transparent."""
    h = len(canvas)
    w = len(canvas[0]) if h else 0
    for sy, row in enumerate(sprite):
        cy = y + sy
        if not (0 <= cy < h):
            continue
        for sx, ch in enumerate(row):
            if ch == ".":
                continue
            cx = x + sx
            if 0 <= cx < w:
                canvas[cy][cx] = ch


def _fill_brick_wall(canvas: list[list[str]], top: int, bottom: int) -> None:
    """Paint a brick-pattern wall behind everything else.

    Bricks are 5 px wide × 2 px tall with 1-px mortar between them.
    Every other brick row is offset by 3 px (half-brick stagger) for the
    classic running-bond brick pattern.
    """
    w = len(canvas[0]) if canvas else 0
    BRICK_W = 6  # 5 brick + 1 mortar
    BRICK_H = 3  # 2 brick + 1 mortar
    for y in range(top, bottom):
        row_in_brick = (y - top) % BRICK_H
        brick_row_idx = (y - top) // BRICK_H
        offset = 0 if brick_row_idx % 2 == 0 else BRICK_W // 2
        for x in range(w):
            if row_in_brick == BRICK_H - 1:
                canvas[y][x] = "M"  # horizontal mortar row
            elif (x + offset) % BRICK_W == 0:
                canvas[y][x] = "M"  # vertical mortar column
            else:
                canvas[y][x] = "B"  # brick body


def _draw_pixel_floor(canvas: list[list[str]], top: int, bottom: int) -> None:
    """Wooden floor strip at the bottom of the pixel canvas — dark
    planks with occasional lighter highlight rows. Named separately
    from the ASCII scene's _draw_floor() to avoid the previous name
    collision (Python silently lets two same-named module functions
    override each other; the second wins, and the first call site
    crashes with a signature mismatch)."""
    w = len(canvas[0]) if canvas else 0
    for y in range(top, bottom):
        accent = (y == top) or ((y - top) % 4 == 1)
        for x in range(w):
            canvas[y][x] = "d" if accent else "D"


# Pixel canvas size — picked so the sprite + scene fit in roughly the
# same terminal real estate as the ASCII scenes.
PX_W = 60
PX_H = 36


def _pixels_to_text_lines(canvas: list[list[str]]) -> list[Text]:
    """Convert a pixel grid into Rich Text lines using upper-half block
    characters with foreground = top pixel color, background = bottom
    pixel color. Two pixel rows become one terminal row."""
    lines: list[Text] = []
    h = len(canvas)
    for r in range(0, h, 2):
        top = canvas[r]
        bot = canvas[r + 1] if r + 1 < h else ["."] * len(top)
        line = Text()
        for tc, bc in zip(top, bot):
            tcol = PIXEL_PALETTE[tc]
            bcol = PIXEL_PALETTE[bc]
            if tcol is None and bcol is None:
                line.append(" ")
            elif tcol is None:
                line.append("▄", style=bcol)
            elif bcol is None:
                line.append("▀", style=tcol)
            else:
                line.append("▀", style=f"{tcol} on {bcol}")
        lines.append(line)
    return lines


# Scene layout constants — pixel coordinates within the canvas.
_FLOOR_TOP_PX = 30
_FLOOR_BOTTOM_PX = 36
_SHELF_X = 44
_SHELF_Y = _FLOOR_TOP_PX - len(_PX_BOOKSHELF)
_DESK_X = 2
_DESK_Y = _FLOOR_TOP_PX - len(_PX_DESK) + 1   # +1 so legs sit on floor
_LANTERN_LEFT = (10, 4)
_LANTERN_RIGHT = (46, 4)

# Walking range for the wizard — between desk and shelf, on the floor.
# X_MIN is the desk-side stopping point: right next to the desk's right
# edge so the wizard can read the book on top of it at close range.
# X_MAX is the shelf-side stopping point — just left of the shelf.
_WIZ_X_MIN = _DESK_X + 14            # right next to the desk's right edge
_WIZ_X_MAX = _SHELF_X - WIZARD_PX_W - 2
_WIZ_Y = _FLOOR_TOP_PX - WIZARD_PX_H

# Book positions on the desk. The book sits on the right side of the
# desk surface so the wizard (standing just to its right) reads it at
# close range. Bottom row of either book sits one pixel above the desk
# top surface — that's "on the desk" in pixel-art parlance.
_BOOK_CLOSED_X = _DESK_X + 8         # right-of-center on desk surface
_BOOK_CLOSED_Y = _DESK_Y - 2         # 2-tall book, bottom just above desk top
_BOOK_OPEN_X = _DESK_X + 3           # open book is wider, shifted left
_BOOK_OPEN_Y = _DESK_Y - 4           # 4-tall book, bottom just above desk top


# ---------- pixel-scene animation state machine ----------
#
# A full loop is six phases: walk to the shelf, browse the books,
# realize you've found the right one (sparkle!), walk back to the desk,
# read the open book, have an "aha!" insight (sparkle!). Sparkles only
# fire on the two realization phases so they read as meaningful beats
# rather than ambient decoration — per user request, no sparkles during
# the walking phases.

_PHASE_WALK_RIGHT = "walk_right"
_PHASE_BROWSE = "browse"
_PHASE_REALIZE_SHELF = "realize_shelf"
_PHASE_WALK_LEFT = "walk_left"
_PHASE_READ = "read"
_PHASE_REALIZE_DESK = "realize_desk"

_PX_PHASE_FRAMES: dict[str, int] = {
    _PHASE_WALK_RIGHT: 8,
    _PHASE_BROWSE: 6,
    _PHASE_REALIZE_SHELF: 3,
    _PHASE_WALK_LEFT: 8,
    _PHASE_READ: 10,
    _PHASE_REALIZE_DESK: 3,
}
_PX_PHASE_ORDER: list[str] = [
    _PHASE_WALK_RIGHT, _PHASE_BROWSE, _PHASE_REALIZE_SHELF,
    _PHASE_WALK_LEFT, _PHASE_READ, _PHASE_REALIZE_DESK,
]
_PX_LOOP_FRAMES: int = sum(_PX_PHASE_FRAMES.values())  # 38 → ~7s at ~5fps


def _pixel_phase(frame: int) -> tuple[str, int, int]:
    """Map a frame counter to (phase, sub_frame_within_phase, loop_idx).

    loop_idx counts completed loops so we can rotate which shelf book
    gets picked each round (the gap moves around the shelf over time).
    """
    loop_idx = frame // _PX_LOOP_FRAMES
    f = frame % _PX_LOOP_FRAMES
    cur = 0
    for phase in _PX_PHASE_ORDER:
        n = _PX_PHASE_FRAMES[phase]
        if f < cur + n:
            return phase, f - cur, loop_idx
        cur += n
    return _PX_PHASE_ORDER[-1], 0, loop_idx


# Bottom-shelf book slot the wizard reaches for. The shelf has 6 books
# per row; we rotate book_idx by loop so the gap moves around.
_PX_SHELF_PICK_ROW = 2  # 0=top shelf, 2=bottom shelf
_PX_SHELF_BOOK_COLS = [1, 3, 5, 7, 9, 11]
_PX_SHELF_ROW_SPRITE_ROWS = {
    0: (2, 3, 4),
    1: (6, 7, 8),
    2: (10, 11, 12),
}


def _erase_shelf_book(
    canvas: list[list[str]],
    shelf_x: int,
    shelf_y: int,
    row_idx: int,
    book_idx: int,
) -> None:
    """Blank out one book on the shelf to show the wizard's picked it up.
    row_idx 0/1/2 = top/middle/bottom; book_idx 0-5 = which of the six
    books in that row. The book pixel is replaced with the mortar color
    so the gap reads as a dark hole rather than just transparent."""
    rows = _PX_SHELF_ROW_SPRITE_ROWS.get(row_idx)
    if rows is None:
        return
    if not (0 <= book_idx < len(_PX_SHELF_BOOK_COLS)):
        return
    bx = shelf_x + _PX_SHELF_BOOK_COLS[book_idx]
    h = len(canvas)
    w = len(canvas[0]) if h else 0
    for sr in rows:
        y = shelf_y + sr
        if 0 <= y < h and 0 <= bx < w:
            canvas[y][bx] = "M"


def _render_pixel_frame(frame: int) -> list[Text]:
    """Render frame `n` of the pixel-art wizard scene.

    The wizard cycles through six phases: walking to the shelf, browsing
    the books, realizing he's found the right one (sparkle!), walking
    back to the desk, reading the open book, having an insight while
    reading (sparkle!). Sparkles fire only on the two realization beats;
    walking phases are sparkle-free.
    """
    canvas = [["."] * PX_W for _ in range(PX_H)]
    phase, sub, loop_idx = _pixel_phase(frame)

    # --- background ---
    _fill_brick_wall(canvas, 0, _FLOOR_TOP_PX)
    _draw_pixel_floor(canvas, _FLOOR_TOP_PX, _FLOOR_BOTTOM_PX)

    # --- lanterns flicker every other frame ---
    lantern_sprite = (
        _PX_LANTERN_BRIGHT if frame % 2 == 0 else _PX_LANTERN_DIM
    )
    _stamp_pixel(canvas, *_LANTERN_LEFT, lantern_sprite)
    _stamp_pixel(canvas, *_LANTERN_RIGHT, lantern_sprite)

    # --- shelf, with the wizard's pick missing once he's grabbed it ---
    _stamp_pixel(canvas, _SHELF_X, _SHELF_Y, _PX_BOOKSHELF)
    book_taken_phases = (
        _PHASE_REALIZE_SHELF, _PHASE_WALK_LEFT,
        _PHASE_READ, _PHASE_REALIZE_DESK,
    )
    if phase in book_taken_phases:
        # Rotate which book gets picked each loop so the gap moves around.
        picked_book = loop_idx % len(_PX_SHELF_BOOK_COLS)
        _erase_shelf_book(
            canvas, _SHELF_X, _SHELF_Y, _PX_SHELF_PICK_ROW, picked_book,
        )

    # --- desk + book on top ---
    _stamp_pixel(canvas, _DESK_X, _DESK_Y, _PX_DESK)
    if phase in (_PHASE_READ, _PHASE_REALIZE_DESK):
        _stamp_pixel(canvas, _BOOK_OPEN_X, _BOOK_OPEN_Y, _PX_BOOK_OPEN)
    else:
        _stamp_pixel(canvas, _BOOK_CLOSED_X, _BOOK_CLOSED_Y, _PX_BOOK_CLOSED)

    # --- wizard position + pose ---
    if phase == _PHASE_WALK_RIGHT:
        n = _PX_PHASE_FRAMES[phase]
        progress = sub / max(1, n - 1)
        wx = int(_WIZ_X_MIN + progress * (_WIZ_X_MAX - _WIZ_X_MIN))
        sprite = _PX_WIZARD_WALK_A if sub % 2 == 0 else _PX_WIZARD_WALK_B
    elif phase == _PHASE_WALK_LEFT:
        n = _PX_PHASE_FRAMES[phase]
        progress = sub / max(1, n - 1)
        wx = int(_WIZ_X_MAX - progress * (_WIZ_X_MAX - _WIZ_X_MIN))
        sprite = _PX_WIZARD_WALK_A if sub % 2 == 0 else _PX_WIZARD_WALK_B
    elif phase == _PHASE_BROWSE:
        # Standing at the shelf, scanning titles. Idle pose — the
        # narrative is conveyed by position (he's at the shelf, not the
        # desk) and the still-full shelf behind him.
        wx = _WIZ_X_MAX
        sprite = _PX_WIZARD_IDLE
    elif phase == _PHASE_REALIZE_SHELF:
        # Found the right book! Idle pose + sparkles + the missing book
        # behind him tell the story.
        wx = _WIZ_X_MAX
        sprite = _PX_WIZARD_IDLE
    elif phase == _PHASE_READ:
        # Reading at the desk: looking-down eyes, with a brief mid-phase
        # blink for a subtle breathing rhythm.
        wx = _WIZ_X_MIN
        sprite = (
            _PX_WIZARD_READING_BLINK if sub in (3, 7)
            else _PX_WIZARD_READING
        )
    else:  # _PHASE_REALIZE_DESK
        # Aha moment while reading. Looking-down eyes + sparkles.
        wx = _WIZ_X_MIN
        sprite = _PX_WIZARD_READING

    _stamp_pixel(canvas, wx, _WIZ_Y, sprite)

    # --- sparkles ONLY on realization phases ---
    if phase in (_PHASE_REALIZE_SHELF, _PHASE_REALIZE_DESK):
        for dx, dy, ch in _PX_SPARKLES:
            cy = _WIZ_Y + dy
            cx = wx + dx
            if 0 <= cy < PX_H and 0 <= cx < PX_W:
                canvas[cy][cx] = ch

    return _pixels_to_text_lines(canvas)


# ---------- end pixel-art scene ----------


def _render_reading_frame(frame: int) -> list[str]:
    """Render frame `n` of the wizard-reading-at-desk scene.

    The wizard sits at his desk facing the open book (mirrored to face
    left), eyes shifting between reading, blinking, squinting, and an
    occasional "aha!" with sparkles over his hat. The bookshelf is
    still in the background. No walking, no carrying — this is a focused
    reading-each-passage scene appropriate for the cite_finder loop.
    """
    canvas = _new_canvas()
    _draw_floor(canvas)
    _draw_desk(canvas, "open")
    _draw_shelf(canvas, missing=None)

    eye_state, has_sparkle = _READING_FRAMES[frame % len(_READING_FRAMES)]
    eyes = _READING_EYES[eye_state]

    sprite = [
        r"   _   ",
        r"  /*\  ",
        f"  {eyes}  ",
        r" /|_|\ ",
        r"//   \\",
        r"\_____/",
    ]
    # Wizard faces left (toward the book on the desk surface).
    sprite = _mirror_sprite(sprite)

    wizard_top_y = FLOOR_Y - WIZARD_H
    _stamp(canvas, WALK_X_MIN, wizard_top_y, sprite)

    if has_sparkle:
        _draw_sparkles(canvas, WALK_X_MIN, wizard_top_y)

    return ["".join(row) for row in canvas]


# ---------- live display ----------

@dataclass
class _State:
    """Mutable display state shared between the Live thread and the caller."""

    status: str = "Thinking..."
    done: int = 0
    total: int | None = None
    start: float = 0.0


# Scene renderers all return list[Text] — each line is a pre-styled
# Rich Text. The ASCII scenes (walking, reading) wrap their lines in a
# uniform green style; the pixel scene returns lines with per-character
# colors via half-block characters + ANSI background colors.

def _green_lines(plain_lines: list[str]) -> list[Text]:
    """Wrap a list of plain strings as green-styled Text lines for the
    retro CRT-terminal look used by the ASCII scenes."""
    return [Text(line, style="green") for line in plain_lines]


_SCENE_RENDERERS: dict = {
    "walking": lambda n: _green_lines(_render_frame(n)),
    "reading": lambda n: _green_lines(_render_reading_frame(n)),
    # "pixel" is registered below, after _render_pixel_frame is defined.
}


class _SceneRenderable:
    """Rich renderable: picks an animation frame from elapsed time and
    yields the scene + status + progress lines."""

    def __init__(
        self,
        state: _State,
        frame_seconds: float = 0.18,
        scene: str = "walking",
    ):
        self.state = state
        # 0.18s per frame = ~5.5 FPS. Smooth enough for walking, not jittery.
        # For the reading scene a slightly slower cadence reads better, so
        # callers can override frame_seconds.
        self.frame_seconds = frame_seconds
        self._render_fn = _SCENE_RENDERERS.get(
            scene, _SCENE_RENDERERS["walking"]
        )

    def __rich_console__(
        self, console: Console, options: ConsoleOptions,
    ) -> RenderResult:
        elapsed = time.monotonic() - self.state.start
        frame_idx = int(elapsed / self.frame_seconds)
        scene_lines = self._render_fn(frame_idx)
        for line in scene_lines:
            yield line
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
    """Context manager wrapping Rich's Live with the wizard scene animation.

    Args:
      console: the Rich console to render into.
      status: initial status line text.
      total: total work units (e.g. number of candidate passages) for the
        progress bar. Pass None for an indeterminate spinner mode.
      scene: which animation to use. "walking" (default) shows the wizard
        walking between desk and shelf; "reading" parks him at the desk
        with an open book and animates eye/page state. Pick based on what
        the operation actually does — `cite` reads passages, so it uses
        "reading"; `find` and chat use "walking".
    """

    def __init__(
        self,
        console: Console,
        *,
        status: str = "Thinking...",
        total: int | None = None,
        scene: str | None = None,
    ) -> None:
        self.state = _State(status=status, total=total)
        # Resolve scene: explicit arg wins, otherwise fall back to the
        # WIZARD_SCENE setting (defaults to "walking").
        if scene is None:
            from bibwizard.utils.config import settings as _settings
            scene = getattr(_settings, "wizard_scene", "walking") or "walking"
        # Slightly slower frame cadence for the reading scene so the
        # blink / aha cycle reads as deliberate rather than twitchy.
        # Pixel mode keeps the snappy ~5 fps cadence since the walking
        # animation looks best at that speed.
        frame_seconds = 0.4 if scene == "reading" else 0.18
        self._renderable = _SceneRenderable(
            self.state, frame_seconds=frame_seconds, scene=scene,
        )
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


# Register the pixel scene now that _render_pixel_frame is defined.
# Pixel frames return Text objects with their own colors, so they don't
# get wrapped in the green-styling like the ASCII scenes.
_SCENE_RENDERERS["pixel"] = _render_pixel_frame
