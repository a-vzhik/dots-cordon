# Dots Cordon

Two players take turns placing dots on a rectangular grid. One player is **offense**; the other is **defense**. Offense tries to surround groups of enemy dots with a closed **cordon**. When a cordon closes, trapped defending dots are removed, the enclosed area leaves play, and offense scores points equal to the number of enemy dots captured.

---

## The board

- The board is a grid of **intersections** with a fixed **width** and **height**.
- Each intersection holds at most one dot.
- Intersections on the outer edge of the grid are **border** intersections.

---

## Pieces

| Role | Dot color (suggested) | Goal |
|------|----------------------|------|
| **Offense** | Blue | Surround defending dots with a closed cordon |
| **Defense** | Red | Keep dots alive and reachable; avoid being surrounded |

---

## Turn order

1. Offense places one dot.
2. Defense places one dot.
3. Repeat.

Because offense moves first, offense may have **at most one more dot** on the board than defense at any time.

---

## Connectivity

Two dots belonging to the **same player** are **connected** if they occupy adjacent intersections on the same row, the same column, or a diagonal. In other words, they are neighbors in all eight directions.

Connected dots form a **group**. Groups can grow as new dots are placed.

```
Connected (8-direction):

  O       O . O       O
  O         O       . O .
            O

Not connected:

  O . O       O   O
              (gap of 2 on a row)
```

---

## Movement and escape

When checking whether defending dots are trapped, imagine walking from intersection to intersection:

- You may step **horizontally or vertically** only (not diagonally).
- You may pass through **empty** intersections and **defending** dots.
- **Offensive** dots block the path completely.

A defending dot (or group of defending dots) **escapes** if it can reach any border intersection by such a path. If it cannot, it is **trapped**.

```
Diagonal neighbors do NOT link for escape:

  D         D and the border dot are diagonal
  .         neighbors — the defending dot is
  D (safe)  NOT considered connected to it
              for escape purposes.

  D         D and the border dot share a row
  D (safe)  with an empty cell between — escape
              is possible.
```

---

## Cordon

A **cordon** is a **connected group of offensive dots** that seals off a region of the board so that at least one defending dot inside that region **cannot escape**.

When offense places a dot that completes one or more cordons:

1. Every defending dot trapped in each sealed region is **captured** (removed from play).
2. The sealed region — including empty intersections inside it — is marked in offense’s color and becomes **dead territory**. Neither player may place new dots there.
3. Offense **scores** points equal to the number of defending dots captured by that closing move (see Scoring).

A single move may close **more than one** cordon at once if it separates one large interior into several sealed regions in the same turn. Each sealed region is resolved separately.

```
One new dot can close two regions at once:

        O               O
    O       O       O       O
      D                   D
      D       →           O  ← new offensive dot
      O                   D
    O       O       O       D
        O               O

Before: upper and lower defending dots connect through
        the gap in the middle.

After:  two separate sealed regions; both defending
        groups are trapped.
```

---

## What counts as “closed”

A cordon is **closed** when the defending dots inside a region have **no escape path** to the border. Offensive dots on the boundary do not need to sit on every side of a single empty cell; they must collectively block all horizontal/vertical routes outward.

**Minimal example** — four offensive dots around one defending dot:

```
    O
  O D O
    O
```

The defending dot at the center cannot reach the border. Offense has closed a cordon and captures that dot.

**Not closed** — gaps allow escape:

```
O . O
. D .
O . O
```

The four offensive dots sit on the corners, but the defending dot reaches the outside through side and diagonal gaps (moving horizontally and vertically through empty cells).

**Closed ring** — two defending dots trapped together:

```
    O
  O D O
  O D O
    O
```

Both defending dots are in the same sealed region. When the cordon closes, both are captured.

---

## Dead territory and nested cordons

Captured regions stay on the board as dead territory. They act like walls: paths cannot pass through them, and no new dots may be placed inside them.

A later cordon may surround dead territory along with live defending dots. The outer cordon can be larger than an older one inside it.

```
Later, a larger cordon may wrap an older one:

    O O O O O
    O ~ ~ ~ O      ~ = dead territory from an
    O ~ D ~ O           earlier capture
    O ~ ~ ~ O
    O O O O O
```

---

## Scoring

- Offense earns points when a cordon closes.
- Points gained on a turn = **the number of defending dots captured that turn**.
- If one move closes two sealed regions and captures 2 defending dots in one and 1 in the other, offense scores **3** for that turn.
- Defense does not score from captures; defense’s objective is to keep dots alive and avoid giving offense large captures.

---

## End of game

The game ends by agreement or when a fixed number of turns has been played (house rule). The player with the **higher score** wins.

If players continue until the board is full of live dots and dead territory, the last cordons are resolved as usual whenever they close.

---

## Summary

| Concept | Rule |
|---------|------|
| Placement | Alternate turns; offense may lead by at most one dot |
| Same-player connectivity | 8 directions (row, column, diagonal) |
| Escape | Horizontal/vertical paths through empty and defending dots; offensive dots block |
| Cordon | Connected offensive group that seals a region containing defending dots |
| Capture | Trapped defending dots removed when cordon closes |
| Dead territory | Sealed interior cannot receive new dots |
| Score | Offense gains 1 point per defending dot captured on that turn |
