# Dots Cordon

Two players take turns placing dots on a rectangular grid. **Offense and defense are not fixed roles** — they swap every turn. On your move, you are offense: you try to surround your opponent’s dots with a closed **cordon**. On your opponent’s move, they are offense and you are defense.

When a cordon closes, trapped enemy dots are removed, the enclosed area leaves play, and the player who closed the cordon scores points equal to the number of enemy dots captured.

---

## The board

- The board is a grid of **intersections** with a fixed **width** and **height**.
- Each intersection holds at most one dot.
- Intersections on the outer edge of the grid are **border** intersections.

---

## Players

Each player has their own dot color (for example, blue and red). Both players follow the same rules; only the **active player** changes each turn.

| On your turn | Goal |
|--------------|------|
| **Offense** (you) | Surround enemy dots with a closed cordon |
| **Defense** (opponent) | Their dots must stay alive and reachable |

---

## Turn order

1. One player places one dot.
2. The other player places one dot.
3. Repeat.

The player who moved first may have **at most one more dot** on the board than their opponent at any time.

After you move, you are defense until your next turn. After your opponent moves, they are defense until their next turn.

---

## Connectivity

Two dots belonging to the **same player** are **connected** if they occupy adjacent intersections on the same row, the same column, or a diagonal. In other words, they are neighbors in all eight directions.

Connected dots form a **group**. Groups can grow as new dots are placed.

```
Connected — neighbors (share an edge or corner):

  •—•          •   •          •
               |   |         /
               •   •        •

Not connected — empty cells between them:

  •   •        • . •
  ↑              ↑
  same row,      same row,
  2 gaps apart   1 gap apart
  (too far)      (still too far)
```

Two dots are connected only if they sit on **neighboring** intersections. Dots on the same row or column with empty cells between them are **not** connected, no matter how “lined up” they look.

---

## Movement and escape

When checking whether dots are trapped, imagine walking from intersection to intersection:

- You may step **horizontally or vertically** only (not diagonally).
- You may pass through **empty** intersections and **friendly** dots (same player).
- **Enemy** dots block the path completely.

An enemy dot (or group of enemy dots) **escapes** if it can reach any border intersection by such a path. If it cannot, it is **trapped**.

```
Diagonal neighbors do NOT link for escape:

  B         Two enemy dots are diagonal
  .         neighbors — they are NOT
  B (safe)  considered connected for escape.

  B         Two enemy dots share a row
  B (safe)  with an empty cell between — escape
              is possible.
```

(In these diagrams, `B` marks the opponent’s dots.)

---

## Cordon

A **cordon** is a **connected group of your dots** that seals off a region of the board so that at least one **enemy** dot inside that region **cannot escape**.

When you place a dot that completes one or more cordons:

1. Every trapped enemy dot in each sealed region is **captured** (removed from play).
2. The sealed region — including empty intersections inside it — is marked in **your** color and becomes **dead territory**. Neither player may place new dots there.
3. **You** score points equal to the number of enemy dots you captured on that move (see Scoring).

Your opponent can close a cordon around your dots on their turn under the same rules. Either player can score; whoever closes the cordon earns the points.

A single move may close **more than one** cordon at once if it separates one large interior into several sealed regions in the same turn. Each sealed region is resolved separately.

```
One new dot can close two regions at once:

        A               A
    A       A       A       A
      B                   B
      B       →           A  ← your new dot
      A                   B
    A       A       A       B
        A               A

Before: upper and lower enemy dots connect through
        the gap in the middle.

After:  two separate sealed regions; both enemy
        groups are trapped. You score for all of them.
```

(`A` = your dots, `B` = enemy dots.)

---

## What counts as “closed”

A cordon is **closed** when the enemy dots inside a region have **no escape path** to the border. Your dots on the boundary do not need to sit on every side of a single empty cell; they must collectively block all horizontal/vertical routes outward.

**Minimal example** — four of your dots around one enemy dot:

```
    A
  A B A
    A
```

The enemy dot at the center cannot reach the border. You have closed a cordon and capture that dot.

**Not closed** — gaps allow escape:

```
A . A
. B .
A . A
```

Your dots sit on the corners, but the enemy dot reaches the outside through gaps (moving horizontally and vertically through empty cells).

**Closed ring** — two enemy dots trapped together:

```
    A
  A B A
  A B A
    A
```

Both enemy dots are in the same sealed region. When the cordon closes, you capture both.

---

## Dead territory and nested cordons

Captured regions stay on the board as dead territory. They act like walls: paths cannot pass through them, and no new dots may be placed inside them.

A later cordon — yours or your opponent’s — may surround dead territory along with live enemy dots. The outer cordon can be larger than an older one inside it.

```
Later, a larger cordon may wrap an older one:

    A A A A A
    A ~ ~ ~ A      ~ = dead territory from an
    A ~ B ~ A           earlier capture
    A ~ ~ ~ A
    A A A A A
```

---

## Scoring

- You earn points when **you** close a cordon on your turn.
- Points gained on a turn = **the number of enemy dots you captured that turn**.
- If one move closes two sealed regions and captures 2 enemy dots in one and 1 in the other, you score **3** for that turn.
- Your opponent scores the same way on their turns. Both players add to their own totals.

---

## End of game

The game ends by agreement or when a fixed number of turns has been played (house rule). The player with the **higher score** wins.

If players continue until the board is full of live dots and dead territory, cordons are resolved as usual whenever they close.

---

## Summary

| Concept | Rule |
|---------|------|
| Roles | Swap each turn — active player is offense, other is defense |
| Placement | Alternate turns; first player may lead by at most one dot |
| Same-player connectivity | 8 directions (row, column, diagonal) |
| Escape | Horizontal/vertical paths through empty and friendly dots; enemy dots block |
| Cordon | Your connected group seals a region containing enemy dots |
| Capture | Trapped enemy dots removed when a cordon closes |
| Dead territory | Sealed interior cannot receive new dots |
| Score | Closing player gains 1 point per enemy dot captured on that turn |
