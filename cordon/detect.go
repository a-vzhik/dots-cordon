package cordon

// Result describes one cordon closed by mover on the current board.
type Result struct {
	Mover    Player
	Pocket   []Point
	Captured []Point
	Wall     []Point
}

// Detect finds every cordon that mover has closed on the board.
//
// Escape uses 4-connected movement through empty cells and the escaper's live
// dots. Mover's live dots and all dead territory block movement.
func Detect(b *Board, mover Player) []Result {
	if b == nil {
		return nil
	}

	escaper := mover.Opponent()
	outside := b.floodOutside(escaper)
	pockets := b.findPockets(outside, escaper)

	results := make([]Result, 0, len(pockets))
	for _, pocket := range pockets {
		captured := liveDotsIn(pocket, escaper, b)
		if len(captured) == 0 {
			continue
		}

		wall := b.wallDots(pocket, mover)
		if len(wall) == 0 || !eightConnected(wall) {
			continue
		}

		results = append(results, Result{
			Mover:    mover,
			Pocket:   append([]Point(nil), pocket...),
			Captured: captured,
			Wall:     wall,
		})
	}

	return results
}

func (b *Board) floodOutside(escaper Player) []bool {
	outside := make([]bool, len(b.Cells))

	queue := make([]Point, 0, b.Width*2+b.Height*2)
	for x := 0; x < b.Width; x++ {
		for _, y := range []int{0, b.Height - 1} {
			p := Point{x, y}
			if b.isWalkableForEscape(b.Get(p), escaper) {
				outside[b.index(p)] = true
				queue = append(queue, p)
			}
		}
	}
	for y := 1; y < b.Height-1; y++ {
		for _, x := range []int{0, b.Width - 1} {
			p := Point{x, y}
			if b.isWalkableForEscape(b.Get(p), escaper) && !outside[b.index(p)] {
				outside[b.index(p)] = true
				queue = append(queue, p)
			}
		}
	}

	for head := 0; head < len(queue); head++ {
		p := queue[head]
		for _, d := range cardinals {
			n := Point{p.X + d.X, p.Y + d.Y}
			if !b.inBounds(n) {
				continue
			}
			idx := b.index(n)
			if outside[idx] {
				continue
			}
			if !b.isWalkableForEscape(b.Cells[idx], escaper) {
				continue
			}
			outside[idx] = true
			queue = append(queue, n)
		}
	}

	return outside
}

func (b *Board) findPockets(outside []bool, escaper Player) [][]Point {
	seen := make([]bool, len(b.Cells))
	pockets := make([][]Point, 0)

	for y := 0; y < b.Height; y++ {
		for x := 0; x < b.Width; x++ {
			p := Point{x, y}
			idx := b.index(p)
			if seen[idx] || outside[idx] {
				continue
			}
			if !b.isWalkableForEscape(b.Cells[idx], escaper) {
				continue
			}

			pocket := b.floodPocket(p, seen, outside, escaper)
			pockets = append(pockets, pocket)
		}
	}

	return pockets
}

func (b *Board) floodPocket(start Point, seen, outside []bool, escaper Player) []Point {
	queue := []Point{start}
	seen[b.index(start)] = true
	pocket := make([]Point, 0, 8)

	for head := 0; head < len(queue); head++ {
		p := queue[head]
		pocket = append(pocket, p)

		for _, d := range cardinals {
			n := Point{p.X + d.X, p.Y + d.Y}
			if !b.inBounds(n) {
				continue
			}
			idx := b.index(n)
			if seen[idx] || outside[idx] {
				continue
			}
			if !b.isWalkableForEscape(b.Cells[idx], escaper) {
				continue
			}
			seen[idx] = true
			queue = append(queue, n)
		}
	}

	return pocket
}

func liveDotsIn(pocket []Point, player Player, b *Board) []Point {
	want := liveCell(player)
	captured := make([]Point, 0)
	for _, p := range pocket {
		if b.Get(p) == want {
			captured = append(captured, p)
		}
	}
	return captured
}

func (b *Board) wallDots(pocket []Point, mover Player) []Point {
	inPocket := make(map[Point]struct{}, len(pocket))
	for _, p := range pocket {
		inPocket[p] = struct{}{}
	}

	wallSet := make(map[Point]struct{})
	for _, p := range pocket {
		for _, d := range neighbors8 {
			n := Point{p.X + d.X, p.Y + d.Y}
			if !b.inBounds(n) {
				continue
			}
			if _, ok := inPocket[n]; ok {
				continue
			}
			if b.isMoverDot(b.Get(n), mover) {
				wallSet[n] = struct{}{}
			}
		}
	}

	wall := make([]Point, 0, len(wallSet))
	for p := range wallSet {
		wall = append(wall, p)
	}
	return wall
}

func eightConnected(points []Point) bool {
	if len(points) <= 1 {
		return len(points) == 1
	}

	inSet := make(map[Point]struct{}, len(points))
	for _, p := range points {
		inSet[p] = struct{}{}
	}

	visited := make(map[Point]struct{}, len(points))
	queue := []Point{points[0]}
	visited[points[0]] = struct{}{}

	for head := 0; head < len(queue); head++ {
		p := queue[head]
		for _, d := range neighbors8 {
			n := Point{p.X + d.X, p.Y + d.Y}
			if _, ok := inSet[n]; !ok {
				continue
			}
			if _, ok := visited[n]; ok {
				continue
			}
			visited[n] = struct{}{}
			queue = append(queue, n)
		}
	}

	return len(visited) == len(points)
}
