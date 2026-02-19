import heapq
import math
from collections import deque

# 8-puzzle state: a tuple of length 9 in row-major order; 0 = blank
GOAL = (1, 2, 3,
        4, 5, 6,
        7, 8, 0)

GOAL_POS = {v: i for i, v in enumerate(GOAL)}

def board_str(s):
    rows = [s[i:i+3] for i in range(0, 9, 3)]
    def fmt(x): return "_" if x == 0 else str(x)
    return "\n".join(" ".join(fmt(x) for x in r) for r in rows)

MOVES = {"U": -3, "D":  3, "L": -1, "R":  1}
INV   = {"U": "D", "D": "U", "L": "R", "R": "L"}

def neighbors(state):
    z = state.index(0)
    x, y = z % 3, z // 3
    for act, d in MOVES.items():
        nz = z + d
        if act == "U" and y == 0: continue
        if act == "D" and y == 2: continue
        if act == "L" and x == 0: continue
        if act == "R" and x == 2: continue
        lst = list(state)
        lst[z], lst[nz] = lst[nz], lst[z]
        yield act, tuple(lst)

def manhattan(state):
    d = 0
    for idx, v in enumerate(state):
        if v == 0:  # ignore blank
            continue
        gi = GOAL_POS[v]
        x, y = idx % 3, idx // 3
        gx, gy = gi % 3, gi // 3
        d += abs(x - gx) + abs(y - gy)
    return d

# Inadmissible heuristic (sometimes overestimates):
def h_over(state):
    return 2 * manhattan(state)

def bfs_shortest(start):
    """Exact shortest path cost for small demonstrations (guaranteed optimal)."""
    q = deque([start])
    parent = {start: (None, None)}
    while q:
        s = q.popleft()
        if s == GOAL:
            path = []
            while parent[s][0] is not None:
                s, act = parent[s]
                path.append(act)
            return list(reversed(path))
        for act, ns in neighbors(s):
            if ns not in parent:
                parent[ns] = (s, act)
                q.append(ns)
    return None

def astar(start, hfunc, tie_break="low_h"):
    """
    A* (or best-first with f=g+h) that stops when GOAL is popped.
    tie_break:
      - "low_h": among equal f, prefer smaller h (more greedy)
      - "high_g": among equal f, prefer larger g (deeper)
    """
    def tiebreak_values(s, g):
        h = hfunc(s)
        if tie_break == "low_h":
            return (h, -g)
        if tie_break == "high_g":
            return (-g, h)
        return (0, 0)

    g = {start: 0}
    parent = {start: (None, None)}
    closed = set()

    heap = []
    counter = 0
    h0 = hfunc(start)
    t1, t2 = tiebreak_values(start, 0)
    heapq.heappush(heap, (0 + h0, t1, t2, counter, start))

    while heap:
        f, _, __, ___, s = heapq.heappop(heap)
        if s in closed:
            continue
        if s == GOAL:
            # reconstruct path
            path = []
            cur = s
            while parent[cur][0] is not None:
                prev, act = parent[cur]
                path.append(act)
                cur = prev
            return list(reversed(path))

        closed.add(s)
        gs = g[s]
        for act, ns in neighbors(s):
            ng = gs + 1
            if ng < g.get(ns, math.inf):
                g[ns] = ng
                parent[ns] = (s, act)
                counter += 1
                hn = hfunc(ns)
                t1, t2 = tiebreak_values(ns, ng)
                heapq.heappush(heap, (ng + hn, t1, t2, counter, ns))
    return None

def apply_moves(state, moves):
    s = state
    for act in moves:
        for a, ns in neighbors(s):
            if a == act:
                s = ns
                break
        else:
            raise RuntimeError(f"Illegal move {act} from state:\n{board_str(s)}")
    return s

if __name__ == "__main__":
    # 1) Show "sometimes overestimates" on a trivial near-goal state
    near = (1, 2, 3,
            4, 5, 6,
            7, 0, 8)
    true_path = bfs_shortest(near)
    print("Near-goal state:")
    print(board_str(near))
    print("True optimal steps =", len(true_path))
    print("Manhattan =", manhattan(near))
    print("h_over = 2*Manhattan =", h_over(near), "  (overestimates here)\n")

    # 2) Concrete counterexample where A* with h_over returns a suboptimal solution
    start = (1, 2, 3,
             5, 8, 0,
             4, 6, 7)

    print("Counterexample start:")
    print(board_str(start))
    print("Manhattan(start) =", manhattan(start), " h_over(start) =", h_over(start), "\n")

    # Optimal (use admissible Manhattan => optimal A*)
    opt = astar(start, manhattan, tie_break="low_h")
    # Inadmissible (overestimating) heuristic
    bad = astar(start, h_over, tie_break="low_h")

    print("Optimal A* (h=Manhattan): steps =", len(opt))
    print("Moves:", "".join(opt))
    print("Reached goal?", apply_moves(start, opt) == GOAL, "\n")

    print("A* with overestimating h (h=2*Manhattan): steps =", len(bad))
    print("Moves:", "".join(bad))
    print("Reached goal?", apply_moves(start, bad) == GOAL, "\n")

    print("Suboptimality gap =", len(bad) - len(opt))
