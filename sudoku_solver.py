import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"

class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c1 = nn.Conv2d(c, c, 3, padding=1); self.b1 = nn.BatchNorm2d(c)
        self.c2 = nn.Conv2d(c, c, 3, padding=1); self.b2 = nn.BatchNorm2d(c)
    def forward(self, x):
        h = F.relu(self.b1(self.c1(x)))
        h = self.b2(self.c2(h))
        return F.relu(x + h)

class SudokuNet(nn.Module):
    def __init__(self, c=128, n_blocks=8):
        super().__init__()
        self.inp = nn.Conv2d(10, c, 3, padding=1)
        self.blocks = nn.Sequential(*[Block(c) for _ in range(n_blocks)])
        self.out = nn.Conv2d(c, 9, 1)
    def forward(self, x):
        return self.out(self.blocks(F.relu(self.inp(x))))

def load_model(path="sudoku_net.pt"):
    m = SudokuNet().to(device)
    m.load_state_dict(torch.load(path, map_location=device))
    return m.eval()

# ---------- rule checks ----------
def _has_dup(vals):
    vals = [int(v) for v in vals if v > 0]
    return len(vals) != len(set(vals))

def validate(g):
    """Error message if the given digits already break Sudoku rules, else None."""
    for i in range(9):
        if _has_dup(g[i]):    return f"Row {i+1} has a repeated digit."
        if _has_dup(g[:, i]): return f"Column {i+1} has a repeated digit."
    for br in range(0, 9, 3):
        for bc in range(0, 9, 3):
            if _has_dup(g[br:br+3, bc:bc+3].ravel()):
                return f"The box at row {br+1}, column {bc+1} has a repeated digit."
    return None

def allowed(b, r, c, d):
    if (b[r] == d).any() or (b[:, c] == d).any():
        return False
    br, bc = 3 * (r // 3), 3 * (c // 3)
    return not (b[br:br+3, bc:bc+3] == d).any()

def no_dead_cells(b):
    """False if some blank cell has no legal digit left."""
    for r in range(9):
        for c in range(9):
            if b[r, c] == 0:
                used = set(b[r].tolist()) | set(b[:, c].tolist()) | \
                       set(b[3*(r//3):3*(r//3)+3, 3*(c//3):3*(c//3)+3].ravel().tolist())
                if len(used - {0}) == 9:
                    return False
    return True

# ---------- model-guided search ----------
@torch.no_grad()
def _search(model, b, budget, path, topk):
    if budget[0] <= 0:
        return None
    blank = (b == 0)
    if not blank.any():
        return b
    inp = F.one_hot(torch.from_numpy(b).to(device), 10).permute(2, 0, 1).float()[None]
    probs = model(inp)[0].softmax(0).cpu().numpy()
    conf = np.where(blank, probs.max(0), -1)
    r, c = divmod(int(conf.argmax()), 9)              # most confident blank cell
    for d in np.argsort(-probs[:, r, c])[:topk] + 1:  # its top-k digits
        d = int(d)
        if not allowed(b, r, c, d):
            continue
        nb = b.copy(); nb[r, c] = d
        if not no_dead_cells(nb):
            continue
        budget[0] -= 1
        path.append((r, c, d))
        res = _search(model, nb, budget, path, topk)
        if res is not None:
            return res
        path.pop()                                    # dead end: undo and try next digit
    return None

def solve_with_steps(model, grid, budget=2000, topk=4):
    """Returns {'solution', 'path', 'tries'} or None. path = ordered (row, col, digit) moves."""
    b = np.array(grid, dtype=np.int64).reshape(9, 9)
    path, bud = [], [budget]
    sol = _search(model, b, bud, path, topk)
    if sol is None:
        return None
    return {"solution": sol, "path": path, "tries": budget - bud[0]}
