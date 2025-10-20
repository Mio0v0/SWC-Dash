from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from constants import SWC_COLS, TYPE_LABEL, DEFAULT_COLORS, label_for_type, color_for_type

def children_lists(arr: np.ndarray) -> List[List[int]]:
    id2idx: Dict[int, int] = {}
    for i, n in enumerate(arr):
        try:
            id2idx[int(n[0])] = i
        except Exception:
            pass
    kids: List[List[int]] = [[] for _ in range(len(arr))]
    for i in range(len(arr)):
        try:
            p = int(arr[i][6])
        except Exception:
            p = -1
        if p == -1:
            continue
        j = id2idx.get(p)
        if j is None or j == i:
            continue
        kids[j].append(i)
    return kids

def edge_length(arr: np.ndarray, i: int, j: int) -> float:
    dx = float(arr[i][2]) - float(arr[j][2])
    dy = float(arr[i][3]) - float(arr[j][3])
    dz = float(arr[i][4]) - float(arr[j][4])
    return float(np.sqrt(dx*dx + dy*dy + dz*dz))

def subtree_nodes(kids: List[List[int]], root: int) -> List[int]:
    stack = [root]
    out: List[int] = []
    seen = set()
    while stack:
        u = stack.pop()
        if u in seen:
            continue
        seen.add(u)
        out.append(u)
        stack.extend(kids[u])
    return out

def pick_root(arr: np.ndarray, kids: List[List[int]]) -> int:
    roots = []
    for i in range(len(arr)):
        try:
            if int(arr[i][6]) == -1:
                roots.append(i)
        except Exception:
            pass
    soma_roots = []
    for i in roots:
        try:
            if int(arr[i][1]) == 1:
                soma_roots.append(i)
        except Exception:
            pass
    if soma_roots:
        return soma_roots[0]
    return roots[0] if roots else 0

def cumlens_from_root(arr: np.ndarray, kids: List[List[int]], root: int) -> List[float]:
    cum = [0.0] * len(arr)
    stack = [root]
    seen = {root}
    while stack:
        u = stack.pop()
        for v in kids[u]:
            if v not in seen:
                cum[v] = cum[u] + edge_length(arr, v, u)
                seen.add(v)
                stack.append(v)
    return cum

def layout_y_positions(kids: List[List[int]], root: int) -> List[float]:
    N = len(kids)
    y = [0.0] * N
    cursor = 0
    stack: List[Tuple[int, int]] = [(root, 0)]
    while stack:
        u, state = stack.pop()
        if state == 0:
            if not kids[u]:
                y[u] = float(cursor)
                cursor += 1
            else:
                stack.append((u, 1))
                for v in reversed(kids[u]):
                    stack.append((v, 0))
        else:
            ch = kids[u]
            if ch:
                y[u] = float(sum(y[v] for v in ch) / len(ch))
    return y
