from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple, Union
import numpy as np
import pandas as pd

from constants import SWC_COLS


@dataclass
class TreeCache:
    ids: np.ndarray
    types: np.ndarray
    xyz: np.ndarray
    radius: np.ndarray
    parent_ids: np.ndarray
    parent_index: np.ndarray
    child_offsets: np.ndarray
    child_indices: np.ndarray
    edge_lengths: np.ndarray

    @property
    def size(self) -> int:
        return int(self.ids.shape[0])

    def iter_children(self, u: int) -> np.ndarray:
        start = int(self.child_offsets[u])
        end = int(self.child_offsets[u + 1])
        return self.child_indices[start:end]


def build_tree_cache(df: pd.DataFrame) -> TreeCache:
    if df is None or df.empty:
        empty_f = np.empty(0, dtype=np.float32)
        empty_i = np.empty(0, dtype=np.int32)
        return TreeCache(
            ids=empty_i.copy(),
            types=empty_i.copy(),
            xyz=np.empty((0, 3), dtype=np.float32),
            radius=empty_f.copy(),
            parent_ids=empty_i.copy(),
            parent_index=empty_i.copy(),
            child_offsets=np.zeros(1, dtype=np.int32),
            child_indices=empty_i.copy(),
            edge_lengths=empty_f.copy(),
        )

    cols = df[SWC_COLS]
    ids = cols["id"].to_numpy(dtype=np.int64, copy=False)
    types = cols["type"].to_numpy(dtype=np.int32, copy=False)
    xyz = cols[["x", "y", "z"]].to_numpy(dtype=np.float32, copy=False)
    radius = cols["radius"].to_numpy(dtype=np.float32, copy=False)
    parent_ids = cols["parent"].to_numpy(dtype=np.int64, copy=False)

    n = int(ids.shape[0])
    parent_index = np.full(n, -1, dtype=np.int32)
    id2idx: Dict[int, int] = {int(ids[i]): i for i in range(n)}

    for i in range(n):
        pid = int(parent_ids[i])
        if pid < 0:
            continue
        parent_index[i] = id2idx.get(pid, -1)

    counts = np.zeros(n, dtype=np.int32)
    valid_children = parent_index >= 0
    if np.any(valid_children):
        np.add.at(counts, parent_index[valid_children], 1)
    offsets = np.empty(n + 1, dtype=np.int32)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])

    child_indices = np.empty(int(valid_children.sum()), dtype=np.int32)
    cursor = offsets[:-1].copy()
    for child in np.nonzero(valid_children)[0]:
        parent = parent_index[child]
        pos = cursor[parent]
        child_indices[pos] = int(child)
        cursor[parent] += 1

    edge_lengths = np.zeros(n, dtype=np.float32)
    if np.any(valid_children):
        parent_points = xyz[parent_index[valid_children]]
        child_points = xyz[valid_children]
        edge_lengths[valid_children] = np.linalg.norm(child_points - parent_points, axis=1).astype(np.float32)

    return TreeCache(
        ids=ids.astype(np.int64, copy=False),
        types=types,
        xyz=xyz,
        radius=radius,
        parent_ids=parent_ids.astype(np.int64, copy=False),
        parent_index=parent_index,
        child_offsets=offsets,
        child_indices=child_indices,
        edge_lengths=edge_lengths,
    )


def pick_root_from_cache(cache: TreeCache) -> int:
    if cache.size == 0:
        return 0
    roots = np.flatnonzero(cache.parent_index < 0)
    if roots.size == 0:
        return 0
    soma_mask = cache.types[roots] == 1
    if np.any(soma_mask):
        return int(roots[np.argmax(soma_mask)])
    return int(roots[0])


def cumlens_from_root_cache(cache: TreeCache, root: int) -> np.ndarray:
    cum = np.zeros(cache.size, dtype=np.float32)
    if cache.size == 0:
        return cum
    stack = [int(root)]
    while stack:
        u = stack.pop()
        start = int(cache.child_offsets[u])
        end = int(cache.child_offsets[u + 1])
        if start == end:
            continue
        children = cache.child_indices[start:end]
        cum_children = cum[u] + cache.edge_lengths[children]
        cum[children] = cum_children
        stack.extend(children.tolist())
    return cum


def layout_y_positions_cache(cache: TreeCache, root: int) -> np.ndarray:
    y = np.zeros(cache.size, dtype=np.float32)
    cursor = 0.0
    stack: List[Tuple[int, int]] = [(int(root), 0)]
    while stack:
        u, state = stack.pop()
        start = int(cache.child_offsets[u])
        end = int(cache.child_offsets[u + 1])
        if state == 0:
            if start == end:
                y[u] = float(cursor)
                cursor += 1.0
            else:
                stack.append((u, 1))
                children = cache.child_indices[start:end]
                for v in reversed(children.tolist()):
                    stack.append((v, 0))
        else:
            if start != end:
                children = cache.child_indices[start:end]
                y[u] = float(np.mean(y[children]))
    return y


def children_payload(cache: TreeCache) -> Dict[str, List[int]]:
    return {
        "offsets": cache.child_offsets.astype(int).tolist(),
        "indices": cache.child_indices.astype(int).tolist(),
    }

def subtree_nodes(kids: Union[List[List[int]], Dict[str, Sequence[int]]], root: int) -> List[int]:
    stack = [int(root)]
    out: List[int] = []
    seen = set()

    if isinstance(kids, dict) and "offsets" in kids and "indices" in kids:
        offsets_seq = kids["offsets"]
        indices_seq = kids["indices"]
        while stack:
            u = int(stack.pop())
            if u in seen:
                continue
            seen.add(u)
            out.append(u)
            start = int(offsets_seq[u])
            end = int(offsets_seq[u + 1])
            if start < end:
                stack.extend(indices_seq[start:end])
        return out

    while stack:
        u = stack.pop()
        if u in seen:
            continue
        seen.add(u)
        out.append(u)
        stack.extend(kids[u])
    return out
