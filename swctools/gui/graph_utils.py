from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple, Union
import numpy as np
import pandas as pd

from .constants import SWC_COLS


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


def compute_levels(cache: TreeCache, root: int) -> np.ndarray:
    """BFS from *root* to assign topological depth (root = level 0)."""
    levels = np.full(cache.size, -1, dtype=np.int32)
    if cache.size == 0:
        return levels
    levels[root] = 0
    stack = [int(root)]
    while stack:
        u = stack.pop()
        start = int(cache.child_offsets[u])
        end = int(cache.child_offsets[u + 1])
        for c in cache.child_indices[start:end].tolist():
            levels[c] = levels[u] + 1
            stack.append(c)
    return levels


def find_all_roots(cache: TreeCache) -> List[int]:
    """
    Return root indices sorted: soma-rooted trees first, then by
    descending subtree size.
    """
    if cache.size == 0:
        return []

    roots = np.flatnonzero(cache.parent_index < 0).tolist()
    if not roots:
        return [0]

    # Compute subtree sizes via BFS from each root
    def _subtree_size(r: int) -> int:
        count = 0
        st = [int(r)]
        while st:
            u = st.pop()
            count += 1
            start = int(cache.child_offsets[u])
            end = int(cache.child_offsets[u + 1])
            st.extend(cache.child_indices[start:end].tolist())
        return count

    info = []
    for r in roots:
        is_soma = int(cache.types[r]) == 1
        size = _subtree_size(r)
        info.append((r, is_soma, size))

    # Sort: soma-rooted first, then by ascending root node id
    info.sort(key=lambda t: (not t[1], cache.ids[t[0]]))
    return [r for r, _, _ in info]


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


def merge_dangling_trees(df: pd.DataFrame) -> pd.DataFrame:
    """Re-parent non-soma tree roots to the nearest soma node.

    Only trees that contain at least one soma node (type == 1) are kept as
    independent trees.  Every other disconnected component ("dangling branch")
    has its root re-parented to the spatially closest soma node.

    Returns a **copy** of *df* with updated ``parent`` values.
    """
    if df is None or df.empty:
        return df

    df = df.copy()

    ids = df["id"].to_numpy()
    types = df["type"].to_numpy()
    parents = df["parent"].to_numpy()
    xyz = df[["x", "y", "z"]].to_numpy(dtype=np.float64)

    id2idx = {int(ids[i]): i for i in range(len(ids))}

    # Build children map (id -> list of child ids)
    children_map: Dict[int, List[int]] = {}
    for i in range(len(ids)):
        pid = int(parents[i])
        if pid >= 0:
            children_map.setdefault(pid, []).append(int(ids[i]))

    # Find all roots
    root_indices = [i for i in range(len(ids)) if int(parents[i]) < 0]

    # Classify each root's tree
    soma_trees: List[Tuple[int, set]] = []     # (root_idx, member_indices)
    dangling_trees: List[Tuple[int, set]] = []  # (root_idx, member_indices)

    for ri in root_indices:
        root_id = int(ids[ri])
        members = set()
        queue = [root_id]
        has_soma = False
        while queue:
            nid = queue.pop(0)
            idx = id2idx.get(nid)
            if idx is None or nid in members:
                continue
            members.add(nid)
            if int(types[idx]) == 1:
                has_soma = True
            for child in children_map.get(nid, []):
                queue.append(child)

        if has_soma:
            soma_trees.append((ri, members))
        else:
            dangling_trees.append((ri, members))

    if not dangling_trees or not soma_trees:
        return df  # nothing to merge

    # Collect all soma node indices (for distance computation)
    soma_node_indices = [i for i in range(len(ids)) if int(types[i]) == 1]
    if not soma_node_indices:
        return df

    soma_xyz = xyz[soma_node_indices]  # (S, 3)
    soma_ids = ids[soma_node_indices]

    # For each dangling tree, find nearest soma and re-parent
    for droot_idx, _members in dangling_trees:
        droot_xyz = xyz[droot_idx]  # (3,)
        dists = np.linalg.norm(soma_xyz - droot_xyz, axis=1)
        nearest = int(np.argmin(dists))
        nearest_soma_id = int(soma_ids[nearest])

        # Update the parent of the dangling root
        df.iloc[droot_idx, df.columns.get_loc("parent")] = nearest_soma_id

    return df
