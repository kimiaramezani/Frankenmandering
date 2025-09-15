# opinions.py — generate initial opinion fields for the grid
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass

@dataclass
class OpinionFiller:
    mode: str = "hbo"                 # "hbo" | "iid-beta" | "constant" | "blobs"
    seed: int = 42                    # random seed

    # iid-beta / HBO params
    alpha: float = 2.0
    beta:  float = 2.0

    # HBO clustering controls
    influence: float = 0.8            # ρ in [0,1] (blend toward neighbor mean)

    # constant mode
    const: float = 0.5

    # blobs mode
    blobs_k: int = 5
    blobs_sigma: float = 2.0

    # ---- construction helpers ----
    @classmethod
    def from_config(cls, cfg: dict | None) -> "OpinionFiller":
        cfg = cfg or {}
        # allow both camel and snake keys if they show up
        return cls(
            mode       = cfg.get("mode", "hbo"),
            seed       = int(cfg.get("seed", 202)),
            alpha      = float(cfg.get("alpha", 2.0)),
            beta       = float(cfg.get("beta", 2.0)),
            influence  = float(cfg.get("influence", 0.8)),
            const      = float(cfg.get("const", 0.5)),
            blobs_k    = int(cfg.get("k", cfg.get("blobs_k", 5))),
            blobs_sigma= float(cfg.get("sigma", cfg.get("blobs_sigma", 2.0))),
        )

    # ---- public API ----
    def apply(self, nodes_df: pd.DataFrame, H: int, W: int, inplace: bool = True) -> pd.DataFrame:
        """
        Fills nodes_df['opinion'] in-place (default) with values in [0,1].
        Returns nodes_df.
        """
        df = nodes_df if inplace else nodes_df.copy()
        if self.mode == "constant":
            vals = np.full(H*W, self.const, dtype=np.float32)

        elif self.mode == "iid-beta":
            rng = np.random.default_rng(self.seed)
            vals = rng.beta(self.alpha, self.beta, size=H*W).astype(np.float32)

        elif self.mode == "blobs":
            rng = np.random.default_rng(self.seed)
            xs = df["x"].to_numpy()
            ys = df["y"].to_numpy()
            field = np.zeros(H*W, dtype=np.float64)
            centers = [(int(rng.integers(0, W)), int(rng.integers(0, H))) for _ in range(self.blobs_k)]
            s2 = float(self.blobs_sigma) ** 2
            for cx, cy in centers:
                field += np.exp(-((xs - cx)**2 + (ys - cy)**2) / (2.0 * s2))
            # normalize to [0,1] then squash to avoid extremes
            field = (field - field.min()) / max(1e-8, (field.max() - field.min()))
            vals = (1.0 / (1.0 + np.exp(-4.0 * (field - 0.5)))).astype(np.float32)

        elif self.mode == "hbo":
            vals = self._fill_hbo(df, H, W)
        else:
            raise ValueError(f"Unknown opinions mode: {self.mode}")

        df["opinion"] = vals.astype(np.float32)
        return df

    # ---- internals ----
    def _fill_hbo(self, df: pd.DataFrame, H: int, W: int) -> np.ndarray:
        """
        HBO fill with paper-style traversal:
        - At each step, pick a cell uniformly at random from ALL unfilled cells U.
        - If it has filled rook-neighbors, compute v̄ (their mean), tilt Beta toward v̄,
            draw X, then set v = (1-ρ)X + ρ v̄. If no filled neighbors, draw Beta(α,β).
        - Repeat until every cell is assigned. (No BFS queue.)
        """
        rng = np.random.default_rng(self.seed)
        N = H * W
        vals   = np.full(N, np.nan, dtype=np.float32)
        filled = np.zeros((H, W), dtype=bool)

        # Helper: 1D id <-> (x,y)
        def id_to_xy(i: int):  # id = y*W + x
            y, x = divmod(i, W)
            return x, y

        # Keep an array of all unfilled ids and shrink it in O(1) each draw
        unfilled = np.arange(N, dtype=np.int32)
        front = N  # active prefix length

        while front > 0:
            # Choose an index r uniformly from [0, front)
            r = int(rng.integers(0, front))
            i = int(unfilled[r])
            # Remove i from U by swapping with last active and shrinking 'front'
            unfilled[r], unfilled[front-1] = unfilled[front-1], unfilled[r]
            front -= 1

            x, y = id_to_xy(i)
            if filled[y, x]:
                continue  # (shouldn't happen, but safe)

            # Collect already-FILLED rook neighbors' opinions
            neigh = []
            if x > 0   and filled[y, x-1]: neigh.append(vals[i - 1])
            if x < W-1 and filled[y, x+1]: neigh.append(vals[i + 1])
            if y > 0   and filled[y-1, x]: neigh.append(vals[i - W])
            if y < H-1 and filled[y+1, x]: neigh.append(vals[i + W])

            if not neigh:
                v = rng.beta(self.alpha, self.beta)
            else:
                vbar = float(np.mean(neigh))
                if vbar > 0.5:
                    a = 2 * (1.0 + self.influence)
                    b = 2 * (1.0 - self.influence)
                else:
                    a = 2 * (1.0 - self.influence)
                    b = 2 * (1.0 + self.influence)
                draw = rng.beta(a, b)
                v = (1.0 - self.influence) * draw + self.influence * vbar

            vals[i] = np.float32(np.clip(v, 0.0, 1.0))
            filled[y, x] = True

        return vals

