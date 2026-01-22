# radiation/gpr.py
# ガウス過程回帰（GPR）による線源追跡
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


def fit_gpr(cumulative_data, grid_width, grid_height,
            predict_scale=3.0, predict_margin_cells=3.0, gpr_fine=100):
    """
    累積測定データに対してGPRをフィッティングし、ピーク位置を予測する

    Parameters
    ----------
    cumulative_data : dict
        (x, y) をキー、測定値を値とする辞書
    grid_width : int
        グリッドの幅
    grid_height : int
        グリッドの高さ
    predict_scale : float
        予測範囲の拡大倍率
    predict_margin_cells : float
        予測範囲のマージン（セル数）
    gpr_fine : int
        予測グリッドの解像度

    Returns
    -------
    dict or None
        GPR予測結果を含む辞書、データ不足の場合はNone
    """
    if len(cumulative_data) == 0:
        return None

    pts = list(cumulative_data.items())
    X = np.array([[k[0], k[1]] for k, v in pts], dtype=float)
    ydata = np.array([v for k, v in pts], dtype=float)

    kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 1e1))
    gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-2, normalize_y=True)
    gpr.fit(X, ydata)

    min_x, max_x = X[:, 0].min(), X[:, 0].max()
    min_y, max_y = X[:, 1].min(), X[:, 1].max()
    width_x = max_x - min_x if max_x > min_x else 1.0
    width_y = max_y - min_y if max_y > min_y else 1.0
    ext_x = width_x * (predict_scale - 1.0) / 2.0 + predict_margin_cells
    ext_y = width_y * (predict_scale - 1.0) / 2.0 + predict_margin_cells
    pred_min_x = max(min_x - ext_x, 0.0)
    pred_max_x = min(max_x + ext_x, grid_width - 1.0)
    pred_min_y = max(min_y - ext_y, 0.0)
    pred_max_y = min(max_y + ext_y, grid_height - 1.0)

    xi = np.linspace(pred_min_x, pred_max_x, gpr_fine)
    yi = np.linspace(pred_min_y, pred_max_y, gpr_fine)
    Xi, Yi = np.meshgrid(xi, yi)
    Xpred = np.vstack([Xi.ravel(), Yi.ravel()]).T
    Zi_flat, sigma = gpr.predict(Xpred, return_std=True)
    Zi = Zi_flat.reshape(Xi.shape)

    max_idx = np.unravel_index(np.nanargmax(Zi), Zi.shape)
    peak_x = Xi[max_idx]
    peak_y = Yi[max_idx]
    peak_val = Zi[max_idx]

    return {
        "Xi": Xi, "Yi": Yi, "Zi": Zi,
        "pred_min_x": pred_min_x, "pred_max_x": pred_max_x,
        "pred_min_y": pred_min_y, "pred_max_y": pred_max_y,
        "peak_x": float(peak_x), "peak_y": float(peak_y), "peak_val": float(peak_val)
    }


class GPRExplorer:
    """GPR探査を管理するクラス"""

    def __init__(self, search_radius=1, max_iters=8,
                 predict_scale=3.0, predict_margin_cells=3.0):
        """
        Parameters
        ----------
        search_radius : int
            探査半径（セル数）
        max_iters : int
            最大反復回数
        predict_scale : float
            予測範囲の拡大倍率
        predict_margin_cells : float
            予測範囲のマージン
        """
        self.search_radius = search_radius
        self.max_iters = max_iters
        self.predict_scale = predict_scale
        self.predict_margin_cells = predict_margin_cells

    def select_measurement_cells(self, center, grid_width, grid_height,
                                 obstacle_grid, cumulative_data):
        """
        測定対象セルを選択する

        Parameters
        ----------
        center : tuple
            探査中心座標 (cx, cy)
        grid_width : int
            グリッドの幅
        grid_height : int
            グリッドの高さ
        obstacle_grid : ndarray
            障害物グリッド
        cumulative_data : dict
            既存の測定データ

        Returns
        -------
        list
            測定対象セルのリスト [(x, y), ...]
        """
        cx, cy = center
        targets = []
        for dx in range(-self.search_radius, self.search_radius + 1):
            for dy in range(-self.search_radius, self.search_radius + 1):
                tx, ty = cx + dx, cy + dy
                if 0 <= tx < grid_width and 0 <= ty < grid_height:
                    if obstacle_grid[ty, tx] == 1:
                        continue
                    if (tx, ty) not in cumulative_data:
                        targets.append((tx, ty))
        return targets

    def should_continue(self, iteration, targets_remaining):
        """
        探査を継続するかどうかを判定する

        Parameters
        ----------
        iteration : int
            現在の反復回数
        targets_remaining : int
            残りの測定対象数

        Returns
        -------
        bool
            継続する場合True
        """
        return iteration < self.max_iters and targets_remaining > 0

    def update_center(self, gpr_pred, grid_width, grid_height, obstacle_grid):
        """
        GPR予測結果に基づいて探査中心を更新する

        Parameters
        ----------
        gpr_pred : dict
            GPR予測結果
        grid_width : int
            グリッドの幅
        grid_height : int
            グリッドの高さ
        obstacle_grid : ndarray
            障害物グリッド

        Returns
        -------
        tuple or None
            新しい中心座標、更新不可の場合None
        """
        if gpr_pred is None:
            return None

        new_cx = int(np.round(gpr_pred['peak_x']))
        new_cy = int(np.round(gpr_pred['peak_y']))

        if 0 <= new_cx < grid_width and 0 <= new_cy < grid_height:
            if obstacle_grid[new_cy, new_cx] == 0:
                return (new_cx, new_cy)
        return None
