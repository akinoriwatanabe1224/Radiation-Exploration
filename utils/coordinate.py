# utils/coordinate.py
# 座標変換ユーティリティ
import numpy as np


def world_to_grid(point, x_min, x_max, y_min, y_max, grid_size):
    """
    ワールド座標をグリッド座標（セルインデックス）に変換する

    座標 N.5 はセル N の中央を表す。
    例: 座標 4.5 → セル 4（表示時にセル中央 4.5 に描画）

    Parameters
    ----------
    point : tuple or array-like
        ワールド座標 (wx, wy)
    x_min, x_max, y_min, y_max : float
        ワールド座標の範囲
    grid_size : int
        グリッドサイズ

    Returns
    -------
    tuple
        グリッド座標（セルインデックス） (gx, gy)
    """
    wx, wy = point
    # floorを使用してセルインデックスを取得（N.5 → セル N）
    scaled_x = (wx - x_min) / (x_max - x_min) * (grid_size - 1)
    scaled_y = (wy - y_min) / (y_max - y_min) * (grid_size - 1)
    gx = int(np.floor(scaled_x))
    gy = int(np.floor(scaled_y))
    gx = np.clip(gx, 0, grid_size - 1)
    gy = np.clip(gy, 0, grid_size - 1)
    return (gx, gy)


def grid_to_world(gx, gy, x_min, x_max, y_min, y_max, grid_size):
    """
    グリッド座標をワールド座標に変換する

    Parameters
    ----------
    gx, gy : int or float
        グリッド座標
    x_min, x_max, y_min, y_max : float
        ワールド座標の範囲
    grid_size : int
        グリッドサイズ

    Returns
    -------
    tuple
        ワールド座標 (wx, wy)
    """
    wx = x_min + (gx / (grid_size - 1)) * (x_max - x_min)
    wy = y_min + (gy / (grid_size - 1)) * (y_max - y_min)
    return (wx, wy)
