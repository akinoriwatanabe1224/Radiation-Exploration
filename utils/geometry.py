# utils/geometry.py
# 幾何計算ユーティリティ
import numpy as np


def find_intersection(meas, direction, x_min, x_max, y_min, y_max):
    """
    測定点から方向ベクトルに沿って、境界との交点を求める

    Parameters
    ----------
    meas : array-like
        測定点の座標 [x, y]
    direction : array-like
        方向ベクトル [dx, dy]
    x_min, x_max, y_min, y_max : float
        境界座標

    Returns
    -------
    tuple
        境界との交点座標 (x, y)
    """
    dx, dy = direction
    intersections = []
    if abs(dx) > 1e-12:
        for x_edge in [x_min, x_max]:
            t = (x_edge - meas[0]) / dx
            y = meas[1] + t * dy
            if y_min <= y <= y_max and t > 0:
                intersections.append((x_edge, y))
    if abs(dy) > 1e-12:
        for y_edge in [y_min, y_max]:
            t = (y_edge - meas[1]) / dy
            x = meas[0] + t * dx
            if x_min <= x <= x_max and t > 0:
                intersections.append((x, y_edge))
    if not intersections:
        return meas
    dists = [np.linalg.norm(np.array(p) - meas) for p in intersections]
    return intersections[int(np.argmin(dists))]


def normalize_arr(x):
    """
    配列を0-1に正規化する

    Parameters
    ----------
    x : array-like
        正規化する配列

    Returns
    -------
    ndarray
        正規化された配列
    """
    x = np.array(x, dtype=float)
    if np.nanmax(x) - np.nanmin(x) < 1e-9:
        return np.zeros_like(x)
    return (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x) + 1e-12)
