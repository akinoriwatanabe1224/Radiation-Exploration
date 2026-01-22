# sensors/lidar.py
# LiDARセンサ機能
import numpy as np


def reveal_with_lidar(pos, discovered, true_grid, obstacle_rects, sensor_range):
    """
    LiDARセンサで環境を認識する（壁を透過しない）

    Parameters
    ----------
    pos : tuple
        現在位置 (x, y) グリッド座標
    discovered : ndarray
        発見済みマスク shape=(H, W)、この関数内で更新される
    true_grid : ndarray
        真の環境グリッド shape=(H, W)
    obstacle_rects : list
        RectangleObstacle のリスト
    sensor_range : int
        センサ範囲（グリッドセル数）
    """
    H, W = true_grid.shape
    x0, y0 = pos

    for y in range(max(0, y0 - sensor_range), min(H, y0 + sensor_range + 1)):
        for x in range(max(0, x0 - sensor_range), min(W, x0 + sensor_range + 1)):
            # 距離チェック
            dist = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
            if dist > sensor_range:
                continue

            visible = True
            # 各障害物と線分の交差判定
            for rect in obstacle_rects:
                intersections = rect.get_intersections((x0, y0), (x, y))
                if intersections is not None:
                    # 線分が障害物を通過している → 視界なし
                    visible = False
                    break
            if visible:
                discovered[y, x] = True
