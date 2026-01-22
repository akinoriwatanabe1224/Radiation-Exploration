# environment/grid.py
# Occupancy grid 作成
import os
import numpy as np

from obstacles.rectangle import RectangleObstacle


def load_obstacles_from_csv(testmap_csv_path=None):
    """
    CSVファイルから障害物を読み込む

    Parameters
    ----------
    testmap_csv_path : str or None
        CSVファイルのパス、Noneの場合は自動検索

    Returns
    -------
    list
        RectangleObstacle のリスト
    bool
        読み込み成功フラグ
    """
    obstacle_rects = []
    csv_loaded = False

    # 手動設定パスを優先的に使用
    if testmap_csv_path is not None:
        if os.path.exists(testmap_csv_path):
            print(f"Loading obstacles from manually specified path: {testmap_csv_path}")
            obstacle_rects = RectangleObstacle.load_obstacles(testmap_csv_path)
            csv_loaded = True
        else:
            print(f"Warning: Manually specified path not found: {testmap_csv_path}")
            print("Falling back to automatic search...")

    # 手動設定がない場合、または見つからない場合は自動検索
    if not csv_loaded:
        possible_paths = [
            "TestMAP.csv",
            "./TestMAP.csv",
            "../Map/TestMAP.csv",
            "./Map/TestMAP.csv",
            "/mnt/user-data/uploads/TestMAP.csv",
            "TestMAP3.csv",
            "./TestMAP3.csv"
        ]

        for csv_path in possible_paths:
            if os.path.exists(csv_path):
                print(f"Found TestMAP.csv at: {csv_path}")
                obstacle_rects = RectangleObstacle.load_obstacles(csv_path)
                csv_loaded = True
                break

    # CSVが見つからない場合はサンプル障害物を使用
    if not csv_loaded or len(obstacle_rects) == 0:
        print("TestMAP.csv not found or empty. Using sample obstacles.")
        obstacle_rects = [
            RectangleObstacle(15, 20, 20, 30, mu=1.0, color='gray', alpha=0.6),
            RectangleObstacle(30, 35, 10, 20, mu=1.0, color='gray', alpha=0.6),
        ]
        print(f"Created {len(obstacle_rects)} sample obstacles")
        csv_loaded = False

    return obstacle_rects, csv_loaded


def create_occupancy_grid(obstacle_rects, grid_size):
    """
    障害物リストからoccupancy gridを作成する

    Parameters
    ----------
    obstacle_rects : list
        RectangleObstacle のリスト
    grid_size : int
        グリッドサイズ

    Returns
    -------
    ndarray
        Occupancy grid（0: 自由空間、1: 障害物）
    """
    true_grid = np.zeros((grid_size, grid_size), dtype=int)

    # 障害物をoccupancy gridに反映
    for rect in obstacle_rects:
        x0 = max(0, int(np.floor(rect.x0)))
        x1 = min(grid_size, int(np.ceil(rect.x1)))
        y0 = max(0, int(np.floor(rect.y0)))
        y1 = min(grid_size, int(np.ceil(rect.y1)))
        true_grid[y0:y1, x0:x1] = 1

    print(f"Occupancy grid created with {np.sum(true_grid)} obstacle cells")
    return true_grid
