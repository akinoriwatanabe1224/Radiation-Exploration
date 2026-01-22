# task_allocation/auction.py
# オークションベースのタスク割り当て
import numpy as np

from .cost import normalize_arr_task, congestion_penalty_task, calculate_bid
from utils.coordinate import world_to_grid


def run_auction(points, peaks_val, num_robots, robot_start_world,
                rho=2.0, alpha=1.0, beta=0.2, gamma=0.5):
    """
    オークションベースの逐次タスク割り当てを実行する

    Parameters
    ----------
    points : ndarray
        タスク点（ピーク位置）の座標配列 shape=(n_points, 2)
    peaks_val : ndarray
        各ピークの強度値
    num_robots : int
        ロボット数
    robot_start_world : list
        各ロボットの開始位置（ワールド座標）
    rho : float
        混雑度ペナルティの減衰パラメータ
    alpha : float
        重み係数
    beta : float
        移動コスト係数
    gamma : float
        混雑度係数

    Returns
    -------
    dict
        ロボットIDをキーとする割り当てタスクインデックスリストの辞書
    """
    if len(points) == 0:
        raise RuntimeError("No peaks detected")

    # 各ロボットの現在位置を追跡
    robots_current_pos = np.array(robot_start_world.copy())

    # タスク割り当て結果
    assigned = {r: [] for r in range(num_robots)}
    unassigned = list(range(len(points)))

    print("\n=== Auction-based sequential task allocation ===")

    # 逐次オークション
    while unassigned:
        A_points = []  # そのラウンド内の「既に取られた点」のインデックス

        # ラウンド内で各ロボットが1点ずつ獲得する
        for r in range(num_robots):
            if len(unassigned) == 0:
                break

            # 全候補点の raw 値を集める
            raw_weights = []
            raw_travel_cost = []
            raw_congestion = []

            for i in unassigned:
                # (A) Weight: ピークの強度値
                weight = peaks_val[i] if peaks_val[i] > 1e-6 else 1e-6
                raw_weights.append(weight)

                # (B) Travel cost: 現在位置からの距離
                dist = np.linalg.norm(robots_current_pos[r] - points[i])
                raw_travel_cost.append(dist)

                # (C) Congestion: 同ラウンド内の既に取られた点による混雑度
                congestion = congestion_penalty_task(i, A_points, points, rho)
                raw_congestion.append(congestion)

            # 正規化（A,B,C）
            Weight_norm = normalize_arr_task(np.array(raw_weights))
            Travel_cost_norm = normalize_arr_task(np.array(raw_travel_cost))
            Congestion_norm = normalize_arr_task(np.array(raw_congestion))

            # 入札値計算
            bids = calculate_bid(
                Weight_norm, Travel_cost_norm, Congestion_norm,
                alpha, beta, gamma
            )

            # 最高入札値の点を選ぶ
            idx = np.argmax(bids)
            point_id = unassigned[idx]

            # 割り当て
            assigned[r].append(point_id)
            A_points.append(point_id)

            # ロボットをその地点に移動（仮想的な位置更新）
            robots_current_pos[r] = points[point_id]

            # 未割り当てリストから削除
            unassigned.remove(point_id)

            print(f"  Round: Robot {r} bid for task {point_id} (bid: {bids[idx]:.4f})")

    print("\n=== Task allocation result ===")
    for r in range(num_robots):
        print(f"  Robot {r}: {len(assigned[r])} peaks => {assigned[r]}")

    return assigned


def run_dynamic_auction(robot_id, robot_position, unassigned_task_ids, task_positions,
                        in_progress_positions, peaks_val,
                        rho=2.0, alpha=1.0, beta=0.2, gamma=0.5):
    """
    単一ロボットに対する動的オークションを実行する

    探査中にロボットがアイドル状態になったとき、次のタスクを割り当てる

    Parameters
    ----------
    robot_id : int
        オークションに参加するロボットのID
    robot_position : tuple or ndarray
        ロボットの現在位置（ワールド座標）
    unassigned_task_ids : list
        未割り当てタスクのインデックスリスト
    task_positions : ndarray
        全タスク点の座標配列 shape=(n_points, 2)
    in_progress_positions : list
        他のロボットが担当中または向かっている位置のリスト（congestion用）
    peaks_val : ndarray
        各ピークの強度値
    rho : float
        混雑度ペナルティの減衰パラメータ
    alpha : float
        重み係数
    beta : float
        移動コスト係数
    gamma : float
        混雑度係数

    Returns
    -------
    int or None
        割り当てられたタスクID、割り当てるタスクがない場合はNone
    float
        入札値（タスクがない場合は0.0）
    """
    if len(unassigned_task_ids) == 0:
        return None, 0.0

    robot_pos = np.array(robot_position)

    # 全候補点の raw 値を集める
    raw_weights = []
    raw_travel_cost = []
    raw_congestion = []

    for task_id in unassigned_task_ids:
        # (A) Weight: ピークの強度値
        weight = peaks_val[task_id] if peaks_val[task_id] > 1e-6 else 1e-6
        raw_weights.append(weight)

        # (B) Travel cost: 現在位置からの距離
        dist = np.linalg.norm(robot_pos - task_positions[task_id])
        raw_travel_cost.append(dist)

        # (C) Congestion: 他のロボットが向かっている/担当中の位置による混雑度
        congestion = 0.0
        task_pos = task_positions[task_id]
        for in_prog_pos in in_progress_positions:
            d = np.linalg.norm(task_pos - np.array(in_prog_pos))
            if d < 1e-6:
                congestion += 10.0  # 同じ位置への大きなペナルティ
            else:
                congestion += np.exp(-d / rho)
        raw_congestion.append(congestion)

    # 正規化
    Weight_norm = normalize_arr_task(np.array(raw_weights))
    Travel_cost_norm = normalize_arr_task(np.array(raw_travel_cost))
    Congestion_norm = normalize_arr_task(np.array(raw_congestion))

    # 入札値計算
    bids = calculate_bid(
        Weight_norm, Travel_cost_norm, Congestion_norm,
        alpha, beta, gamma
    )

    # 最高入札値の点を選ぶ
    idx = np.argmax(bids)
    selected_task_id = unassigned_task_ids[idx]

    return selected_task_id, bids[idx]


def convert_to_grid_coordinates(assigned, points, robot_start_world,
                                true_grid, x_min, x_max, y_min, y_max, grid_size):
    """
    タスク割り当て結果をグリッド座標に変換する

    Parameters
    ----------
    assigned : dict
        ロボットIDをキーとする割り当てタスクインデックスリストの辞書
    points : ndarray
        タスク点の座標配列
    robot_start_world : list
        各ロボットの開始位置（ワールド座標）
    true_grid : ndarray
        障害物グリッド
    x_min, x_max, y_min, y_max : float
        ワールド座標の範囲
    grid_size : int
        グリッドサイズ

    Returns
    -------
    list
        各ロボットの開始位置（グリッド座標）
    dict
        ロボットIDをキーとするタスク位置（グリッド座標）リストの辞書
    """
    num_robots = len(assigned)

    # ロボット開始位置をグリッド座標に変換
    robot_start_grid = []
    for r in range(num_robots):
        start_pos = world_to_grid(robot_start_world[r], x_min, x_max, y_min, y_max, grid_size)
        sx, sy = start_pos
        # 開始位置が障害物でないことを確認
        if true_grid[sy, sx] == 1:
            print(f"Warning: Robot {r} start position {start_pos} is on obstacle, finding alternative...")
            found = False
            for offset in range(1, 10):
                for dx in range(-offset, offset + 1):
                    for dy in range(-offset, offset + 1):
                        nx, ny = sx + dx, sy + dy
                        if 0 <= nx < grid_size and 0 <= ny < grid_size and true_grid[ny, nx] == 0:
                            start_pos = (nx, ny)
                            found = True
                            print(f"  -> Moved to {start_pos}")
                            break
                    if found:
                        break
                if found:
                    break
        robot_start_grid.append(start_pos)

    # タスク位置をグリッド座標に変換
    assigned_goals_grid = {}
    for r in range(num_robots):
        assigned_goals_grid[r] = []
        for p in assigned[r]:
            goal_pos = world_to_grid(points[p], x_min, x_max, y_min, y_max, grid_size)
            gx, gy = goal_pos
            # タスク点が障害物でないことを確認
            if true_grid[gy, gx] == 1:
                print(f"Warning: Obs {p} position {goal_pos} is on obstacle, finding alternative...")
                found = False
                for offset in range(1, 10):
                    for dx in range(-offset, offset + 1):
                        for dy in range(-offset, offset + 1):
                            nx, ny = gx + dx, gy + dy
                            if 0 <= nx < grid_size and 0 <= ny < grid_size and true_grid[ny, nx] == 0:
                                goal_pos = (nx, ny)
                                found = True
                                print(f"  -> Moved to {goal_pos}")
                                break
                        if found:
                            break
                    if found:
                        break
            assigned_goals_grid[r].append(goal_pos)

    return robot_start_grid, assigned_goals_grid
