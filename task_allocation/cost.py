# task_allocation/cost.py
# コスト・スコア計算
import numpy as np


def normalize_arr_task(x):
    """
    タスク割り当て用の正規化関数

    Parameters
    ----------
    x : array-like
        正規化する配列

    Returns
    -------
    ndarray
        正規化された配列
    """
    x = np.array(x)
    xmin, xmax = np.min(x), np.max(x)
    if xmax - xmin < 1e-9:
        return np.zeros_like(x)
    return (x - xmin) / (xmax - xmin)


def congestion_penalty_task(i, assigned_points_indices, points, rho=2.0):
    """
    同ラウンド内の既に割り当てられた点を参照して混雑度を計算

    Parameters
    ----------
    i : int
        候補タスク点のインデックス
    assigned_points_indices : list
        そのラウンドで既に割り当てられたタスク点のインデックスリスト
    points : ndarray
        タスク点の座標配列
    rho : float
        混雑度ペナルティの減衰パラメータ

    Returns
    -------
    float
        混雑度ペナルティ値
    """
    if len(assigned_points_indices) == 0:
        return 0.0
    d = np.linalg.norm(points[assigned_points_indices] - points[i], axis=1)
    return np.sum(np.exp(-d / rho))


def calculate_bid(weights, travel_costs, congestions, alpha=1.0, beta=0.2, gamma=0.5):
    """
    入札値（bid）を計算する

    Parameters
    ----------
    weights : array-like
        ピーク強度値（正規化済み）
    travel_costs : array-like
        移動コスト（正規化済み）
    congestions : array-like
        混雑度（正規化済み）
    alpha : float
        重み係数（高いほど重要）
    beta : float
        移動コスト係数（低いほど良い）
    gamma : float
        混雑度係数（低いほど良い）

    Returns
    -------
    ndarray
        入札値配列
    """
    weights = np.array(weights)
    travel_costs = np.array(travel_costs)
    congestions = np.array(congestions)

    bids = (
        alpha * weights
        - beta * travel_costs
        - gamma * congestions
        + np.random.uniform(-1e-4, 1e-4, len(weights))
    )
    return bids
