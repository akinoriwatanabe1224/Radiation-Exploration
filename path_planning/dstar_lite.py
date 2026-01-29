# path_planning/dstar_lite.py
# D* Lite アルゴリズム実装
import heapq


class DStarLite:
    """D* Lite 経路計画アルゴリズム"""

    def __init__(self, grid, start, goal, discovered_grid=None, unknown_penalty=1.0):
        """
        Parameters
        ----------
        grid : ndarray
            障害物グリッド（0: 自由空間、1: 障害物）
        start : tuple
            開始位置 (x, y)
        goal : tuple
            目標位置 (x, y)
        discovered_grid : ndarray, optional
            探索済みグリッド（True: 既知、False: 未知）
        unknown_penalty : float, optional
            未知領域を通過する際のコストペナルティ係数（デフォルト: 1.0）
        """
        self.grid = grid
        self.start = start
        self.goal = goal
        self.discovered_grid = discovered_grid
        self.unknown_penalty = unknown_penalty
        self.g = {}
        self.rhs = {}
        self.U = []
        self.km = 0
        self.width = grid.shape[1]
        self.height = grid.shape[0]

        for y in range(self.height):
            for x in range(self.width):
                self.g[(x, y)] = float("inf")
                self.rhs[(x, y)] = float("inf")
        self.rhs[self.goal] = 0
        heapq.heappush(self.U, (self.calculate_key(self.goal), self.goal))

    def heuristic(self, a, b):
        """
        マンハッタン距離によるヒューリスティック

        Parameters
        ----------
        a, b : tuple
            座標

        Returns
        -------
        int
            マンハッタン距離
        """
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def edge_cost(self, to_cell):
        """
        セルへの移動コストを計算する

        Parameters
        ----------
        to_cell : tuple
            移動先セルの座標 (x, y)

        Returns
        -------
        float
            移動コスト（未知領域の場合はペナルティを適用）
        """
        x, y = to_cell
        # discovered_gridがない場合、または既知セルの場合はコスト1
        if self.discovered_grid is None or self.discovered_grid[y, x]:
            return 1.0
        # 未知セルの場合はペナルティを適用
        return self.unknown_penalty

    def calculate_key(self, s):
        """
        キー値を計算する

        Parameters
        ----------
        s : tuple
            座標

        Returns
        -------
        tuple
            キー値 (k1, k2)
        """
        return (
            min(self.g[s], self.rhs[s]) + self.heuristic(self.start, s) + self.km,
            min(self.g[s], self.rhs[s]),
        )

    def get_neighbors(self, s):
        """
        隣接セルを取得する

        Parameters
        ----------
        s : tuple
            座標

        Returns
        -------
        list
            隣接セルのリスト
        """
        x, y = s
        moves = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        result = []
        for dx, dy in moves:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.width and 0 <= ny < self.height:
                if self.grid[ny, nx] == 0:
                    result.append((nx, ny))
        return result

    def update_vertex(self, u):
        """
        頂点を更新する

        Parameters
        ----------
        u : tuple
            座標
        """
        if u != self.goal:
            nbrs = self.get_neighbors(u)
            self.rhs[u] = min([self.g[s] + self.edge_cost(u) for s in nbrs] or [float("inf")])
        if any(v == u for _, v in self.U):
            self.U = [(k, v) for k, v in self.U if v != u]
            heapq.heapify(self.U)
        if self.g[u] != self.rhs[u]:
            heapq.heappush(self.U, (self.calculate_key(u), u))

    def compute_shortest_path(self):
        """最短経路を計算する"""
        while self.U and (
            self.U[0][0] < self.calculate_key(self.start)
            or self.rhs[self.start] != self.g[self.start]
        ):
            k_old, u = heapq.heappop(self.U)
            if k_old < self.calculate_key(u):
                heapq.heappush(self.U, (self.calculate_key(u), u))
            elif self.g[u] > self.rhs[u]:
                self.g[u] = self.rhs[u]
                for s in self.get_neighbors(u):
                    self.update_vertex(s)
            else:
                self.g[u] = float("inf")
                for s in [u] + self.get_neighbors(u):
                    self.update_vertex(s)

    def extract_path(self):
        """
        経路を抽出する

        Returns
        -------
        list or None
            経路のリスト、到達不可の場合None
        """
        path = [self.start]
        s = self.start
        visited = set()
        while s != self.goal:
            neighbors = self.get_neighbors(s)
            if not neighbors:
                return None
            # 隣接セルへの移動コストを考慮して次のセルを選択
            next_s = min(neighbors, key=lambda x: self.g.get(x, float("inf")) + self.edge_cost(x))
            if self.g.get(next_s, float("inf")) == float("inf"):
                return None
            s = next_s
            if s in visited:
                return None
            visited.add(s)
            path.append(s)
        return path
