import math
from heapq import heappop, heappush

import networkx as nx
from scipy.spatial import KDTree


# 보행 평균 속도: 4 km/h = 약 67 m/min
_WALK_SPEED_M_PER_MIN = 67.0

# 지구 반지름 (m) - 하버사인 공식용
_EARTH_RADIUS_M = 6_371_000.0

# 최근접 노드 스냅 거리 상한 (도 단위, 위도 1° ≈ 111 km)
# 1 km 이상 떨어진 도로에 스냅되면 사용자 좌표가 서비스 범위 밖으로 판단
_MAX_SNAP_DISTANCE_DEG = 0.009  # ≈ 1 km


def _haversine(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * _EARTH_RADIUS_M * math.asin(math.sqrt(a))


def _make_heuristic(G: nx.DiGraph, scale: float):
    def heuristic(u, v):
        u_data, v_data = G.nodes[u], G.nodes[v]
        # bbox 경계 노드에 좌표가 없을 수 있으므로 방어 처리
        if "y" not in u_data or "y" not in v_data:
            return 0.0
        return _haversine(u_data["y"], u_data["x"], v_data["y"], v_data["x"]) * scale

    return heuristic


def _nearest_node(tree: KDTree, node_ids: list, lat: float, lng: float):
    dist, idx = tree.query([lat, lng])
    if dist > _MAX_SNAP_DISTANCE_DEG:
        raise ValueError(
            f"입력 좌표({lat}, {lng}) 근처에 보행 가능한 도로가 없습니다. "
            f"(최근접 도로까지 약 {dist * 111_000:.0f}m, 허용 상한 {_MAX_SNAP_DISTANCE_DEG * 111_000:.0f}m)"
        )
    return node_ids[idx]


def _path_to_coords(G: nx.DiGraph, path: list) -> list[dict]:
    coords = []
    for n in path:
        data = G.nodes[n]
        if "y" not in data or "x" not in data:
            raise RuntimeError(f"경로 노드(id={n})에 좌표 정보가 없습니다.")
        coords.append({"lat": data["y"], "lng": data["x"]})
    return coords


def _path_to_segments(G: nx.DiGraph, path: list, w_avg_attr: str) -> list[dict]:
    segments = []
    for u, v in zip(path[:-1], path[1:]):
        u_data, v_data = G.nodes[u], G.nodes[v]
        segments.append({
            "from":  {"lat": u_data["y"], "lng": u_data["x"]},
            "to":    {"lat": v_data["y"], "lng": v_data["x"]},
            "W_avg": round(G[u][v].get(w_avg_attr, 1.0), 4),
        })
    return segments


def _path_stats(G: nx.DiGraph, path: list, safe_weight_attr: str) -> dict:
    total_dist = 0.0
    total_safe_weight = 0.0

    for u, v in zip(path[:-1], path[1:]):
        data = G[u][v]
        length = data.get("length", 0.0)
        total_dist += length
        total_safe_weight += data.get(safe_weight_attr, length)

    estimated_time_min = max(1, round(total_dist / _WALK_SPEED_M_PER_MIN))

    return {
        "distance_m": round(total_dist, 1),
        "estimated_time_min": estimated_time_min,
        "total_safe_weight": round(total_safe_weight, 2),
    }


def _astar_path(G: nx.DiGraph, source, target, heuristic, weight_attr: str) -> list:
    if source not in G:
        raise nx.NodeNotFound(f"Source {source} is not in G")
    if target not in G:
        raise nx.NodeNotFound(f"Target {target} is not in G")

    # (추정 총비용 f, 누적비용 g, 동률 해소 순번, 노드)
    frontier = []
    sequence = 0
    heappush(frontier, (heuristic(source, target), 0.0, sequence, source))
    g_score = {source: 0.0}
    came_from = {}

    while frontier:
        _, current_cost, _, current = heappop(frontier)

        # 더 좋은 경로가 이미 큐에 들어간 경우의 오래된 항목은 건너뛴다.
        if current_cost != g_score.get(current):
            continue

        if current == target:
            path = [target]
            while path[-1] != source:
                path.append(came_from[path[-1]])
            path.reverse()
            return path

        for neighbor, edge_data in G[current].items():
            edge_cost = edge_data.get(weight_attr, 1.0)
            if edge_cost is None:
                continue
            edge_cost = float(edge_cost)
            if not math.isfinite(edge_cost) or edge_cost < 0:
                raise ValueError(f"유효하지 않은 A* 엣지 비용: {current!r} -> {neighbor!r}")

            tentative_cost = current_cost + edge_cost
            if tentative_cost < g_score.get(neighbor, math.inf):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_cost
                sequence += 1
                estimated_total = tentative_cost + heuristic(neighbor, target)
                heappush(frontier, (estimated_total, tentative_cost, sequence, neighbor))

    raise nx.NetworkXNoPath(f"Node {target} not reachable from {source}")


def find_route(
    G: nx.DiGraph,
    tree: KDTree,
    node_ids: list,
    start_lat: float,
    start_lng: float,
    end_lat: float,
    end_lng: float,
    route_type: str,
    weight_attr: str,
    w_avg_attr: str,
    safe_weight_attr: str,
    min_W: float = 1.0,
) -> dict:
    src = _nearest_node(tree, node_ids, start_lat, start_lng)
    dst = _nearest_node(tree, node_ids, end_lat, end_lng)

    scale = 1.0 if weight_attr == "length" else min_W
    heuristic = _make_heuristic(G, scale)

    path = _astar_path(G, src, dst, heuristic, weight_attr)
    coords = _path_to_coords(G, path)
    segments = _path_to_segments(G, path, w_avg_attr)
    stats = _path_stats(G, path, safe_weight_attr)

    return {"route_type": route_type, "path": coords, "segments": segments, **stats}
