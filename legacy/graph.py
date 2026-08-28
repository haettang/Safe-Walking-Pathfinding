import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
from scipy.spatial import KDTree
# pyrosm 은 load_osm_walking_network 내부에서만 lazy import 한다.
# OSM 추출 캐시가 살아 있으면 pyrosm 없이도 그래프 캐시를 재빌드할 수 있다.


# CACHE_VERSION 은 version.py(경량 모듈)로 분리되어 있다.
# 가중치 공식·그래프 구성 로직이 바뀌면 version.py 의 값을 올려 캐시를 무효화한다.


def _ensure_wgs84(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        return gdf.set_crs("EPSG:4326")
    if gdf.crs.to_epsg() != 4326:
        return gdf.to_crs("EPSG:4326")
    return gdf


def load_osm_walking_network(pbf_path: str, bbox) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    from pyrosm import OSM  # lazy: OSM 추출 캐시 miss 시에만 필요

    osm = OSM(pbf_path, bounding_box=[float(b) for b in bbox])
    nodes, edges = osm.get_network(nodes=True, network_type="walking")
    if nodes is None or edges is None or len(edges) == 0:
        raise RuntimeError("OSM 보행 네트워크 데이터가 없습니다. bbox를 확인하세요.")
    nodes = _ensure_wgs84(nodes)
    edges = _ensure_wgs84(edges)
    return nodes, edges


def build_graph(
    nodes: gpd.GeoDataFrame,
    edges: gpd.GeoDataFrame,
    grid: gpd.GeoDataFrame,
    K_dict: dict,
) -> tuple[nx.DiGraph, KDTree, list, dict]:
    if not K_dict:
        raise ValueError("K_dict는 최소 1개 라벨을 포함해야 합니다.")

    grid = _ensure_wgs84(grid.copy())

    # ------------------------------------------------------------------
    # 1. K 무관 격자 컬럼 (C, V, L) 및 D 계산 — 1회만
    # ------------------------------------------------------------------
    n_grid = len(grid)
    C = grid["crime_indicator"].fillna(0.0).to_numpy(dtype=float) if "crime_indicator" in grid.columns else np.zeros(n_grid)
    V = grid["cctv_count"].fillna(0.0).to_numpy(dtype=float)      if "cctv_count"       in grid.columns else np.zeros(n_grid)
    L = grid["light_count"].fillna(0.0).to_numpy(dtype=float)     if "light_count"      in grid.columns else np.zeros(n_grid)
    D = np.minimum(0.60, 0.20 * np.log(V + 1) + 0.14 * np.log(L + 1))

    # ------------------------------------------------------------------
    # 2. 엣지-격자 공간 조인 — K 무관, 1회만
    #    joined: 엣지 인덱스 → 겹치는 격자 인덱스 (다대다)
    # ------------------------------------------------------------------
    edges_slim = edges[["u", "v", "geometry", "length"]].copy()
    joined = gpd.sjoin(
        edges_slim,
        grid[["geometry"]],            # 격자 인덱스만 보존 — W는 K별로 다르게 매핑
        how="left",
        predicate="intersects",
    )
    grid_idx_per_row = joined["index_right"]
    valid_row_mask = grid_idx_per_row.notna().to_numpy()
    grid_idx_valid = grid_idx_per_row[valid_row_mask].astype(int).to_numpy()
    joined_index_arr = joined.index.to_numpy()

    # ------------------------------------------------------------------
    # 3. K별 W_avg / safe_weight 컬럼을 edges에 추가
    # ------------------------------------------------------------------
    edges_out = edges.copy()
    edges_out["length"] = edges_out["length"].astype(float)
    length_arr_for_sw = edges_out["length"].to_numpy(dtype=float)

    for label, K in K_dict.items():
        P = 1.0 + C * K
        W_grid = np.maximum(1.0, P * (1.0 - D))   # 격자별 W

        # joined의 각 행에 격자 W를 매핑
        joined_W = np.full(len(joined), np.nan)
        joined_W[valid_row_mask] = W_grid[grid_idx_valid]

        # 같은 엣지(원본 인덱스)가 여러 격자와 겹칠 수 있으므로 평균
        s = pd.Series(joined_W, index=joined_index_arr)
        W_avg_per_edge = s.groupby(s.index).mean()

        col_w_avg = f"W_avg_{label}"
        col_sw    = f"safe_weight_{label}"
        # Edge별 안전 가중치 적용
        edges_out[col_w_avg] = W_avg_per_edge

        # 격자 외 도로는 중립 가중치 적용
        edges_out[col_w_avg] = edges_out[col_w_avg].fillna(1.0)

        # 이동 거리와 안전 가중치를 결합하여 Safe Cost 계산
        edges_out[col_sw] = (
            length_arr_for_sw
            * edges_out[col_w_avg].to_numpy(dtype=float)
        )

    # ------------------------------------------------------------------
    # 4. NetworkX DiGraph 구성 (노드/엣지 추가는 1회만)
    # ------------------------------------------------------------------
    G = nx.DiGraph()

    id_col = "id" if "id" in nodes.columns else None
    ids = nodes[id_col].to_numpy() if id_col else nodes.index.to_numpy()
    G.add_nodes_from(
        (nid, {"y": geom.y, "x": geom.x})
        for nid, geom in zip(ids, nodes.geometry)
    )

    oneway_col = "oneway" in edges_out.columns
    u_arr      = edges_out["u"].to_numpy()
    v_arr      = edges_out["v"].to_numpy()
    length_arr = edges_out["length"].to_numpy(dtype=float)
    w_avg_arrs = {label: edges_out[f"W_avg_{label}"].to_numpy(dtype=float) for label in K_dict}
    sw_arrs    = {label: edges_out[f"safe_weight_{label}"].to_numpy(dtype=float) for label in K_dict}
    oneway_arr = edges_out["oneway"].to_numpy() if oneway_col else None

    G_node_set = set(G.nodes())
    fwd, rev, bi_fwd, bi_rev = [], [], [], []

    for i in range(len(u_arr)):
        u, v = u_arr[i], v_arr[i]
        if u not in G_node_set or v not in G_node_set:
            continue
        attrs = {"length": float(length_arr[i])}
        for label in K_dict:
            attrs[f"W_avg_{label}"]       = float(w_avg_arrs[label][i])
            attrs[f"safe_weight_{label}"] = float(sw_arrs[label][i])

        ow = oneway_arr[i] if oneway_col else None
        if ow in (True, "yes", 1, "1"):
            fwd.append((u, v, attrs))
        elif str(ow) == "-1":
            rev.append((v, u, attrs))
        else:
            bi_fwd.append((u, v, attrs))
            bi_rev.append((v, u, attrs))

    G.add_edges_from(fwd)
    G.add_edges_from(rev)
    G.add_edges_from(bi_fwd)
    G.add_edges_from(bi_rev)

    # ------------------------------------------------------------------
    # 5. KDTree 구성 (좌표 있는 노드만)
    # ------------------------------------------------------------------
    node_ids = [
        n for n in G.nodes()
        if "y" in G.nodes[n] and "x" in G.nodes[n]
    ]
    coords = np.array([[G.nodes[n]["y"], G.nodes[n]["x"]] for n in node_ids])
    tree = KDTree(coords)

    # ------------------------------------------------------------------
    # 6. 라벨별 min_W (A* 휴리스틱 admissibility 보장용 — 그래프 구성 시 1회 계산)
    # ------------------------------------------------------------------
    min_W_dict: dict = {}
    for label in K_dict:
        col = f"W_avg_{label}"
        min_W_dict[label] = float(min(
            (d.get(col, 1.0) for _, _, d in G.edges(data=True)),
            default=1.0,
        ))

    return G, tree, node_ids, min_W_dict


def load_graph(
    pbf_path: str,
    grid: gpd.GeoDataFrame,
    K_dict: dict,
) -> tuple[nx.DiGraph, KDTree, list, dict]:
    grid_w = _ensure_wgs84(grid.copy())
    bounds = grid_w.total_bounds
    bbox = [float(b) for b in bounds]
    nodes, edges = load_osm_walking_network(pbf_path, bbox)
    return build_graph(nodes, edges, grid_w, K_dict)
