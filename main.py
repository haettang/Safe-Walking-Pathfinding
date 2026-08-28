import json
import logging
import os

import psycopg2
from psycopg2.extras import RealDictCursor
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine, text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres:password@host.docker.internal:5432/postgres",
)
engine = create_engine(DATABASE_URL)

app = FastAPI(title="Safe Walking Route API Gateway - pgRouting Core")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RouteRequest(BaseModel):
    start_lat: float
    start_lng: float
    end_lat: float
    end_lng: float
    route_type: str = "all"  # "shortest" | "safe" | "balanced" | "all"


_WALK_SPEED_M_PER_MIN = 67.0


def _get_connection():
    return psycopg2.connect(
        dbname=engine.url.database,
        user=engine.url.username,
        password=engine.url.password,
        host=engine.url.host,
        port=engine.url.port,
        cursor_factory=RealDictCursor,
    )


def _append_path_coords(path_coords: list[dict], coords: list, tol: float = 1e-9):
    for lng, lat in coords:
        if path_coords:
            prev = path_coords[-1]
            if abs(prev["lat"] - lat) < tol and abs(prev["lng"] - lng) < tol:
                continue
        path_coords.append({"lat": lat, "lng": lng})


def _snap_point(cur, lng: float, lat: float) -> tuple[int, float]:
    """
    가장 가까운 엣지와 해당 엣지에서의 fraction을 찾는다.
    반환값: (edge_id, fraction)
    """
    cur.execute(
        """
        SELECT
            id AS edge_id,
            ST_LineLocatePoint(geom, ST_SetSRID(ST_MakePoint(%s, %s), 4326)) AS fraction
        FROM capstone.road_edges
        ORDER BY geom <-> ST_SetSRID(ST_MakePoint(%s, %s), 4326)
        LIMIT 1;
        """,
        (lng, lat, lng, lat),
    )
    row = cur.fetchone()
    if row is None:
        raise HTTPException(
            status_code=404,
            detail="시작점 또는 도착점 부근에서 연결 가능한 도로를 찾지 못했습니다.",
        )
    return int(row["edge_id"]), float(row["fraction"])


def _execute_pg_routing(cur, req: RouteRequest, cost_column: str, route_label: str):
    """
    pgr_withPoints 기반 경로 탐색.
    반환된 엣지 geometry를 실제 이동 방향에 맞춰 뒤집거나 부분 절단해서
    path가 지도상에서 꼬이지 않도록 조립한다.
    """
    margin = 0.02
    min_lat = min(req.start_lat, req.end_lat) - margin
    max_lat = max(req.start_lat, req.end_lat) + margin
    min_lng = min(req.start_lng, req.end_lng) - margin
    max_lng = max(req.start_lng, req.end_lng) + margin

    start_edge_id, start_fraction = _snap_point(cur, req.start_lng, req.start_lat)
    end_edge_id, end_fraction = _snap_point(cur, req.end_lng, req.end_lat)

    logger.info(
        f"[{route_label}] snap: start edge={start_edge_id} frac={start_fraction:.4f} | "
        f"end edge={end_edge_id} frac={end_fraction:.4f}"
    )

    query = f"""
        WITH route AS (
            SELECT
                res.*,
                LEAD(res.node) OVER (ORDER BY res.seq) AS next_node
            FROM pgr_withPoints(
                $$
                    SELECT
                        id,
                        source,
                        target,
                        COALESCE({cost_column}, length) AS cost
                    FROM capstone.road_edges
                    WHERE geom && ST_MakeEnvelope(%(min_lng)s, %(min_lat)s, %(max_lng)s, %(max_lat)s, 4326)
                $$,
                $$
                    SELECT * FROM (VALUES
                        (1, %(start_edge_id)s, %(start_frac)s),
                        (2, %(end_edge_id)s, %(end_frac)s)
                    ) AS t(pid, edge_id, fraction)
                $$,
                -1, -2,
                directed := false,
                details := true
            ) AS res
        ),
        shaped AS (
            SELECT
                route.seq,
                route.node,
                route.edge,
                route.cost,
                route.next_node,
                e.grid_id,
                COALESCE(e.cost_safe, e.length) AS edge_raw_safe_cost, -- [수정] 공통 비교를 위해 안전 가중치 원본 비용 추출
                CASE
                    WHEN route.edge = -1 OR e.geom IS NULL OR route.next_node IS NULL THEN NULL

                    -- 시작점과 도착점이 같은 edge 위에 있는 경우
                    WHEN route.node = -1 AND route.next_node = -2 THEN
                        CASE
                            WHEN %(start_frac)s <= %(end_frac)s THEN
                                ST_LineSubstring(e.geom, %(start_frac)s, %(end_frac)s)
                            ELSE
                                ST_Reverse(ST_LineSubstring(e.geom, %(end_frac)s, %(start_frac)s))
                        END

                    -- 시작 edge
                    WHEN route.node = -1 THEN
                        CASE
                            WHEN route.next_node = e.target THEN
                                ST_LineSubstring(e.geom, %(start_frac)s, 1.0)
                            WHEN route.next_node = e.source THEN
                                ST_Reverse(ST_LineSubstring(e.geom, 0.0, %(start_frac)s))
                            ELSE
                                e.geom
                        END

                    -- 종료 edge
                    WHEN route.next_node = -2 THEN
                        CASE
                            WHEN route.node = e.source THEN
                                ST_LineSubstring(e.geom, 0.0, %(end_frac)s)
                            WHEN route.node = e.target THEN
                                ST_Reverse(ST_LineSubstring(e.geom, %(end_frac)s, 1.0))
                            ELSE
                                e.geom
                        END

                    -- 일반 중간 edge
                    ELSE
                        CASE
                            WHEN route.node = e.source AND route.next_node = e.target THEN e.geom
                            WHEN route.node = e.target AND route.next_node = e.source THEN ST_Reverse(e.geom)
                            ELSE e.geom
                        END
                END AS path_geom
            FROM route
            LEFT JOIN capstone.road_edges e ON route.edge = e.id
        )
        SELECT
            seq,
            node,
            edge,
            cost,
            grid_id,
            edge_raw_safe_cost, -- [수정] 반환 데이터에 추가
            ST_AsGeoJSON(path_geom) AS geojson,
            CASE
                WHEN path_geom IS NULL THEN 0.0
                ELSE ST_Length(path_geom::geography)
            END AS segment_length_m
        FROM shaped
        ORDER BY seq;
    """

    params = {
        "start_frac": start_fraction,
        "end_frac": end_fraction,
        "start_edge_id": start_edge_id,
        "end_edge_id": end_edge_id,
        "min_lng": min_lng,
        "min_lat": min_lat,
        "max_lng": max_lng,
        "max_lat": max_lat,
    }

    cur.execute(query, params)
    rows = cur.fetchall()

    if not rows:
        logger.warning(f"[{route_label}] pgr_withPoints 결과가 비어 있습니다.")
        return None

    path_coords = []
    segments = []
    total_distance_m = 0.0
    total_safe_cost_accumulated = 0.0 # [수정] 공통으로 비교할 안전 비용 누적 변수

    for row in rows:
        geojson_str = row.get("geojson")
        if not geojson_str:
            continue

        geom_data = json.loads(geojson_str)
        geom_type = geom_data.get("type")
        coords = geom_data.get("coordinates", [])

        if geom_type != "LineString" or len(coords) < 2:
            continue

        _append_path_coords(path_coords, coords)

        segment_length_m = float(row.get("segment_length_m") or 0.0)
        
        # [수정] 탐색된 엣지의 안전 패널티 비용을 가져와 누적 (부분 절단 등을 고려하여 원본 값 매핑)
        edge_raw_safe_cost = float(row.get("edge_raw_safe_cost") or segment_length_m)
        total_safe_cost_accumulated += edge_raw_safe_cost

        total_distance_m += segment_length_m

        segments.append(
            {
                "from": {"lat": coords[0][1], "lng": coords[0][0]},
                "to": {"lat": coords[-1][1], "lng": coords[-1][0]},
                "grid_id": row.get("grid_id"),
            }
        )

    if not path_coords:
        return None

    estimated_time_min = max(1, round(total_distance_m / _WALK_SPEED_M_PER_MIN))

    return {
        "route_type": route_label,
        "distance_m": round(total_distance_m, 1),
        "estimated_time_min": estimated_time_min,
        "total_safe_cost": round(total_safe_cost_accumulated, 2), # [수정] 프론트엔드 연산용 공통 분모 전달
        "path": path_coords,
        "segments": segments,
    }


def _process_routing(req: RouteRequest, mode: str):
    conn = _get_connection()
    conn.autocommit = True
    cur = conn.cursor()

    try:
        modes_map = {
            "shortest": "cost_shortest",
            "safe": "cost_safe",
            "balanced": "cost_balanced",
        }

        if mode == "single":
            cost_col = modes_map.get(req.route_type)
            if not cost_col:
                return None
            return _execute_pg_routing(cur, req, cost_col, req.route_type)

        results = {}
        for label, cost_col in modes_map.items():
            if req.route_type in (label, "all"):
                res = _execute_pg_routing(cur, req, cost_col, label)
                if res:
                    results[label] = res
        return results

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"라우팅 처리 중 오류: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cur.close()
        conn.close()


@app.post("/route")
def get_single_route(req: RouteRequest):
    if req.route_type == "all":
        raise HTTPException(
            status_code=400,
            detail="단일 경로 요청에서는 'shortest' | 'safe' | 'balanced' 중 하나를 선택해야 합니다.",
        )

    res = _process_routing(req, mode="single")
    if res is None:
        raise HTTPException(status_code=404, detail="사용 가능한 경로를 찾을 수 없습니다.")
    return res


@app.post("/routes")
def get_all_routes(req: RouteRequest):
    results = _process_routing(req, mode="all")
    if not results or results.get("shortest") is None:
        raise HTTPException(
            status_code=404,
            detail="세 경로를 계산할 수 없거나 기준 경로(shortest)를 찾지 못했습니다.",
        )

    # [수정] 최단 경로가 지나간 엣지들의 총 위험 비용(기준점)을 가져옵니다.
    shortest_danger = results["shortest"]["total_safe_cost"]
    
    if shortest_danger > 0:
        for route_type in ("safe", "balanced"):
            r = results.get(route_type)
            if r is None:
                continue
            # [수정] (최단경로 위험비용 - 안전or밸런스 위험비용) / 최단경로 위험비용
            improvement = (shortest_danger - r["total_safe_cost"]) / shortest_danger * 100
            r["safety_improvement_pct"] = max(0.0, round(improvement, 1))

    return results


@app.get("/health")
def health_check():
    try:
        with engine.connect() as conn:
            result = conn.execute(
                text("SELECT COUNT(*) FROM capstone.road_edges")
            ).fetchone()
            total_edges = result[0] if result else 0

        return {
            "status": "healthy",
            "database_connected": True,
            "total_active_edges": total_edges,
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "database_connected": False,
            "detail": str(e),
        }
