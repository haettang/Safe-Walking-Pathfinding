"""
cache.py
2단계 디스크 캐시: OSM 추출 캐시 + 최종 그래프 캐시.
원본 .pbf / .geojson 은 절대 수정하지 않고 data/cache/ 아래에만 산출물을 보관한다.

빌드/로드 분리
--------------
- 빌드(load_or_build_graph): 메모리가 충분한 독립 환경에서만 수행한다.
  pyrosm/geopandas/graph 는 이 경로에서만 lazy import 된다 — 서비스가 cache 를
  import 해도 무거운 전처리 의존성이 끌려오지 않게 한다.
- 로드(load_graph_cache): 서비스 진입점. 미리 빌드된 캐시 파일만 읽고
  버전만 가볍게 검증한다. 원본 .pbf/.geojson 도, pyrosm/geopandas 도 불필요.

캐시 hit 조건
-------------
OSM 추출 캐시 (dalseo_osm_extract.pkl) — 빌드 환경 내부 전용
    - 원본 .pbf mtime
    - bbox (지오JSON에서 derive)

그래프 캐시 (graph_cache.pkl) — 외부로 전달되는 산출물, 메타 내장
    - cache_version (version.CACHE_VERSION)
    - 원본 .pbf mtime / .geojson mtime (빌드 경로의 재빌드 판정용)
    - K_dict (라벨별 K 값)
"""

import json
import os
import pickle
import tempfile
import time
from pathlib import Path

from version import CACHE_VERSION


_OSM_PKL = "dalseo_osm_extract.pkl"
_OSM_META = "dalseo_osm_extract.meta.json"
_GRAPH_PKL = "graph_cache.pkl"


def _cache_dir(pbf_path: str) -> Path:
    d = Path(pbf_path).parent / "cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _atomic_write(path: Path, write_fn) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as f:
            write_fn(f)
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def _write_pickle(path: Path, obj) -> None:
    _atomic_write(path, lambda f: pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL))


def _write_json(path: Path, obj) -> None:
    payload = json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8")
    _atomic_write(path, lambda f: f.write(payload))


def _read_meta(path: Path):
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def load_or_build_osm_extract(pbf_path: str, bbox):
    cache_dir = _cache_dir(pbf_path)
    pkl = cache_dir / _OSM_PKL
    meta_path = cache_dir / _OSM_META

    expected = {
        "pbf_mtime": os.path.getmtime(pbf_path),
        "bbox": [float(b) for b in bbox],
    }
    actual = _read_meta(meta_path)

    if pkl.exists() and actual == expected:
        t0 = time.perf_counter()
        with open(pkl, "rb") as f:
            nodes, edges = pickle.load(f)
        print(f"[cache] OSM 추출 캐시 hit → {time.perf_counter() - t0:.2f}s")
        return nodes, edges

    print("[cache] OSM 추출 캐시 miss → 원본 .pbf 파싱 (수 분 소요 가능)")
    from graph import load_osm_walking_network  # lazy: pyrosm 는 빌드 시에만 로드

    t0 = time.perf_counter()
    nodes, edges = load_osm_walking_network(pbf_path, bbox)
    print(f"[cache] OSM 추출 완료 → {time.perf_counter() - t0:.2f}s, 캐시 저장")

    _write_pickle(pkl, (nodes, edges))
    _write_json(meta_path, expected)
    return nodes, edges


def _graph_meta(pbf_path: str, geojson_path: str, K_dict: dict) -> dict:
    return {
        "cache_version": CACHE_VERSION,
        "pbf_mtime": os.path.getmtime(pbf_path),
        "geojson_mtime": os.path.getmtime(geojson_path),
        "K_dict": {k: float(K_dict[k]) for k in sorted(K_dict)},
    }


def load_or_build_graph(pbf_path: str, geojson_path: str, K_dict: dict, out_path=None):
    pkl = Path(out_path) if out_path else _cache_dir(pbf_path) / _GRAPH_PKL

    expected_meta = _graph_meta(pbf_path, geojson_path, K_dict)

    if pkl.exists():
        try:
            with open(pkl, "rb") as f:
                record = pickle.load(f)
            cached_meta = {k: record[k] for k in expected_meta}
            if cached_meta == expected_meta:
                print("[cache] 그래프 캐시 hit (메타 일치)")
                return record["payload"]
        except (pickle.UnpicklingError, OSError, EOFError, KeyError, TypeError):
            pass  # 손상/구포맷 → 재빌드

    print("[cache] 그래프 캐시 miss → 재빌드")

    # 빌드 전용 의존성은 여기서만 lazy import (서비스 로드 경로 오염 방지)
    import geopandas as gpd
    from graph import build_graph, _ensure_wgs84

    # 1) 격자 로드 + bbox 산출 (geojson은 비교적 작음 — 매번 읽어도 빠름)
    t0 = time.perf_counter()
    grid = _ensure_wgs84(gpd.read_file(geojson_path))
    bounds = grid.total_bounds
    bbox = [float(b) for b in bounds]
    print(f"[cache] 격자 로드 완료 → {time.perf_counter() - t0:.2f}s")

    # 2) OSM 추출 (1단계 캐시 활용)
    nodes, edges = load_or_build_osm_extract(pbf_path, bbox)

    # 3) 그래프 구성
    t0 = time.perf_counter()
    G, tree, node_ids, min_W_dict = build_graph(nodes, edges, grid, K_dict)
    print(f"[cache] 그래프 구성 완료 → {time.perf_counter() - t0:.2f}s")

    service_bbox = {
        "min_lat": float(bounds[1]),
        "max_lat": float(bounds[3]),
        "min_lng": float(bounds[0]),
        "max_lng": float(bounds[2]),
    }

    payload = (G, tree, node_ids, min_W_dict, service_bbox)
    record = {**expected_meta, "payload": payload}
    _write_pickle(pkl, record)
    print(f"[cache] 그래프 캐시 저장 → {pkl}")
    return payload


def load_graph_cache(cache_path: str):
    p = Path(cache_path)
    if not p.exists():
        raise RuntimeError(
            f"그래프 캐시 파일이 없습니다: {cache_path}\n"
            "독립(메모리 충분) 환경에서 scripts/build_cache.py 로 캐시를 먼저 "
            "생성해 전달하고, 서비스에는 GRAPH_CACHE_PATH 로 경로를 지정하세요."
        )

    t0 = time.perf_counter()
    with open(p, "rb") as f:
        record = pickle.load(f)

    if not isinstance(record, dict) or "payload" not in record:
        raise RuntimeError(
            f"캐시 파일 포맷이 올바르지 않습니다(구포맷일 수 있음): {cache_path}\n"
            "scripts/build_cache.py 로 재생성한 캐시를 사용하세요."
        )

    cached_ver = record.get("cache_version")
    if cached_ver != CACHE_VERSION:
        raise RuntimeError(
            f"캐시 버전 불일치: 캐시={cached_ver}, 코드={CACHE_VERSION}.\n"
            "코드와 동일한 버전으로 scripts/build_cache.py 를 재실행해 "
            "캐시를 재생성하세요."
        )

    print(f"[cache] 그래프 캐시 로드 → {time.perf_counter() - t0:.2f}s")
    return record["payload"]
