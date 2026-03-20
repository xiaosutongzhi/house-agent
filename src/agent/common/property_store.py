from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

import pymysql
from dotenv import load_dotenv
from pymysql.cursors import DictCursor

from src.agent.common.property_seed import normalized_seed_properties


def _project_root() -> Path:
    current = Path(__file__).resolve()
    return current.parents[3]


def _is_db_disabled() -> bool:
    if os.getenv("DISABLE_DB", "").strip().lower() in {"1", "true", "yes", "on"}:
        return True
    required = ["DB_USER", "DB_PASSWORD", "DB_HOST", "DB_PORT", "DB_NAME"]
    return any(not _clean_env_value(os.getenv(k)) for k in required)


def _clean_env_value(value: str | None) -> str:
    if value is None:
        return ""
    return value.strip().strip('"').strip("'")


def _normalize_base_url(base_url: str | None) -> str | None:
    cleaned = _clean_env_value(base_url)
    if not cleaned:
        return None
    if cleaned.startswith(("http://", "https://")):
        return cleaned
    return f"https://{cleaned.lstrip('/')}"


def _split_soft_prefs(text: str) -> list[str]:
    s = (text or "").replace("，", ";").replace("、", ";")
    parts = [x.strip() for x in s.split(";") if x.strip()]
    if not parts and s.strip():
        parts = [x for x in s.split() if x.strip()]
    return parts


class PropertyStore:
    """统一房源数据访问层：MySQL 优先，失败回退内置数据。"""

    def __init__(self) -> None:
        load_dotenv()
        self.seed_data = normalized_seed_properties()
        self.db_enabled = not _is_db_disabled()
        self._init_error: str | None = None
        if self.db_enabled:
            try:
                self._ensure_table_and_seed()
            except Exception as exc:
                self._init_error = str(exc)
                self.db_enabled = False

    def _connect(self):
        return pymysql.connect(
            host=_clean_env_value(os.getenv("DB_HOST", "127.0.0.1")),
            port=int(_clean_env_value(os.getenv("DB_PORT", "3306")) or "3306"),
            user=_clean_env_value(os.getenv("DB_USER")),
            password=_clean_env_value(os.getenv("DB_PASSWORD")),
            database=_clean_env_value(os.getenv("DB_NAME")),
            charset="utf8mb4",
            cursorclass=DictCursor,
            autocommit=True,
        )

    def _ensure_table_and_seed(self) -> None:
        ddl = """
        CREATE TABLE IF NOT EXISTS property_listings (
            id VARCHAR(32) PRIMARY KEY,
            title VARCHAR(255) NOT NULL,
            price DECIMAL(10,2) NOT NULL,
            layout VARCHAR(32) NOT NULL,
            area INT NOT NULL,
            city VARCHAR(32) NOT NULL,
            region VARCHAR(32) NOT NULL,
            district VARCHAR(32) NOT NULL,
            bedrooms INT NOT NULL,
            features_json TEXT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            INDEX idx_city_region (city, region),
            INDEX idx_price (price),
            INDEX idx_bedrooms (bedrooms)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """
        upsert = """
        INSERT INTO property_listings (
            id, title, price, layout, area, city, region, district, bedrooms, features_json
        ) VALUES (
            %(id)s, %(title)s, %(price)s, %(layout)s, %(area)s, %(city)s, %(region)s, %(district)s, %(bedrooms)s, %(features_json)s
        )
        ON DUPLICATE KEY UPDATE
            title = VALUES(title),
            price = VALUES(price),
            layout = VALUES(layout),
            area = VALUES(area),
            city = VALUES(city),
            region = VALUES(region),
            district = VALUES(district),
            bedrooms = VALUES(bedrooms),
            features_json = VALUES(features_json);
        """
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(ddl)
                for row in self.seed_data:
                    payload = dict(row)
                    for key in ("id", "title", "layout", "city", "region", "district"):
                        if payload.get(key) is not None:
                            payload[key] = str(payload[key]).strip()
                    payload["features_json"] = json.dumps(row.get("features") or [], ensure_ascii=False)
                    cur.execute(upsert, payload)

    def list_properties(self, region: str | None = None, limit: int | None = None) -> list[dict[str, Any]]:
        if self.db_enabled:
            try:
                where = []
                params: list[Any] = []
                if region and region != "全部":
                    where.append("region = %s")
                    params.append(region)
                sql = (
                    "SELECT id, title, price, layout, area, city, region, district, bedrooms, features_json "
                    "FROM property_listings"
                )
                if where:
                    sql += " WHERE " + " AND ".join(where)
                sql += " ORDER BY price ASC"
                if limit is not None:
                    sql += " LIMIT %s"
                    params.append(int(limit))
                with self._connect() as conn:
                    with conn.cursor() as cur:
                        cur.execute(sql, params)
                        rows = cur.fetchall() or []
                return [self._row_to_property(r) for r in rows]
            except Exception:
                self.db_enabled = False

        rows = [dict(x) for x in self.seed_data]
        if region and region != "全部":
            rows = [x for x in rows if x.get("region") == region]
        rows.sort(key=lambda x: x.get("price", 0))
        if limit is not None:
            rows = rows[:limit]
        return rows

    def search_sql(
        self,
        *,
        budget_min: float | None = None,
        budget_max: float | None = None,
        city: str | None = None,
        bedrooms: int | None = None,
        district: str | None = None,
    ) -> dict[str, Any]:
        rows = self.list_properties(region=district)

        def _ok(r: dict[str, Any]) -> bool:
            if city and r.get("city") != city:
                return False
            if district and r.get("district") != district and r.get("region") != district:
                return False
            if budget_min is not None and float(r.get("price", 0)) < float(budget_min):
                return False
            if budget_max is not None and float(r.get("price", 0)) > float(budget_max):
                return False
            if bedrooms is not None and int(r.get("bedrooms", 0)) != int(bedrooms):
                return False
            return True

        filtered = [r for r in rows if _ok(r)]
        candidates = [
            {
                "id": x["id"],
                "title": x["title"],
                "city": x["city"],
                "district": x["district"],
                "price": x["price"],
                "bedrooms": x["bedrooms"],
            }
            for x in filtered
        ]
        return {"candidate_ids": [x["id"] for x in filtered], "candidates": candidates}

    def vector_search(self, *, query_soft_prefs: str, candidate_ids: list[str], top_k: int = 5) -> list[dict[str, Any]]:
        if not candidate_ids:
            return []

        vector_result = self._vector_search_chroma(query_soft_prefs, candidate_ids, top_k)
        if vector_result is not None:
            return vector_result
        return self._vector_search_keyword(query_soft_prefs, candidate_ids, top_k)

    def _vector_search_keyword(self, query_soft_prefs: str, candidate_ids: list[str], top_k: int) -> list[dict[str, Any]]:
        prefs = _split_soft_prefs(query_soft_prefs)
        allowed = set(candidate_ids)
        rows = [x for x in self.list_properties() if x.get("id") in allowed]
        ranked: list[tuple[int, dict[str, Any]]] = []
        for row in rows:
            text = f"{row.get('title', '')} {' '.join(row.get('features') or [])}"
            score = sum(1 for p in prefs if p and p in text)
            ranked.append((score, row))
        ranked.sort(key=lambda x: (x[0], -float(x[1].get("price", 0))), reverse=True)
        return [
            {"id": r["id"], "title": r["title"], "price": r["price"], "score": float(s)}
            for s, r in ranked[:top_k]
        ]

    def _vector_search_chroma(
        self,
        query_soft_prefs: str,
        candidate_ids: list[str],
        top_k: int,
    ) -> list[dict[str, Any]] | None:
        try:
            import chromadb  # type: ignore
            from langchain_openai import OpenAIEmbeddings
        except Exception:
            return None

        api_key = _clean_env_value(os.getenv("OPENAI_API_KEY"))
        if not api_key:
            return None

        try:
            emb_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
            base_url = _normalize_base_url(os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE"))
            if base_url:
                os.environ["OPENAI_BASE_URL"] = base_url

            embeddings = OpenAIEmbeddings(
                model=emb_model,
                api_key=api_key,
                base_url=base_url,
            )

            persist_dir = os.getenv("CHROMA_PERSIST_DIR") or str(_project_root() / "data" / "chroma")
            client = chromadb.PersistentClient(path=persist_dir)
            coll = client.get_or_create_collection("property_listings_v1")

            all_rows = self.list_properties()
            ids = [x["id"] for x in all_rows]
            docs = [
                f"{x['title']}；城市{x['city']}；区域{x['region']}；户型{x['layout']}；面积{x['area']}平；特征{' '.join(x.get('features') or [])}"
                for x in all_rows
            ]
            metas = [
                {
                    "property_id": x["id"],
                    "title": x["title"],
                    "price": float(x["price"]),
                    "city": x["city"],
                    "district": x["district"],
                }
                for x in all_rows
            ]
            doc_embs = embeddings.embed_documents(docs)
            coll.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=doc_embs)

            query_emb = embeddings.embed_query(query_soft_prefs or "户型采光交通宠物")
            where_filter: dict[str, Any] | None = None
            # 严格元数据过滤：仅在 candidate_ids 范围内向量检索
            # Chroma 支持 $in 语法；若后端不支持，异常会被兜底并回退关键词检索。
            if candidate_ids:
                where_filter = {"property_id": {"$in": [str(x) for x in candidate_ids]}}
            result = coll.query(
                query_embeddings=[query_emb],
                n_results=max(top_k * 4, top_k),
                include=["metadatas", "distances"],
                where=where_filter,
            )

            allowed = set(candidate_ids)
            output: list[dict[str, Any]] = []
            seen: set[str] = set()
            metadatas = (result.get("metadatas") or [[]])[0]
            distances = (result.get("distances") or [[]])[0]
            for meta, dist in zip(metadatas, distances):
                pid = str((meta or {}).get("property_id") or "")
                if not pid or pid not in allowed or pid in seen:
                    continue
                seen.add(pid)
                output.append(
                    {
                        "id": pid,
                        "title": (meta or {}).get("title", ""),
                        "price": (meta or {}).get("price", 0),
                        "score": float(-(dist or 0.0)),
                    }
                )
                if len(output) >= top_k:
                    break
            if output:
                return output
        except Exception:
            return None
        return None

    @staticmethod
    def _row_to_property(row: dict[str, Any]) -> dict[str, Any]:
        features: list[str] = []
        raw_features = row.get("features_json")
        if raw_features:
            try:
                obj = json.loads(raw_features)
                if isinstance(obj, list):
                    features = [str(x) for x in obj]
            except Exception:
                features = []
        return {
            "id": row.get("id"),
            "title": row.get("title"),
            "price": float(row.get("price", 0)),
            "layout": row.get("layout"),
            "area": int(row.get("area", 0)),
            "city": str(row.get("city") or "广州").strip(),
            "region": str(row.get("region") or row.get("district") or "").strip(),
            "district": str(row.get("district") or row.get("region") or "").strip(),
            "bedrooms": int(row.get("bedrooms", 0)),
            "features": features,
        }


_STORE_LOCK = threading.Lock()
_STORE_SINGLETON: PropertyStore | None = None


def get_property_store() -> PropertyStore:
    global _STORE_SINGLETON
    if _STORE_SINGLETON is not None:
        return _STORE_SINGLETON
    with _STORE_LOCK:
        if _STORE_SINGLETON is None:
            _STORE_SINGLETON = PropertyStore()
    return _STORE_SINGLETON
