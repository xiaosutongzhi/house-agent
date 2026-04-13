#!/usr/bin/env python3
"""Static-export LangGraph architecture from langgraph.json.

Outputs per graph:
- <graph_name>.mmd        Mermaid source
- <graph_name>.json       Node/edge metadata
- <graph_name>.png        Optional PNG (when --png and backend available)
- index.json              Export summary
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

from langgraph.graph.state import CompiledStateGraph


def _load_dotenv_if_present(project_root: Path) -> None:
    env_file = project_root / ".env"
    if not env_file.exists():
        return
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv(env_file)
    except Exception:
        pass


def _bootstrap_paths(project_root: Path) -> None:
    src_root = project_root / "src"
    for p in (str(project_root), str(src_root)):
        if p not in sys.path:
            sys.path.insert(0, p)


def _load_module_from_file(module_name: str, file_path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载模块文件: {file_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _resolve_compiled_graph(obj: Any, graph_name: str) -> CompiledStateGraph:
    if isinstance(obj, CompiledStateGraph):
        return obj

    # 兼容工厂函数返回图对象
    if callable(obj):
        out = obj()
        if isinstance(out, CompiledStateGraph):
            return out

    raise TypeError(
        f"图 '{graph_name}' 不是 CompiledStateGraph（或可返回 CompiledStateGraph 的可调用对象）。"
    )


def _serialize_drawable_graph(drawable: Any) -> dict[str, Any]:
    nodes = []
    for node_id, node in (getattr(drawable, "nodes", {}) or {}).items():
        nodes.append(
            {
                "id": str(getattr(node, "id", node_id)),
                "name": str(getattr(node, "name", node_id)),
                "metadata": getattr(node, "metadata", None),
                "data_type": type(getattr(node, "data", None)).__name__,
            }
        )

    edges = []
    for edge in (getattr(drawable, "edges", []) or []):
        edges.append(
            {
                "source": str(getattr(edge, "source", "")),
                "target": str(getattr(edge, "target", "")),
                "conditional": bool(getattr(edge, "conditional", False)),
                "data": getattr(edge, "data", None),
            }
        )

    return {
        "node_count": len(nodes),
        "edge_count": len(edges),
        "nodes": nodes,
        "edges": edges,
    }


def _parse_graph_ref(graph_ref: str) -> tuple[str, str]:
    if ":" not in graph_ref:
        raise ValueError(f"非法 graph 引用: {graph_ref}. 预期格式: ./path/to/file.py:symbol")
    file_part, symbol = graph_ref.split(":", 1)
    return file_part, symbol


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def export_from_config(
    config_path: Path,
    output_dir: Path,
    *,
    xray: bool = True,
    png: bool = False,
    only: set[str] | None = None,
) -> dict[str, Any]:
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    graphs: dict[str, str] = cfg.get("graphs", {})
    if not graphs:
        raise RuntimeError("langgraph.json 中没有 graphs 字段。")

    project_root = config_path.parent.resolve()
    _load_dotenv_if_present(project_root)
    _bootstrap_paths(project_root)

    # 仅做静态导出，不会真正调用模型；避免因缺 key 导致导入失败。
    os.environ.setdefault("OPENAI_API_KEY", "DUMMY_FOR_STATIC_EXPORT")

    summary: dict[str, Any] = {
        "config": str(config_path),
        "output_dir": str(output_dir),
        "xray": xray,
        "graphs": {},
    }

    for graph_name, graph_ref in graphs.items():
        if only and graph_name not in only:
            continue

        file_part, symbol = _parse_graph_ref(graph_ref)
        file_path = (project_root / file_part).resolve()
        if not file_path.exists():
            raise FileNotFoundError(f"图文件不存在: {file_path}")

        module_name = f"graph_export_{graph_name}"
        mod = _load_module_from_file(module_name, file_path)
        if not hasattr(mod, symbol):
            raise AttributeError(f"模块 {file_path} 中不存在符号: {symbol}")

        raw_obj = getattr(mod, symbol)
        compiled = _resolve_compiled_graph(raw_obj, graph_name)
        drawable = compiled.get_graph(xray=xray)

        mermaid = drawable.draw_mermaid()
        mmd_path = output_dir / f"{graph_name}.mmd"
        _write_text(mmd_path, mermaid)

        meta = _serialize_drawable_graph(drawable)
        meta_path = output_dir / f"{graph_name}.json"
        _write_json(meta_path, meta)

        png_path = None
        if png:
            try:
                png_bytes = drawable.draw_mermaid_png()
                png_path = output_dir / f"{graph_name}.png"
                png_path.parent.mkdir(parents=True, exist_ok=True)
                png_path.write_bytes(png_bytes)
            except Exception:
                png_path = None

        summary["graphs"][graph_name] = {
            "graph_ref": graph_ref,
            "mermaid_file": str(mmd_path),
            "meta_file": str(meta_path),
            "png_file": str(png_path) if png_path else None,
            "node_count": meta["node_count"],
            "edge_count": meta["edge_count"],
        }

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="导出 LangGraph 静态架构（Mermaid/JSON）")
    parser.add_argument("--config", default="langgraph.json", help="langgraph.json 路径")
    parser.add_argument("--output", default="static/graph_arch", help="导出目录")
    parser.add_argument("--no-xray", action="store_true", help="关闭 xray 视图")
    parser.add_argument("--png", action="store_true", help="尝试导出 PNG")
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="只导出指定图名（空格分隔），如: --only house_agent reserve_agent",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = export_from_config(
        config_path,
        output_dir,
        xray=not args.no_xray,
        png=args.png,
        only=set(args.only) if args.only else None,
    )

    index_path = output_dir / "index.json"
    _write_json(index_path, summary)

    print("✅ 导出完成")
    print(f"- config: {config_path}")
    print(f"- output: {output_dir}")
    print(f"- summary: {index_path}")
    for name, info in summary.get("graphs", {}).items():
        print(
            f"  • {name}: nodes={info['node_count']} edges={info['edge_count']} -> {Path(info['mermaid_file']).name}"
        )


if __name__ == "__main__":
    main()
