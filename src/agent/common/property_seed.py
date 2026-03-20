from __future__ import annotations

from typing import Any


# 广州精选房源（价格单位：万）
PROPERTY_SEED_DATA: list[dict[str, Any]] = [
    {"id": "GZ-TH-001", "title": "车陂 地铁旁 刚需两房 采光好", "price": 188, "layout": "2室1厅", "area": 67, "region": "天河"},
    {"id": "GZ-TH-002", "title": "东圃 学位片区 小三房 近公园", "price": 268, "layout": "3室1厅", "area": 86, "region": "天河"},
    {"id": "GZ-TH-003", "title": "员村 南向两房 带阳台 可养宠", "price": 238, "layout": "2室1厅", "area": 72, "region": "天河"},
    {"id": "GZ-TH-004", "title": "棠下 精装三房 双阳台 生活便利", "price": 318, "layout": "3室2厅", "area": 98, "region": "天河"},
    {"id": "GZ-YX-005", "title": "东山口 核心地段 一房一厅 交通便捷", "price": 220, "layout": "1室1厅", "area": 45, "region": "越秀"},
    {"id": "GZ-YX-006", "title": "淘金 距地铁近 两房带阳台 视野好", "price": 360, "layout": "2室1厅", "area": 62, "region": "越秀"},
    {"id": "GZ-YX-007", "title": "环市东 小两房 总价友好 适合上车", "price": 298, "layout": "2室1厅", "area": 58, "region": "越秀"},
    {"id": "GZ-HZ-008", "title": "江南西 地铁口 两房 带生活阳台", "price": 245, "layout": "2室1厅", "area": 63, "region": "海珠"},
    {"id": "GZ-HZ-009", "title": "客村 近商圈 三房两厅 南北通透", "price": 398, "layout": "3室2厅", "area": 102, "region": "海珠"},
    {"id": "GZ-HZ-010", "title": "工业大道 中楼层 两房双阳台 可养猫", "price": 278, "layout": "2室1厅", "area": 70, "region": "海珠"},
    {"id": "GZ-HZ-011", "title": "琶洲 周边配套成熟 两房 适合自住", "price": 480, "layout": "2室2厅", "area": 80, "region": "海珠"},
    {"id": "GZ-BY-012", "title": "同和 近地铁 两房带阳台 首付友好", "price": 175, "layout": "2室1厅", "area": 66, "region": "白云"},
    {"id": "GZ-BY-013", "title": "嘉禾望岗 三房两厅 近换乘 适合通勤", "price": 260, "layout": "3室2厅", "area": 95, "region": "白云"},
    {"id": "GZ-BY-014", "title": "黄石 近公园 两房 采光通风好", "price": 168, "layout": "2室1厅", "area": 64, "region": "白云"},
    {"id": "GZ-BY-015", "title": "白云大道北 改善三房 双卫 带大阳台", "price": 420, "layout": "3室2厅", "area": 112, "region": "白云"},
    {"id": "GZ-PY-016", "title": "市桥 刚需两房 南向阳台 生活配套全", "price": 150, "layout": "2室1厅", "area": 68, "region": "番禺"},
    {"id": "GZ-PY-017", "title": "大石 近长隆 两房带阳台 适合小家庭", "price": 182, "layout": "2室1厅", "area": 72, "region": "番禺"},
    {"id": "GZ-PY-018", "title": "洛溪 成熟小区 三房两厅 双阳台", "price": 285, "layout": "3室2厅", "area": 103, "region": "番禺"},
    {"id": "GZ-PY-019", "title": "亚运城 改善四房 视野开阔 带大阳台", "price": 468, "layout": "4室2厅", "area": 128, "region": "番禺"},
    {"id": "GZ-HP-020", "title": "大沙地 地铁沿线 两房 适合上车", "price": 210, "layout": "2室1厅", "area": 69, "region": "黄埔"},
    {"id": "GZ-HP-021", "title": "科学城 三房两厅 近学校 适合改善", "price": 388, "layout": "3室2厅", "area": 110, "region": "黄埔"},
    {"id": "GZ-HP-022", "title": "萝岗 精装两房 双阳台 小区环境好", "price": 258, "layout": "2室2厅", "area": 78, "region": "黄埔"},
    {"id": "GZ-HP-023", "title": "文冲 近地铁 三房 南北通透", "price": 335, "layout": "3室2厅", "area": 99, "region": "黄埔"},
    {"id": "GZ-LW-024", "title": "芳村 地铁旁 两房带阳台 适合自住", "price": 198, "layout": "2室1厅", "area": 65, "region": "荔湾"},
    {"id": "GZ-LW-025", "title": "西关 低楼层 两房 小区安静", "price": 268, "layout": "2室1厅", "area": 60, "region": "荔湾"},
    {"id": "GZ-LW-026", "title": "花地湾 三房两厅 总价可控 带生活阳台", "price": 320, "layout": "3室2厅", "area": 95, "region": "荔湾"},
    {"id": "GZ-NS-027", "title": "蕉门河 近地铁 一房一厅 首付轻松", "price": 98, "layout": "1室1厅", "area": 44, "region": "南沙"},
    {"id": "GZ-NS-028", "title": "金洲 刚需两房 带阳台 小区配套齐", "price": 128, "layout": "2室1厅", "area": 69, "region": "南沙"},
    {"id": "GZ-NS-029", "title": "横沥 三房两厅 近公园 适合家庭", "price": 168, "layout": "3室2厅", "area": 98, "region": "南沙"},
    {"id": "GZ-NS-030", "title": "明珠湾 两房两厅 高层景观 总价友好", "price": 185, "layout": "2室2厅", "area": 80, "region": "南沙"},
    {"id": "GZ-ZC-031", "title": "新塘 刚需两房 带阳台 交通便利", "price": 118, "layout": "2室1厅", "area": 70, "region": "增城"},
    {"id": "GZ-ZC-032", "title": "荔城 一房一厅 总价 80+ 上车盘", "price": 86, "layout": "1室1厅", "area": 40, "region": "增城"},
    {"id": "GZ-ZC-033", "title": "永宁 三房两厅 南北通透 适合自住", "price": 145, "layout": "3室2厅", "area": 96, "region": "增城"},
    {"id": "GZ-HD-034", "title": "新华 两房带阳台 生活配套成熟", "price": 108, "layout": "2室1厅", "area": 68, "region": "花都"},
    {"id": "GZ-HD-035", "title": "花城路 三房两厅 近学校 适合家庭", "price": 158, "layout": "3室2厅", "area": 99, "region": "花都"},
    {"id": "GZ-HD-036", "title": "机场北 两房两厅 高层视野 通勤方便", "price": 199, "layout": "2室2厅", "area": 82, "region": "花都"},
]


def _extract_bedrooms(layout: str) -> int:
    try:
        return int(str(layout).split("室", 1)[0])
    except Exception:
        return 0


def _extract_features(title: str) -> list[str]:
    feature_words = [
        "阳台",
        "双阳台",
        "地铁",
        "近地铁",
        "可养宠",
        "可养猫",
        "南向",
        "南北通透",
        "学位",
        "近学校",
        "电梯",
        "高层",
    ]
    found = [w for w in feature_words if w in title]
    return list(dict.fromkeys(found))


def normalized_seed_properties() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in PROPERTY_SEED_DATA:
        rows.append(
            {
                "id": item["id"],
                "title": item["title"],
                "price": float(item["price"]),
                "layout": item["layout"],
                "area": int(item["area"]),
                "region": item["region"],
                "city": "广州",
                "district": item["region"],
                "bedrooms": _extract_bedrooms(item["layout"]),
                "features": _extract_features(item["title"]),
            }
        )
    return rows
