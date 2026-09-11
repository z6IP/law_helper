"""查询扩展与意图识别测试（R2 / R3）。"""
from __future__ import annotations

from app.query_expansion import classify_query_intents, expand_query, expand_synonyms


def test_expand_synonyms_map_legal_terms():
    """R4：同义词/近义词表把口语化关键词映射到法条术语。"""
    synonyms = expand_synonyms("酒驾怎么处罚，追尾责任怎么划分")
    assert any("饮酒后驾驶机动车" in s for s in synonyms)
    assert any("后车未与前车保持安全距离" in s for s in synonyms)
    assert expand_synonyms("") == []


def test_expand_query_new_traffic_rules():
    """R3：新增的查询扩展规则能覆盖常见交通/治安场景。"""
    assert "机动车行驶超过规定时速 超速 处罚 记分" in expand_query("超速行驶会被怎么处罚")
    assert "机动车违反规定停放 临时停车 妨碍其他车辆行人通行 处罚" in expand_query("违停扣分吗")
    assert "未取得机动车驾驶证 机动车驾驶证被吊销 暂扣期间 驾驶机动车 处罚" in expand_query("无证驾驶怎么处罚")
    assert "机动车行经人行横道 遇行人正在通过 停车让行 处罚 记分" in expand_query("不礼让行人扣分吗")
    assert "非紧急情况在应急车道行驶 停车 处罚 记分" in expand_query("占用应急车道")
    assert "机动车逆向行驶 处罚 记分" in expand_query("逆行怎么处罚")
    assert "摩托车驾驶人 乘坐人员 未按规定戴安全头盔 处罚 记分" in expand_query("骑摩托车没戴头盔")
    assert "交通事故 人身损害赔偿 医疗费 误工费 护理费 残疾赔偿金 死亡赔偿金" in expand_query("事故误工费怎么赔")


def test_classify_negation_intent():
    """R2：否定意图通过正则模式精确识别，避免'不'字误命中所有查询。"""
    assert "negation" in classify_query_intents("没戴头盔会被怎么处罚")
    assert "negation" in classify_query_intents("未系安全带扣分吗")
    assert "negation" in classify_query_intents("不礼让行人怎么处罚")
    assert "negation" not in classify_query_intents("怎么处罚")
    assert "negation" not in classify_query_intents("处罚种类有哪些")
    assert classify_query_intents("") == []


def test_classify_exception_intent():
    """R2：例外/从轻/减轻/免予处罚意图识别。"""
    assert "exception" in classify_query_intents("可以免予处罚吗")
    assert "exception" in classify_query_intents("首违不罚吗")
    assert "exception" in classify_query_intents("能否从轻处罚")


def test_classify_scenario_intent():
    """R2：情形意图识别（轻微事故、自行协商、快处快赔等）。"""
    assert "scenario" in classify_query_intents("轻微事故怎么处理")
    assert "scenario" in classify_query_intents("财产损失事故自行协商")
    assert "scenario" not in classify_query_intents("醉酒驾驶怎么处罚")
