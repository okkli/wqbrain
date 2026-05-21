"""评论爬取白名单。

默认**不**爬取帖子评论（论坛大量帖子评论是闲聊、占位、表情）。
只对标题命中关键词的帖子拉评论——这些帖子通常是"征文"/"教程"/
"工具分享"类，评论本身就是核心内容。

调整方式：
- 编辑下面的 KEYWORDS 列表，加/减关键词
- 在 EXTRA_POST_IDS 加上你要强制保留评论的 post_id
- 在 EXCLUDE_POST_IDS 加上你要强制排除评论的 post_id
（id/keyword 任一命中即触发；排除优先级最高）
"""

from __future__ import annotations

import re

KEYWORDS: list[str] = [
    # 显式征集类（评论本身就是核心干货）
    "征文", "征集", "邀请分享", "分享你",
    # 系列性技术帖
    "SuperAlpha", "Super Alpha", "super alpha",
    "Community Leader", "community leader",
    "Brain Labs", "BRAIN Labs",
    # 技术分享前缀（标题里有这些约等于声明"我要分享技术"）
    "经验分享", "代码分享", "心得分享", "案例分享", "工具分享",
    "工作流分享", "深度分享", "工程技术分享", "代码优化", "工程优化",
    "Alpha灵感", "Alpha Template", "alpha template", "模板",
    "新人分享", "踩坑分享", "学习笔记", "设计思路",
    # 教程
    "教程", "实训", "教学", "攻略",
    # 算法/技术关键词
    "selection", "combination", "neutralize", "decay",
    "Risk Neutralized", "Power Pool", "Pyramid", "Genius",
    "Robust universe Sharpe", "Sub-universe Sharpe",
    "operator", "datafield",
    # 比赛
    "AIAC", "Osmosis", "IQC", "PPAC", "SAC", "比赛",
    # 工具配置
    "MCP", "wqb", "API技巧", "AI打工人",
    # 经验类（命中量级合理且通常是技术内容）
    "经验之谈", "心得体会", "VF 0", "VF 1", "ValueFactor",
    "踩坑", "技巧",
]

EXTRA_POST_IDS: set[str] = set()
EXCLUDE_POST_IDS: set[str] = set()

_PAT = re.compile("|".join(re.escape(k) for k in KEYWORDS), re.IGNORECASE)


def should_fetch_comments(post_id: str, title: str) -> bool:
    if post_id in EXCLUDE_POST_IDS:
        return False
    if post_id in EXTRA_POST_IDS:
        return True
    return bool(_PAT.search(title or ""))
