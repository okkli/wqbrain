"""统一入口：`python -m sync forum|kb [...]`"""

from __future__ import annotations

import sys


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("用法: python -m sync {forum|kb} [options]")
        print("  forum  - 同步论坛（python -m sync forum --help 查看参数）")
        print("  kb     - 同步官方知识库（python -m sync kb --help 查看参数）")
        sys.exit(0)

    target, rest = sys.argv[1], sys.argv[2:]
    if target == "forum":
        from .forum.cli import main as forum_main
        forum_main(rest)
    elif target == "kb":
        from .kb.cli import main as kb_main
        kb_main(rest)
    else:
        print(f"未知目标: {target}（应为 forum 或 kb）", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
