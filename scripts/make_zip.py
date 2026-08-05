"""Đóng gói output/ thành file nộp bài, đúng cấu trúc bộ chấm mong đợi.

Vì sao không dùng Compress-Archive của PowerShell: bản 5.1 ghi dấu phân cách
đường dẫn bằng backslash (`output\\EC_001.json`), sai chuẩn ZIP vốn quy định
forward slash. Bộ chấm dùng Python zipfile sẽ coi đó là một tên file phẳng chứ
không phải thư mục `output/`.

Cấu trúc tạo ra:
    output.zip
      └── output/
            ├── EC_001.json
            ...
            └── EC_050.json

Chỉ gồm đúng 50 file JSON — mọi file khác trong output/ (ví dụ .gitkeep) đều bị bỏ.

    uv run scripts/make_zip.py
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
N_CASES = 50


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output")
    parser.add_argument("--zip", type=Path, default=ROOT / "output.zip")
    args = parser.parse_args()

    missing = [
        f"EC_{i:03d}.json"
        for i in range(1, N_CASES + 1)
        if not (args.output_dir / f"EC_{i:03d}.json").exists()
    ]
    if missing:
        raise SystemExit(f"Thiếu {len(missing)} file: {', '.join(missing[:5])}...")

    if args.zip.exists():
        args.zip.unlink()

    with zipfile.ZipFile(args.zip, "w", zipfile.ZIP_DEFLATED) as archive:
        for i in range(1, N_CASES + 1):
            name = f"EC_{i:03d}.json"
            data = (args.output_dir / name).read_text(encoding="utf-8")
            payload = json.loads(data)  # chặn file JSON hỏng lọt vào bài nộp
            if payload.get("case_id") != name[:-5]:
                raise SystemExit(f"{name}: case_id bên trong là {payload.get('case_id')!r}")
            archive.writestr(f"output/{name}", data)

    # Kiểm lại đúng cách bộ chấm sẽ đọc
    with zipfile.ZipFile(args.zip) as archive:
        names = archive.namelist()
        expected = [f"output/EC_{i:03d}.json" for i in range(1, N_CASES + 1)]
        assert names == expected, f"cấu trúc sai: {names[:3]}"
        assert all("\\" in n is False for n in names), "có backslash trong đường dẫn"
        assert archive.testzip() is None, "zip hỏng"

    size_kb = args.zip.stat().st_size / 1024
    print(f"Đã tạo {args.zip.name}: {len(expected)} file trong thư mục output/, {size_kb:.1f} KB")
    print(f"  {expected[0]} ... {expected[-1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
