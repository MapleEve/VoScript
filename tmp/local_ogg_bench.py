#!/usr/bin/env python3
"""
本地音频文件 bench 脚本 — 测试 OSD + MossFormer2 分离效果。

用法：
  python3 tmp/local_ogg_bench.py --dir /path/to/audio/files --onset 0.08
  python3 tmp/local_ogg_bench.py --dir /path/to/audio/files --onset 0.08 --separate

参数：
  --dir       包含 .ogg/.wav/.mp3 文件的目录（至多取前 10 个）
  --base-url  voscript API 地址（默认 http://localhost:8780）
  --api-key   API Key（默认 1sa1SA1sa）
  --onset     OSD 门限（默认 0.08）
  --separate  是否启用 MossFormer2 分离（默认 False）
  --limit     最多处理 N 个文件（默认 10）
  --out       输出 JSON 路径（默认 tmp/local_ogg_bench_results.json）
"""

import argparse
import json
import os
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

POLL_INTERVAL = 10  # seconds
POLL_TIMEOUT = 30 * 60  # 30 minutes per file


def vos_get(base_url: str, api_key: str, path: str) -> dict:
    req = urllib.request.Request(
        base_url + path,
        headers={"X-API-Key": api_key},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def vos_post_file(
    base_url: str,
    api_key: str,
    path: str,
    filepath: str,
    filename: str,
    extra: dict = None,
) -> dict:
    boundary = "----voscriptbenchboundary"
    body = b""
    for k, v in (extra or {}).items():
        body += (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{k}"\r\n'
            f"\r\n{v}\r\n"
        ).encode()
    with open(filepath, "rb") as f:
        data = f.read()
    body += (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: application/octet-stream\r\n"
        f"\r\n"
    ).encode()
    body += data + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        base_url + path,
        data=body,
        headers={
            "X-API-Key": api_key,
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def scan_audio_files(directory: str, limit: int) -> list:
    """扫描目录下的 .ogg/.wav/.mp3 文件，返回前 limit 个路径。"""
    d = Path(directory)
    if not d.is_dir():
        print(f"ERROR: {directory} is not a directory", file=sys.stderr)
        sys.exit(1)
    exts = {".ogg", ".wav", ".mp3"}
    files = sorted(
        [p for p in d.iterdir() if p.suffix.lower() in exts and p.is_file()],
        key=lambda p: p.name,
    )
    return files[:limit]


def poll_job(base_url: str, api_key: str, job_id: str) -> dict | None:
    """轮询 job 直到 completed/failed 或超时。返回 result dict 或 None。"""
    deadline = time.time() + POLL_TIMEOUT
    while time.time() < deadline:
        time.sleep(POLL_INTERVAL)
        try:
            jj = vos_get(base_url, api_key, f"/api/jobs/{job_id}")
        except Exception as e:
            print(f"    poll error: {e}", flush=True)
            continue
        st = jj.get("status")
        print(f"    status={st}", flush=True)
        if st == "completed":
            return jj.get("result") or {}
        if st == "failed":
            print(f"    FAILED: {jj.get('error', '')[:120]}")
            return None
    print(f"    TIMEOUT after {POLL_TIMEOUT}s")
    return None


def extract_stats(result: dict) -> dict:
    """从 result dict 中提取 bench 所需的统计信息。"""
    segments = result.get("segments", [])
    speaker_map = result.get("speaker_map", {})
    overlap_stats = result.get("overlap_stats") or {}
    separated_tracks = result.get("separated_tracks", [])

    n_segs = len(segments)
    n_spk = len(speaker_map)
    overlap_ratio = overlap_stats.get("ratio", 0.0)
    overlap_cnt = overlap_stats.get("count", 0)
    sep_tracks = len(separated_tracks)
    overlap_segs = sum(1 for s in segments if s.get("has_overlap", False))

    return {
        "n_segs": n_segs,
        "n_spk": n_spk,
        "overlap_ratio": overlap_ratio,
        "overlap_cnt": overlap_cnt,
        "sep_tracks": sep_tracks,
        "overlap_segs": overlap_segs,
    }


def print_table(rows: list):
    """打印对比表，格式类似 ab_adaptive_threshold_result.txt。"""
    header = (
        f"{'文件名':<30} {'dur_s':>6} {'segs':>5} {'spk':>4} "
        f"{'overlap%':>9} {'ovlp_cnt':>9} {'sep_trk':>8} {'ovlp_segs':>10}"
    )
    sep = "-" * len(header)
    print()
    print(header)
    print(sep)
    for r in rows:
        name = r["filename"]
        if len(name) > 30:
            name = name[:27] + "..."
        ratio_pct = r["overlap_ratio"] * 100
        print(
            f"{name:<30} {r['dur_s']:>6} {r['n_segs']:>5} {r['n_spk']:>4} "
            f"{ratio_pct:>8.2f}% {r['overlap_cnt']:>9} {r['sep_tracks']:>8} {r['overlap_segs']:>10}"
        )
    print(sep)


def print_summary(rows: list, separate: bool):
    total = len(rows)
    if total == 0:
        print("\n无成功结果。")
        return

    avg_ratio = sum(r["overlap_ratio"] for r in rows) / total * 100
    high_overlap = sum(1 for r in rows if r["overlap_ratio"] > 0.08)
    with_tracks = sum(1 for r in rows if r["sep_tracks"] > 0)

    print(f"\n{'='*60}")
    print(f"汇总（共 {total} 个文件）：")
    print(f"  平均 overlap_ratio : {avg_ratio:.2f}%")
    print(f"  高重叠(>8%) 文件数 : {high_overlap}")
    if separate:
        print(f"  分离收益(有 sep_tracks) : {with_tracks} 个文件")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="本地音频 bench — OSD + MossFormer2 分离"
    )
    parser.add_argument("--dir", required=True, help="含 .ogg/.wav/.mp3 文件的目录")
    parser.add_argument(
        "--base-url", default="http://localhost:8780", help="voscript API 地址"
    )
    parser.add_argument("--api-key", default="1sa1SA1sa", help="API Key")
    parser.add_argument(
        "--onset", type=float, default=0.08, help="OSD 门限（默认 0.08）"
    )
    parser.add_argument(
        "--separate", action="store_true", help="启用 MossFormer2 语音分离"
    )
    parser.add_argument("--limit", type=int, default=10, help="最多处理 N 个文件")
    parser.add_argument(
        "--out", default="tmp/local_ogg_bench_results.json", help="输出 JSON 路径"
    )
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    audio_files = scan_audio_files(args.dir, args.limit)
    if not audio_files:
        print("ERROR: 目录中未找到 .ogg/.wav/.mp3 文件", file=sys.stderr)
        sys.exit(1)

    print(f"找到 {len(audio_files)} 个文件（最多 {args.limit} 个）")
    print(f"API: {base_url}  onset={args.onset}  separate={args.separate}\n")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = []

    for i, audio_path in enumerate(audio_files):
        filename = audio_path.name
        file_size = audio_path.stat().st_size
        print(
            f"[{i+1}/{len(audio_files)}] {filename} ({file_size:,} bytes)",
            flush=True,
        )

        # 估算时长（仅用于显示，不解析音频头）
        dur_s = 0

        extra_fields = {
            "language": "zh",
            "osd": "true",
            "osd_onset": str(args.onset),
            "separate_speech": "true" if args.separate else "false",
        }

        try:
            resp = vos_post_file(
                base_url,
                args.api_key,
                "/api/transcribe",
                str(audio_path),
                filename,
                extra_fields,
            )
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")[:200]
            print(f"  submit HTTPError {e.code}: {body}")
            continue
        except Exception as e:
            print(f"  submit error: {e}")
            continue

        if resp.get("deduplicated"):
            job_id = resp.get("id")
            print(f"  dedup hit → {job_id}, fetching existing result", flush=True)
            try:
                result_data = vos_get(
                    base_url, args.api_key, f"/api/transcriptions/{job_id}"
                )
            except Exception as e:
                print(f"  fetch dedup result error: {e}")
                continue
        else:
            job_id = resp.get("id")
            if not job_id:
                print(f"  no job id in response: {resp}")
                continue
            print(f"  job={job_id}", flush=True)
            result_data = poll_job(base_url, args.api_key, job_id)
            if result_data is None:
                continue

        stats = extract_stats(result_data)

        row = {
            "filename": filename,
            "dur_s": dur_s,
            "job_id": job_id,
            **stats,
            "overlap_ratio": stats["overlap_ratio"],
        }
        results.append(row)

        print(
            f"  segs={stats['n_segs']} spk={stats['n_spk']} "
            f"overlap={stats['overlap_ratio']*100:.2f}% cnt={stats['overlap_cnt']} "
            f"sep_tracks={stats['sep_tracks']} overlap_segs={stats['overlap_segs']}",
            flush=True,
        )

        # 增量写出
        out_path.write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print()

    # 打印对比表
    print_table(results)
    print_summary(results, args.separate)

    print(f"\n结果已写入 {out_path.resolve()}")


if __name__ == "__main__":
    main()
