#!/usr/bin/env python3
"""Batch OSD analysis script for voscript.

Supports three modes:
  --mode a   Method A only: OSD overlap analysis (default, original behavior)
  --mode b   Method B only: MossFormer2 speaker separation
  --mode ab  A/B comparison: run both and print a side-by-side table

Usage:
    python3 tmp/osd_batch_analysis.py [--base-url URL] [--onset FLOAT] \
        [--api-key KEY] [--limit N] [--out PATH] [--mode {a,b,ab}] \
        [--n-speakers N]
"""

import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


# ---------------------------------------------------------------------------
# HTTP helpers (stdlib only)
# ---------------------------------------------------------------------------


def _make_request(
    url: str,
    method: str = "GET",
    data: bytes | None = None,
    headers: dict | None = None,
    timeout: int = 300,
) -> Any:
    """Perform an HTTP request and return the parsed JSON body."""
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Accept", "application/json")
    if headers:
        for k, v in headers.items():
            req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        print(f"  [HTTP {exc.code}] {url}: {body[:200]}", file=sys.stderr)
        return None
    except Exception as exc:
        print(f"  [ERROR] {url}: {exc}", file=sys.stderr)
        return None


def _post_form(url: str, fields: dict, api_key: str | None, timeout: int = 300) -> Any:
    """POST application/x-www-form-urlencoded."""
    body = urllib.parse.urlencode(fields).encode()
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    if api_key:
        headers["X-API-Key"] = api_key
    return _make_request(
        url, method="POST", data=body, headers=headers, timeout=timeout
    )


def _get(url: str, api_key: str | None, timeout: int = 60) -> Any:
    headers = {}
    if api_key:
        headers["X-API-Key"] = api_key
    return _make_request(url, method="GET", headers=headers, timeout=timeout)


# ---------------------------------------------------------------------------
# Method A: OSD overlap analysis
# ---------------------------------------------------------------------------


def run_method_a(
    base_url: str, api_key: str | None, tr_id: str, onset: float
) -> dict | None:
    """Call POST /api/transcriptions/{tr_id}/analyze-overlap.

    Returns dict with keys: ratio, total_s, overlap_s, count
    Returns None on failure.
    """
    result = _post_form(
        f"{base_url}/api/transcriptions/{tr_id}/analyze-overlap",
        {"onset": str(onset)},
        api_key,
        timeout=300,
    )
    return result


# ---------------------------------------------------------------------------
# Method B: MossFormer2 speaker separation
# ---------------------------------------------------------------------------


def run_method_b(
    base_url: str, api_key: str | None, tr_id: str, n_speakers: int = 2
) -> dict | None:
    """Call POST /api/transcriptions/{tr_id}/separate.

    Returns dict with keys: n_tracks, tracks (list of {track, n_segments, text_len})
    Returns None on failure.
    """
    result = _post_form(
        f"{base_url}/api/transcriptions/{tr_id}/separate",
        {"n_speakers": str(n_speakers)},
        api_key,
        timeout=600,
    )
    return result


# ---------------------------------------------------------------------------
# Mode A: OSD-only (original behavior, preserved)
# ---------------------------------------------------------------------------


def run_mode_a(
    base_url: str, api_key: str | None, onset: float, tr_list: list, out_path: str
) -> None:
    rows = []
    for item in tr_list:
        tr_id = item["id"]
        print(f"  [{tr_id}] Running OSD ...", end=" ", flush=True)

        osd = run_method_a(base_url, api_key, tr_id, onset)
        if osd is None:
            print("FAILED")
            rows.append({"tr_id": tr_id, "error": "analyze-overlap failed"})
            continue

        # Fetch full transcription for speaker_map
        tr_detail = _get(f"{base_url}/api/transcriptions/{tr_id}", api_key)
        maple_sim = None
        matched_name = None
        spk_count = item.get("speaker_count", 0)

        if tr_detail:
            speaker_map = tr_detail.get("speaker_map", {})
            for spk_label, info in speaker_map.items():
                name = (info.get("matched_name") or "").strip()
                if name.lower() == "maple":
                    maple_sim = info.get("similarity")
                    matched_name = name
                    break

        dur_s = osd.get("total_s", 0.0)
        overlap_s = osd.get("overlap_s", 0.0)
        ratio = osd.get("ratio", 0.0)
        count = osd.get("count", 0)
        match = matched_name is not None

        print(
            f"dur={dur_s:.0f}s  overlap={overlap_s:.2f}s  ratio={ratio*100:.1f}%  cnt={count}"
        )

        rows.append(
            {
                "tr_id": tr_id,
                "filename": item.get("filename", ""),
                "created_at": item.get("created_at", ""),
                "dur_s": dur_s,
                "spk_count": spk_count,
                "maple_sim": maple_sim,
                "maple_match": match,
                "overlap_s": overlap_s,
                "ratio": ratio,
                "overlap_cnt": count,
                "onset": onset,
            }
        )

    # Print table
    print()
    header = (
        f"{'tr_id':<28} {'dur_s':>6} {'spk':>3} {'maple_sim':>9} {'match':>5}"
        f"  {'overlap_s':>9} {'ratio%':>7} {'overlap_cnt':>11}"
    )
    print(header)
    print("-" * len(header))

    for r in rows:
        if "error" in r:
            print(f"{r['tr_id']:<28}  ERROR: {r['error']}")
            continue
        maple_sim_str = (
            f"{r['maple_sim']:.4f}" if r["maple_sim"] is not None else "   N/A"
        )
        match_str = "yes" if r["maple_match"] else "no"
        print(
            f"{r['tr_id']:<28} {r['dur_s']:>6.0f} {r['spk_count']:>3} {maple_sim_str:>9} {match_str:>5}"
            f"  {r['overlap_s']:>9.3f} {r['ratio']*100:>6.1f}%  {r['overlap_cnt']:>11}"
        )

    # Summary statistics
    valid = [r for r in rows if "error" not in r]
    if valid:
        avg_ratio = sum(r["ratio"] for r in valid) / len(valid)
        high_overlap = sum(1 for r in valid if r["ratio"] > 0.08)
        print()
        print(f"Summary ({len(valid)} recordings):")
        print(f"  Average overlap ratio : {avg_ratio*100:.2f}%")
        print(f"  High overlap (>8%)    : {high_overlap}")

    # Write JSON output
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "onset": onset,
                "base_url": base_url,
                "total": len(rows),
                "rows": rows,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\nResults written to {out_path}")


# ---------------------------------------------------------------------------
# Mode B: MossFormer2 separation only
# ---------------------------------------------------------------------------


def run_mode_b(
    base_url: str, api_key: str | None, n_speakers: int, tr_list: list, out_path: str
) -> None:
    rows = []
    for item in tr_list:
        tr_id = item["id"]
        print(
            f"  [{tr_id}] Running MossFormer2 separation (n_speakers={n_speakers}) ...",
            end=" ",
            flush=True,
        )

        sep = run_method_b(base_url, api_key, tr_id, n_speakers)
        if sep is None:
            print("SEP_FAIL")
            rows.append({"tr_id": tr_id, "error": "separate failed"})
            continue

        n_tracks = sep.get("n_tracks", 0)
        tracks = sep.get("tracks", [])
        spk1 = tracks[0] if len(tracks) > 0 else {}
        spk2 = tracks[1] if len(tracks) > 1 else {}

        print(
            f"n_tracks={n_tracks}  spk1_segs={spk1.get('n_segments', 0)}"
            f"  spk2_segs={spk2.get('n_segments', 0)}"
        )

        rows.append(
            {
                "tr_id": tr_id,
                "filename": item.get("filename", ""),
                "created_at": item.get("created_at", ""),
                "n_tracks": n_tracks,
                "spk1_segs": spk1.get("n_segments", 0),
                "spk2_segs": spk2.get("n_segments", 0),
                "spk1_chars": spk1.get("text_len", 0),
                "spk2_chars": spk2.get("text_len", 0),
                "tracks": tracks,
            }
        )

    # Print table
    print()
    header = (
        f"{'tr_id':<28} {'sep_trk':>7} {'spk1_segs':>9} {'spk2_segs':>9}"
        f" {'spk1_chars':>10} {'spk2_chars':>10}"
    )
    print(header)
    print("-" * len(header))

    for r in rows:
        if "error" in r:
            print(f"{r['tr_id']:<28}  ERROR: {r['error']}")
            continue
        print(
            f"{r['tr_id']:<28} {r['n_tracks']:>7} {r['spk1_segs']:>9} {r['spk2_segs']:>9}"
            f" {r['spk1_chars']:>10} {r['spk2_chars']:>10}"
        )

    # Summary
    valid = [r for r in rows if "error" not in r]
    failed = len(rows) - len(valid)
    print()
    print(
        f"Summary: {len(valid)}/{len(rows)} recordings separated successfully ({failed} failed)"
    )

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "n_speakers": n_speakers,
                "base_url": base_url,
                "total": len(rows),
                "rows": rows,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\nResults written to {out_path}")


# ---------------------------------------------------------------------------
# Mode AB: side-by-side comparison
# ---------------------------------------------------------------------------


def run_mode_ab(
    base_url: str,
    api_key: str | None,
    onset: float,
    n_speakers: int,
    tr_list: list,
    out_path: str,
) -> None:
    rows = []
    for item in tr_list:
        tr_id = item["id"]
        spk_count = item.get("speaker_count", 0)

        # --- Method A ---
        print(f"  [{tr_id}] Method A (OSD) ...", end=" ", flush=True)
        osd = run_method_a(base_url, api_key, tr_id, onset)
        if osd is None:
            print("FAILED", end="  ")
            method_a_data = None
        else:
            dur_s = osd.get("total_s", 0.0)
            overlap_s = osd.get("overlap_s", 0.0)
            ratio = osd.get("ratio", 0.0)
            count = osd.get("count", 0)
            method_a_data = {
                "ratio": ratio,
                "overlap_s": overlap_s,
                "total_s": dur_s,
                "count": count,
            }
            print(
                f"overlap={overlap_s:.2f}s ({ratio*100:.1f}%)",
                end="  ",
                flush=True,
            )

        # --- Method B ---
        print(f"Method B (MossFormer2) ...", end=" ", flush=True)
        sep = run_method_b(base_url, api_key, tr_id, n_speakers)
        if sep is None:
            print("SEP_FAIL")
            method_b_data = None
        else:
            n_tracks = sep.get("n_tracks", 0)
            tracks = sep.get("tracks", [])
            method_b_data = {
                "n_tracks": n_tracks,
                "tracks": tracks,
                "error": None,
            }
            print(f"n_tracks={n_tracks}")

        # Determine display values
        if method_a_data is not None:
            dur_s_display = method_a_data["total_s"]
            overlap_pct = method_a_data["ratio"] * 100
            overlap_s_display = method_a_data["overlap_s"]
        else:
            dur_s_display = 0.0
            overlap_pct = 0.0
            overlap_s_display = 0.0

        if method_b_data is not None:
            tracks = method_b_data["tracks"]
            sep_trk = method_b_data["n_tracks"]
            spk1 = tracks[0] if len(tracks) > 0 else {}
            spk2 = tracks[1] if len(tracks) > 1 else {}
            spk1_segs = spk1.get("n_segments", 0)
            spk2_segs = spk2.get("n_segments", 0)
            spk1_chars = spk1.get("text_len", 0)
            spk2_chars = spk2.get("text_len", 0)
        else:
            sep_trk = 0
            spk1_segs = spk2_segs = 0
            spk1_chars = spk2_chars = 0

        rows.append(
            {
                "tr_id": tr_id,
                "filename": item.get("filename", ""),
                "created_at": item.get("created_at", ""),
                "dur_s": dur_s_display,
                "spk_count": spk_count,
                "overlap_pct": overlap_pct,
                "overlap_s": overlap_s_display,
                "sep_trk": sep_trk,
                "spk1_segs": spk1_segs,
                "spk2_segs": spk2_segs,
                "spk1_chars": spk1_chars,
                "spk2_chars": spk2_chars,
                "method_a": method_a_data,
                "method_b": (
                    method_b_data
                    if method_b_data is not None
                    else {"n_tracks": 0, "tracks": [], "error": "separate failed"}
                ),
            }
        )

    # --- Print AB comparison table ---
    sep_line = "=" * 80
    dash_line = "-" * 80
    print()
    print(sep_line)
    print("A/B COMPARISON: OSD-only vs MossFormer2 Separation")
    print(f"Onset: {onset} | N_speakers: {n_speakers}")
    print(sep_line)

    col_header = (
        f"  {'tr_id':<28} {'dur_s':>6} {'spk':>3} {'overlap%':>8}  {'overlap_s':>9}"
        f"  {'sep_trk':>7} {'spk1_segs':>9} {'spk2_segs':>9} {'spk1_chars':>10} {'spk2_chars':>10}"
    )
    print(col_header)
    print(dash_line)

    for r in rows:
        a_ok = r["method_a"] is not None
        b_ok = r["method_b"].get("error") is None

        overlap_str = f"{r['overlap_pct']:>7.1f}%" if a_ok else "   A_FAIL"
        sep_trk_str = f"{r['sep_trk']:>7}" if b_ok else "SEP_FAIL"
        spk1_segs_str = f"{r['spk1_segs']:>9}" if b_ok else "        -"
        spk2_segs_str = f"{r['spk2_segs']:>9}" if b_ok else "        -"
        spk1_chars_str = f"{r['spk1_chars']:>10}" if b_ok else "         -"
        spk2_chars_str = f"{r['spk2_chars']:>10}" if b_ok else "         -"
        overlap_s_str = f"{r['overlap_s']:>9.1f}" if a_ok else "        -"

        print(
            f"  {r['tr_id']:<28} {r['dur_s']:>6.0f} {r['spk_count']:>3}"
            f" {overlap_str}  {overlap_s_str}"
            f"  {sep_trk_str} {spk1_segs_str} {spk2_segs_str} {spk1_chars_str} {spk2_chars_str}"
        )

    print(dash_line)

    # Summary
    total = len(rows)
    valid_a = [r for r in rows if r["method_a"] is not None]
    valid_b = [r for r in rows if r["method_b"].get("error") is None]

    avg_overlap = (
        sum(r["overlap_pct"] for r in valid_a) / len(valid_a) if valid_a else 0.0
    )
    high_overlap = sum(1 for r in valid_a if r["overlap_pct"] > 8.0)

    print(f"Total: {total} recordings")
    print(f"Avg overlap: {avg_overlap:.1f}%")
    print(f"High overlap (>8%): {high_overlap}/{len(valid_a)} recordings")
    print(
        f"Separation coverage: {len(valid_b)}/{total} recordings have separated tracks"
    )
    print(sep_line)

    # Write JSON output
    out_rows = [
        {
            "tr_id": r["tr_id"],
            "method_a": r["method_a"],
            "method_b": r["method_b"],
        }
        for r in rows
    ]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "mode": "ab",
                "onset": onset,
                "n_speakers": n_speakers,
                "base_url": base_url,
                "total": total,
                "rows": out_rows,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\nResults written to {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch OSD / separation analysis for voscript"
    )
    parser.add_argument(
        "--base-url", default="http://localhost:8780", help="API base URL"
    )
    parser.add_argument(
        "--onset", type=float, default=0.08, help="OSD onset threshold (default: 0.08)"
    )
    parser.add_argument(
        "--api-key", default="1sa1SA1sa", help="API key (X-API-Key header)"
    )
    parser.add_argument(
        "--limit", type=int, default=999, help="Max number of transcriptions to process"
    )
    parser.add_argument(
        "--out", default="tmp/osd_batch_results.json", help="Output JSON path"
    )
    parser.add_argument(
        "--mode",
        choices=["a", "b", "ab"],
        default="a",
        help="a=OSD only (default), b=MossFormer2 only, ab=A/B comparison",
    )
    parser.add_argument(
        "--n-speakers",
        type=int,
        default=2,
        help="Number of speakers for Method B separation (default: 2)",
    )
    args = parser.parse_args()

    base = args.base_url.rstrip("/")
    api_key = args.api_key or None
    onset = args.onset
    n_speakers = args.n_speakers

    # Fetch transcription list
    print(f"Fetching transcription list from {base}/api/transcriptions ...")
    tr_list = _get(f"{base}/api/transcriptions", api_key)
    if not tr_list:
        print("No transcriptions found or request failed.", file=sys.stderr)
        sys.exit(1)

    tr_list = tr_list[: args.limit]
    print(
        f"Processing {len(tr_list)} transcription(s) | mode={args.mode}"
        + (f" | onset={onset}" if args.mode in ("a", "ab") else "")
        + (f" | n_speakers={n_speakers}" if args.mode in ("b", "ab") else "")
        + "\n"
    )

    if args.mode == "a":
        run_mode_a(base, api_key, onset, tr_list, args.out)
    elif args.mode == "b":
        run_mode_b(base, api_key, n_speakers, tr_list, args.out)
    else:  # ab
        run_mode_ab(base, api_key, onset, n_speakers, tr_list, args.out)


if __name__ == "__main__":
    main()
