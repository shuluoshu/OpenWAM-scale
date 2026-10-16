#!/usr/bin/env python3
"""Check local HTML assets and optionally decode a bounded number of videos.

Checks HTML src/href/poster/data attributes, not JS-generated or CSS url assets.
Frame diversity proves encoded images differ, not that source cameras moved.
"""
import argparse
import base64
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import tempfile
from urllib.parse import unquote, urlsplit


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.refs = []

    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if key in ("src", "href", "poster", "data") and value:
                self.refs.append((tag, key, value))


def decode_video(source):
    import cv2
    cv2.setNumThreads(1)
    with tempfile.TemporaryDirectory(prefix="openwam-video-check-") as folder:
        if source.startswith("data:"):
            header, payload = source.split(",", 1)
            if ";base64" not in header:
                raise ValueError("Only base64 embedded video is supported")
            path = Path(folder) / "embedded.mp4"
            path.write_bytes(base64.b64decode(payload))
        else:
            path = Path(source)
        cap = cv2.VideoCapture(str(path))
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            code = int(cap.get(cv2.CAP_PROP_FOURCC))
            codec = "".join(chr((code >> (8*i)) & 255) for i in range(4))
            count, hashes = 0, set()
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                count += 1
                hashes.add(hashlib.sha256(frame.tobytes()).digest())
            if count < 2:
                raise ValueError(f"Only {count} frames decoded")
            return dict(decoded_frames=count, distinct_encoded_frames=len(hashes), fps=fps,
                        duration_s=count/fps if fps else None, codec_fourcc=codec,
                        h264_fourcc=codec.lower() in {"avc1", "h264"}, source_motion_verified=False)
        finally:
            cap.release()


def validate(page, root, require_video=False, require_offline=False, decode_videos=0):
    parser = Links()
    parser.feed(page.read_text(encoding="utf-8"))
    missing, outside, absolute, external, local, videos = set(), set(), set(), set(), set(), {}
    for tag, attr, value in parser.refs:
        uri = urlsplit(value)
        if uri.scheme == "data":
            if value.lower().startswith("data:video/"):
                videos[value] = "embedded"
            continue
        if uri.scheme in {"http", "https"} or uri.netloc:
            if attr != "href":
                external.add(value)
            continue
        if uri.scheme in {"mailto", "tel", "javascript", "about"}:
            continue
        if not uri.path:
            continue
        if uri.scheme or uri.path.startswith("/"):
            absolute.add(value)
            continue
        path = (page.parent / unquote(uri.path)).resolve()
        if not path.is_relative_to(root):
            outside.add(value)
            continue
        if not path.is_file():
            missing.add(value)
            continue
        local.add(str(path.relative_to(root)))
        if path.suffix.lower() in {".mp4", ".webm", ".mov", ".m4v"}:
            videos[str(path)] = str(path.relative_to(root))
    errors = []
    if missing or outside or absolute:
        errors.append("Some local references cannot be moved with this report root")
    if require_video and not videos:
        errors.append("No video source found")
    if require_offline and external:
        errors.append("Runtime assets require a network connection")
    decoded = []
    for source, label in list(videos.items())[:decode_videos]:
        try:
            decoded.append(dict(source=label, **decode_video(source)))
        except (ValueError, ImportError, OSError) as error:
            errors.append(f"Video decode failed ({label}): {error}")
    return dict(all_passed=not errors, page=str(page), root=str(root),
                local_assets_checked=len(local), missing=sorted(missing), outside_root=sorted(outside),
                absolute_references=sorted(absolute), external_runtime=sorted(external),
                video_sources=len(videos), embedded_video_sources=sum(x == "embedded" for x in videos.values()),
                decoded_videos=decoded, errors=errors,
                scope="HTML attributes only; anchors, CSS/JS-generated URLs and source-frame motion require separate checks")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("page", type=Path)
    p.add_argument("--root", type=Path)
    p.add_argument("--require-video", action="store_true")
    p.add_argument("--require-offline", action="store_true")
    p.add_argument("--decode-videos", type=int, default=0)
    p.add_argument("--output-json", type=Path)
    args = p.parse_args()
    if args.decode_videos < 0:
        p.error("--decode-videos must be nonnegative")
    page = args.page.expanduser().resolve()
    root = (args.root or page.parent).expanduser().resolve()
    if not page.is_file() or not page.is_relative_to(root):
        p.error("page must exist inside --root")
    report = validate(page, root, args.require_video, args.require_offline, args.decode_videos)
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text, encoding="utf-8")
    print(text, end="")
    return int(not report["all_passed"])


if __name__ == "__main__":
    raise SystemExit(main())
