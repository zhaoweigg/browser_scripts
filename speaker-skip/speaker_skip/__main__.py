"""CLI:
  python -m speaker_skip enroll NAME VIDEO_ID [VIDEO_ID ...]   build a voiceprint
  python -m speaker_skip analyze VIDEO_ID [--full]              analyze one video (--full: skip the transcript shortcut)
  python -m speaker_skip serve [--no-menubar]                   run the helper for the userscript
"""

import json
import sys


def main(argv):
    if len(argv) >= 3 and argv[0] == "enroll":
        from .pipeline import enroll
        print(json.dumps(enroll(argv[1], argv[2:]), ensure_ascii=False, indent=1))
    elif len(argv) >= 2 and argv[0] == "analyze":
        from .pipeline import analyze_video
        r = analyze_video(argv[1], method="full" if "--full" in argv else "auto",
                          status=lambda s, p: print(f"\r{s} {p:5.0%}", end="", file=sys.stderr))
        print(file=sys.stderr)
        print(json.dumps({k: r[k] for k in ("method", "note", "duration", "speakers")}, ensure_ascii=False, indent=1))
        for name, ranges in r["skip"].items():
            print(f"{name}: {len(ranges)} ranges, {sum(e - s for s, e in ranges) / 60:.1f} min")
    elif len(argv) == 2 and argv[0] == "worker":
        # Used by the helper: one JSON progress line per change on stdout.
        import traceback
        from .pipeline import analyze_video
        last = [None]

        def report(state, progress):
            cur = (state, round(progress, 2))
            if cur != last[0]:
                last[0] = cur
                print(json.dumps({"state": state, "progress": cur[1]}), flush=True)

        try:
            analyze_video(argv[1], status=report)
        except Exception as e:
            traceback.print_exc()
            text = str(e).strip()
            print(json.dumps({"state": "error", "message": (text.splitlines()[-1] if text else repr(e))[:300]}), flush=True)
            return 1
    elif argv[:1] == ["serve"]:
        from .server import serve
        serve(menubar="--no-menubar" not in argv)
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
