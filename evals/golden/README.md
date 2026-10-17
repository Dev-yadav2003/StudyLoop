# Golden set

Ground truth for measuring perception (step 3.5) and notes quality (5.8, 6.x).
Format is enforced by `evals/golden_set.py` + `tests/unit/test_golden.py`;
copy `references/_template/` to start a new video.

| File | Content |
|---|---|
| `videos.yaml` | id, title, type (`slides`/`whiteboard`/`code`/`mixed`), tags |
| `references/<id>/captions.vtt` | Real YouTube captions — **manual** subs if they exist, else auto |
| `references/<id>/board_text.md` | `## MM:SS <type>` sections with verbatim visible text |
| `references/<id>/boundaries.yaml` | Times where the board/slide changed |
| `references/<id>/keypoints.yaml` | 10–20 facts the notes must contain |

Download captions once on your laptop (dev only — the server never touches YouTube):

```bash
uvx yt-dlp --skip-download --write-subs --write-auto-subs --sub-langs "en.*" \
  --sub-format vtt -o "evals/golden/references/%(id)s/captions" "<url>"
# then rename the produced captions.en.vtt -> captions.vtt
```
