# Notepad ScreenSeekeR

**Vision-based desktop automation with dynamic icon grounding.** Finds the Notepad
desktop icon *anywhere* on a Windows 10/11 desktop using a training-free
implementation of **ScreenSeekeR** ([arXiv:2504.07981](https://arxiv.org/abs/2504.07981)),
launches it, and writes the first 10
[JSONPlaceholder](https://jsonplaceholder.typicode.com/posts) posts to
`Desktop/tjm-project/post_{id}.txt`.

Grounding is **pure vision** — pixels in, coordinates out. No template images, no
hardcoded coordinates, no accessibility tree. The same mechanism locates *any*
described icon/button and can handle pop-ups it has never seen (IPA, not
selector-based RPA).

## Quick start (free, no GPU)

The default backend uses one free **Gemini** key for both grounding and planning.

```bash
# Prereqs: a "Notepad" shortcut on the Desktop, a Desktop\tjm-project folder,
#          and a free key from https://aistudio.google.com
uv sync --extra gemini
setx GEMINI_API_KEY "your-key"        # or put GEMINI_API_KEY=... in a .env file
uv run notepad-screenseeker           # -v = verbose, --offline = skip the API
```

Run from a **standalone PowerShell window** (not an editor terminal) and keep hands
off the mouse/keyboard while it runs — corner-slam the mouse to abort.

## How it works

Grounders are weak on full high-res screens (tiny targets) but strong on small
crops, so we **reduce the search space before grounding**:

```
locate: one full-screen grounding; if confident AND verified -> click (1-2 calls)
        else -> recursive search:
          plan candidate regions -> ground each -> dilate/score(Eq.1)/NMS/sort
          -> dive best-first, verify at each leaf, backtrack on failure
```

Two roles, no training: a **planner** (reasons *where*, verifies `is_target` /
`elsewhere` / `not_found`) and a **grounder** (returns a pixel box). The default
Gemini backend plays both; the `"paper"` backend uses OS-Atlas-7B + GPT-4o (GPU).

Grounding is gated by the planner's **`verify()`**, not the VLM's self-reported confidence (which is unreliable — a general VLM often returns `0` even when the box is correct). The search also never dead-ends: if the planner proposes no candidate regions, it grounds the current crop directly.

## Architecture

| Module | Role |
|---|---|
| `search.py` | Engine: fast single-shot path + recursive best-first DFS with 3-way verification. |
| `scoring.py` | Dilation, Eq.1 centrality (σ=0.3), NMS, sort. Unit-tested. |
| `transforms.py` | Per-crop coordinate remapping to original pixels. Unit-tested. |
| `models/gemini_backend.py` | One VLM as both Grounder and Planner (+ `.env` key loading). |
| `models/` | `base` interfaces · `mock` (no-GPU tests) · `hf_grounder` + `openai_planner`. |
| `automation.py` | Screenshot / click / type / save / close; DPI awareness; focused-window guard. |
| `popups.py` | OCR-keyword detection → keystroke; VLM fallback for unknown dialogs; recovery. |
| `data.py` | Post fetch (requests → PowerShell/.NET → embedded) + `Title: … \n\n body`. |
| `eval.py` | Centre-in-box accuracy by position/size/theme; annotated-screenshot renderer. |
| `main.py` | Per-post loop, launch verification, debug capture, circuit breaker, report. |

Full write-up + flow diagram: `Design_Document.docx`.

## Robustness & failure handling

- **DPI awareness** so clicks match the screenshot's coordinate space.
- **Launch verification** — polls until Notepad is focused before typing; `Alt+F4`
  only fires when Notepad is focused (never the desktop → no Shut-Down dialog).
- **Detection failure** → no on-screen action: save debug screenshot, log, skip the
  post, recover, continue.
- **Known pop-ups** → detected by **OCR keyword match (faster than a VLM call)**,
  dismissed with a single `Enter` (default button).
- **Unknown pop-ups** → VLM picks the safest non-destructive button, clicks, verifies,
  logs; `Esc` fallback, safe skip last.
- **Circuit breaker** — 3 consecutive failures abort; otherwise one failure just
  skips that post.
- **Run report** — every run writes a timestamped report to `assets/reports/`
  (saved / skipped-with-reasons / aborted, plus any unknown pop-ups and their
  debug-screenshot paths); it's also printed to the console.

## Performance

Common case is **1–2 model calls** (fast path); the recursive fallback is bounded
(`d_max=2`, `top_k=2`) with a 30 s client timeout. Typing/saving/closing use
keyboard shortcuts — grounding is used only for clicks. Knobs in `config.py`;
`gemini-flash-lite-latest` is a faster drop-in.

## Bonuses

Multiple icons (description + ranking + verify), icon sizes (crop enlarges small
targets), and light/dark themes (described by form, not colour) — all handled by
vision grounding where template/OCR methods break.

## Tests

```bash
uv run pytest        # or:  python run_tests.py   (pytest-free, no GPU/network)
```

20 tests: scoring, coordinate round-trips, and the full search flow (top-left /
centre / bottom-right grounding, ReGround, 3-way verdict) via mock models.

## Deliverables

- [x] Structured source · `uv` config · 20 tests
- [x] Design doc → `Design_Document.docx` (with flow diagram)
- [ ] 3 annotated screenshots (top-left / centre / bottom-right) — via `eval.annotate`
