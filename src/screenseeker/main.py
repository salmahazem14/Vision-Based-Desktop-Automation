"""Orchestrator: the per-post loop with per-post recovery + circuit breaker.

    for post in first 10:
        screenshot -> ScreenSeekeR ground the Notepad icon -> double-click
        ensure_clear (launch popup?)
        type formatted post -> save_as post_{id}.txt -> close
    On an UNKNOWN popup: notify + log, attempt verified Esc recovery, else SKIP
    just that post (batch survives). Abort only on N consecutive failures.
"""
from __future__ import annotations
import argparse
import logging
import time
from pathlib import Path

from .config import Config
from .data import fetch_posts, Post
from .search import ScreenSeekeR
from .popups import PopupHandler, RunReport, UnknownBlocker
from . import automation as auto

log = logging.getLogger("screenseeker")


def build_engine(cfg: Config):
    """Wire the real backends. Import here so the sandbox/tests need no heavy deps.

    Default "gemini": ONE Gemini model serves BOTH grounder and planner (one free
    key, no GPU). "paper": OS-Atlas-7B grounder + GPT-4o planner (needs GPU + key).
    """
    if cfg.workflow.backend == "gemini":
        from .models.gemini_backend import GeminiBackend
        backend = GeminiBackend()
        grounder = planner = backend            # same object plays both roles
    elif cfg.workflow.backend == "paper":
        from .models.hf_grounder import HFGrounder
        from .models.openai_planner import OpenAIPlanner
        grounder = HFGrounder()
        planner = OpenAIPlanner()
    else:
        raise ValueError(f"unknown backend: {cfg.workflow.backend!r}")
    engine = ScreenSeekeR(grounder, planner, cfg.grounding,
                          image_provider=_pixel_provider)
    return engine, planner


def _pixel_provider(crop):
    """Return real cropped pixels for the models given a crop Box (original px)."""
    import numpy as np
    full = auto.screenshot()
    h, w = full.shape[:2]
    x1 = max(0, int(crop.x1)); y1 = max(0, int(crop.y1))
    x2 = min(w, int(crop.x2)); y2 = min(h, int(crop.y2))
    return full[y1:y2, x1:x2]


def _save_debug_shot(name: str, cfg: Config, image=None) -> Path:
    """Save a timestamped debug screenshot to the debug dir and return its path.
    If `image` is None, grabs a fresh screenshot. Best-effort (never raises)."""
    cfg.debug_dir.mkdir(parents=True, exist_ok=True)
    path = cfg.debug_dir / f"{name}_{int(time.time())}.png"
    try:
        from PIL import Image
        if image is None:
            image = auto.screenshot()
        Image.fromarray(image).save(path)
    except Exception as e:
        log.warning("could not save debug shot %s: %s", path, e)
    return path


def _launch_and_confirm(center, cfg: Config) -> bool:
    """Double-click the icon and wait for Notepad to become the focused window,
    retrying the double-click once. Returns True once Notepad is in front."""
    for attempt in range(2):
        auto.double_click(*center)
        deadline = time.time() + cfg.automation.launch_wait_s + 2.0
        while time.time() < deadline:
            if auto.notepad_is_focused():
                return True
            time.sleep(0.25)
        log.warning("Notepad not focused after click attempt %d (front='%s')",
                    attempt + 1, auto.foreground_title())
    return False


def process_one_post(post: Post, engine: ScreenSeekeR, popups: PopupHandler,
                     cfg: Config) -> None:
    """Ground+launch, type, save, close. Raises UnknownBlocker on unrecoverable
    popup (caught per-post by run())."""
    auto.press("win", "m")                 # minimize everything -> desktop + icon visible
    time.sleep(0.6)
    full = auto.screenshot()
    h, w = full.shape[:2]

    res = engine.locate(w, h, cfg.workflow.icon_target)
    if not res.found:
        shot = _save_debug_shot(f"no_icon_post{post.id}", cfg, image=full)
        raise UnknownBlocker(shot,
                             f"Notepad icon not grounded (conf={res.confidence:.2f})")
    # Double-click, then poll for Notepad to actually come to the front, retrying
    # the click once. Verify BEFORE typing or Alt+F4 -- if it never launches we
    # skip safely (never type into the void or Alt+F4 the desktop).
    if not _launch_and_confirm(res.box.center, cfg):
        shot = _save_debug_shot(f"no_launch_post{post.id}", cfg)
        raise UnknownBlocker(shot,
                             f"Notepad did not open after click "
                             f"(front='{auto.foreground_title()}') - skipped, no Alt+F4")
    popups.ensure_clear(post_id=post.id)                     # launch popup?

    auto.type_text(post.render(), settle=cfg.automation.action_pause_s)
    save_path = cfg.workflow.save_dir / post.filename

    # Save, then VERIFY the file was actually written before counting success.
    # Retry once; if it still isn't on disk, record it as a real failure (not saved).
    saved_ok = False
    for attempt in range(2):
        auto.save_as(save_path)
        time.sleep(0.5)
        if save_path.exists():
            saved_ok = True
            break
        log.warning("post %d: %s not on disk after save attempt %d - retrying",
                    post.id, save_path.name, attempt + 1)
    if not saved_ok:
        shot = _save_debug_shot(f"no_save_post{post.id}", cfg)
        raise UnknownBlocker(shot, f"{save_path.name} was not written to disk")

    auto.close_window(on_dialog=lambda: popups.ensure_clear(post_id=post.id))


def run(cfg: Config) -> RunReport:
    cfg.workflow.save_dir.mkdir(parents=True, exist_ok=True)
    log.info("Starting in 3s - do NOT touch mouse/keyboard. Corner-slam mouse to abort.")
    time.sleep(3)
    posts = fetch_posts(cfg.workflow.api_url, cfg.workflow.num_posts,
                        offline=getattr(cfg.workflow, 'offline', False))
    engine, planner = build_engine(cfg)
    popups = PopupHandler(
        grounder_locate=lambda w, h, instr: engine.locate(w, h, instr),
        planner=planner, screenshot_fn=auto.screenshot,
        click_fn=auto.left_click, press_fn=auto.press,
        debug_dir=cfg.debug_dir, min_conf=cfg.grounding.min_confidence,
    )

    report = RunReport()
    consecutive = 0
    for post in posts:
        try:
            process_one_post(post, engine, popups, cfg)
            report.saved.append(post.id)
            consecutive = 0
        except UnknownBlocker as e:
            log.warning("post %d hit a blocker: %s (shot: %s)",
                        post.id, e.summary, e.screenshot_path)
            report.skipped.append((post.id, e.summary))
            consecutive += 1
            _recover_to_desktop(popups)                      # get back to a clean state
            if consecutive >= cfg.workflow.max_consecutive_failures:
                report.aborted = True
                report.abort_reason = f"{consecutive} consecutive failures"
                break
        except Exception as e:                               # unexpected
            log.exception("post %d unexpected error", post.id)
            report.skipped.append((post.id, f"error: {e}"))
            consecutive += 1
            if consecutive >= cfg.workflow.max_consecutive_failures:
                report.aborted = True
                report.abort_reason = str(e)
                break

    if popups.unknown_events:
        log.warning("Encountered %d unknown popup(s) this run:", len(popups.unknown_events))
        for ev in popups.unknown_events:
            log.warning("  unknown popup -> %s", ev)

    _write_report(report, cfg, popups)
    log.info("RUN COMPLETE\n%s", report.summary())
    return report


def _write_report(report: RunReport, cfg: Config, popups: PopupHandler) -> Path:
    """Write the final run report to a timestamped text file. Best-effort."""
    from datetime import datetime
    reports_dir = cfg.debug_dir.parent / "reports"     # assets/reports/
    reports_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now()
    path = reports_dir / f"run_report_{ts:%Y%m%d_%H%M%S}.txt"
    lines = [
        "Notepad ScreenSeekeR - Run Report",
        f"Timestamp : {ts:%Y-%m-%d %H:%M:%S}",
        f"Backend   : {cfg.workflow.backend}",
        f"Posts     : {cfg.workflow.num_posts} requested",
        "-" * 48,
        report.summary(),
    ]
    if popups.unknown_events:
        lines.append("-" * 48)
        lines.append(f"Unknown pop-ups encountered: {len(popups.unknown_events)}")
        for ev in popups.unknown_events:
            lines.append(f"  post {ev.get('post_id')}: {ev.get('summary')} "
                         f"| action={ev.get('action')} cleared={ev.get('cleared')} "
                         f"| shot={ev.get('screenshot')}")
    try:
        path.write_text("\n".join(lines), encoding="utf-8")
        log.info("run report written to %s", path)
    except Exception as e:
        log.warning("could not write run report: %s", e)
    return path


def _recover_to_desktop(popups: PopupHandler) -> None:
    """Best-effort: dismiss whatever is up and close stray Notepad windows."""
    try:
        auto.press("esc")
        time.sleep(0.2)
        auto.close_window()
        time.sleep(0.3)
    except Exception:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="ScreenSeekeR Notepad automation")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--offline", action="store_true",
                        help="use embedded posts instead of calling the API")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cfg = Config()
    cfg.workflow.offline = args.offline
    report = run(cfg)
    print("\n=== SUMMARY ===")
    print(report.summary())


if __name__ == "__main__":
    main()
