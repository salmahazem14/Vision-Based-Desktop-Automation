"""Central configuration. Everything the paper leaves as a hyperparameter lives here."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class GroundingConfig:
    # --- ScreenSeekeR search hyperparameters (paper defaults) ---
    d_max: int = 2                 # max recursion depth (2 is plenty for a flat desktop)
    top_k_candidates: int = 2      # candidate regions kept per level (2 caps fallback cost)
    sigma: float = 0.3             # Eq.1 Gaussian centrality width
    dilate_scale: float = 2.5      # grow tiny grounder boxes to ~2-3x
    nms_iou: float = 0.5           # overlap above which NMS drops the weaker box
    ground_threshold_px: int = 1280  # only trust grounder once crop <= this
    reground_crop_px: int = 1024   # ReGround crop size around the coarse guess

    # --- confidence / robustness ---
    min_confidence: float = 0.20   # below this we do not click; retry or fail
    max_ground_retries: int = 2


@dataclass
class AutomationConfig:
    launch_wait_s: float = 1.5     # max wait for Notepad to appear
    action_pause_s: float = 0.3    # small settle after paste/hotkey
    poll_interval_s: float = 0.25  # wait-until-grounded poll cadence
    poll_timeout_s: float = 8.0


@dataclass
class WorkflowConfig:
    api_url: str = "https://jsonplaceholder.typicode.com/posts"
    num_posts: int = 10
    offline: bool = False   # True -> embedded posts, skip network
    save_dir: Path = Path.home() / "Desktop" / "tjm-project"
    backend: str = "gemini"   # "gemini" (one key, both roles) | "paper" (OS-Atlas+GPT-4o)
    icon_target: str = (
        "the Notepad app shortcut icon on the desktop: a small spiral-bound "
        "notepad icon (blue header, white pages) with the text label 'notepad' "
        "beneath it"
    )
    # per-post recovery
    max_consecutive_failures: int = 3   # circuit breaker
    blocking_notify: bool = False       # False = log & continue, report at end


@dataclass
class Config:
    grounding: GroundingConfig = field(default_factory=GroundingConfig)
    automation: AutomationConfig = field(default_factory=AutomationConfig)
    workflow: WorkflowConfig = field(default_factory=WorkflowConfig)
    debug_dir: Path = Path("assets/debug")
