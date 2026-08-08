"""Data source: fetch blog posts from JSONPlaceholder, with layered fallbacks.

Fetch strategy (each falls back to the next on failure):
  1. requests (normal path).
  2. PowerShell Invoke-WebRequest (.NET stack) -- Windows escape hatch for machines
     where a security filter driver resets OpenSSL/curl handshakes but .NET works.
  3. Embedded offline copy of the real first-10 posts (no external file needed).

`offline=True` skips straight to the embedded copy. Grounding always uses live Gemini.
"""
from __future__ import annotations
import json
import logging
import subprocess
import sys
from dataclasses import dataclass

import requests

log = logging.getLogger(__name__)

# Embedded copy of JSONPlaceholder's first 10 posts -- self-contained fallback,
# so the app never depends on a network or an external fixture file.
_EMBEDDED_POSTS: list[dict] = [
    {"id": 1, "title": "sunt aut facere repellat provident occaecati excepturi optio reprehenderit", "body": "quia et suscipit\nsuscipit recusandae consequuntur expedita et cum\nreprehenderit molestiae ut ut quas totam\nnostrum rerum est autem sunt rem eveniet architecto"},
    {"id": 2, "title": "qui est esse", "body": "est rerum tempore vitae\nsequi sint nihil reprehenderit dolor beatae ea dolores neque\nfugiat blanditiis voluptate porro vel nihil molestiae ut reiciendis\nqui aperiam non debitis possimus qui neque nisi nulla"},
    {"id": 3, "title": "ea molestias quasi exercitationem repellat qui ipsa sit aut", "body": "et iusto sed quo iure\nvoluptatem occaecati omnis eligendi aut ad\nvoluptatem doloribus vel accusantium quis pariatur\nmolestiae porro eius odio et labore et velit aut"},
    {"id": 4, "title": "eum et est occaecati", "body": "ullam et saepe reiciendis voluptatem adipisci\nsit amet autem assumenda provident rerum culpa\nquis hic commodi nesciunt rem tenetur doloremque ipsam iure\nquis sunt voluptatem rerum illo velit"},
    {"id": 5, "title": "nesciunt quas odio", "body": "repudiandae veniam quaerat sunt sed\nalias aut fugiat sit autem sed est\nvoluptatem omnis possimus esse voluptatibus quis\nest aut tenetur dolor neque"},
    {"id": 6, "title": "dolorem eum magni eos aperiam quia", "body": "ut aspernatur corporis harum nihil quis provident sequi\nmollitia nobis aliquid molestiae\nperspiciatis et ea nemo ab reprehenderit accusantium quas\nvoluptate dolores velit et doloremque molestiae"},
    {"id": 7, "title": "magnam facilis autem", "body": "dolore placeat quibusdam ea quo vitae\nmagni quis enim qui quis quo nemo aut saepe\nquidem repellat excepturi ut quia\nsunt ut sequi eos ea sed quas"},
    {"id": 8, "title": "dolorem dolore est ipsam", "body": "dignissimos aperiam dolorem qui eum\nfacilis quibusdam animi sint suscipit qui sint possimus cum\nquaerat magni maiores excepturi\nipsam ut commodi dolore voluptatum modi aut vitae"},
    {"id": 9, "title": "nesciunt iure omnis dolorem tempora et accusantium", "body": "consectetur animi nesciunt iure dolore\nenim quia ad\nveniam autem ut quam aut nobis\net est aut quod aut provident voluptas autem voluptas"},
    {"id": 10, "title": "optio molestias id quia eum", "body": "quo et expedita modi cum officiis vel magni\ndoloribus qui repudiandae\nvero nisi sit\nquos veniam quod sed accusamus veniam culpa"},
]


@dataclass(frozen=True)
class Post:
    id: int
    title: str
    body: str

    def render(self) -> str:
        """Assignment format: 'Title: {title}\\n\\n{body}'."""
        return f"Title: {self.title}\n\n{self.body}"

    @property
    def filename(self) -> str:
        return f"post_{self.id}.txt"


def _to_posts(raw: list[dict], n: int) -> list[Post]:
    return [Post(id=p["id"], title=p["title"], body=p["body"]) for p in raw[:n]]


def _fetch_via_requests(api_url: str, n: int, timeout: float, verify_ssl: bool) -> list[Post]:
    resp = requests.get(api_url, timeout=timeout, verify=verify_ssl)
    resp.raise_for_status()
    return _to_posts(resp.json(), n)


def _fetch_via_powershell(api_url: str, n: int, timeout: float) -> list[Post]:
    """Windows-only escape hatch: fetch through .NET when Python's TLS is reset."""
    if not sys.platform.startswith("win"):
        raise RuntimeError("PowerShell fallback is Windows-only")
    cmd = ["powershell", "-NoProfile", "-NonInteractive", "-Command",
           f"(Invoke-WebRequest -UseBasicParsing -Uri '{api_url}').Content"]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 15)
    if out.returncode != 0 or not out.stdout.strip():
        raise RuntimeError(f"PowerShell fetch failed: {out.stderr.strip()[:200]}")
    return _to_posts(json.loads(out.stdout), n)


def fetch_posts(api_url: str, n: int, timeout: float = 10.0,
                offline: bool = False, verify_ssl: bool = True) -> list[Post]:
    """Return the first n posts, trying live paths before the embedded copy."""
    if offline:
        log.info("offline mode: using embedded posts")
        return _to_posts(_EMBEDDED_POSTS, n)

    try:
        return _fetch_via_requests(api_url, n, timeout, verify_ssl)
    except requests.exceptions.RequestException as e:
        log.warning("requests fetch failed (%s); trying PowerShell (.NET) path", e)

    try:
        posts = _fetch_via_powershell(api_url, n, timeout)
        log.info("fetched live data via PowerShell (.NET) fallback")
        return posts
    except Exception as e:
        log.warning("PowerShell fetch failed (%s); using embedded posts", e)

    return _to_posts(_EMBEDDED_POSTS, n)
