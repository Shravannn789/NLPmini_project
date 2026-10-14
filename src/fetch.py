"""Fetch a news article from a URL and extract its main text.

Uses requests + BeautifulSoup, with a small SSRF guard: only public http(s)
targets are fetched, and loopback / private / link-local addresses are refused
both before the request and after redirects.
"""

from __future__ import annotations

import ipaddress
import re
import socket
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
MAX_CHARS = 20_000


class FetchError(Exception):
    """Friendly, user-facing error for anything wrong with a link."""


def _assert_public(url: str) -> None:
    host = urlparse(url).hostname
    if not host:
        raise FetchError("That does not look like a valid URL.")
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        raise FetchError(f"Could not resolve the domain '{host}'.")
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if (ip.is_private or ip.is_loopback or ip.is_link_local
                or ip.is_reserved or ip.is_multicast):
            raise FetchError("Only publicly reachable article links can be fetched.")


def fetch_article(url: str) -> dict:
    """Return {"title", "text", "final_url"} for the article at ``url``."""
    url = url.strip()
    if not re.match(r"^https?://", url, re.IGNORECASE):
        raise FetchError("Only http:// or https:// links are supported.")
    _assert_public(url)

    try:
        resp = requests.get(url, timeout=12, headers={"User-Agent": USER_AGENT},
                            allow_redirects=True)
    except requests.Timeout:
        raise FetchError("The site took too long to respond.")
    except requests.RequestException as exc:
        raise FetchError(f"Could not fetch the link ({exc.__class__.__name__}).")

    _assert_public(str(resp.url))  # guard the post-redirect target too
    if resp.status_code >= 400:
        raise FetchError(f"The page returned HTTP {resp.status_code}.")

    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "noscript", "nav", "header", "footer",
                     "aside", "form", "svg", "iframe"]):
        tag.decompose()

    title = ""
    og = soup.find("meta", property="og:title")
    if og and og.get("content"):
        title = og["content"].strip()
    elif soup.title and soup.title.string:
        title = soup.title.string.strip()

    node = soup.find("article") or soup.body or soup
    paragraphs = [p.get_text(" ", strip=True) for p in node.find_all("p")]
    text = " ".join(p for p in paragraphs if len(p) > 40)
    if len(text) < 150:  # unusual layout -> fall back to all visible text
        text = node.get_text(" ", strip=True)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) < 80:
        raise FetchError("Could not extract article text from that page. "
                         "Try pasting the text directly.")

    return {"title": title[:300], "text": text[:MAX_CHARS], "final_url": str(resp.url)}
