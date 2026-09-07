#!/usr/bin/env python3
"""Fetch the DACON 236749 talkboard and report what changed.

DACON is a Nuxt SSR app: the rendered DOM is not in the HTML, but the page
embeds `window.__NUXT__=(function(a,b,...){...}(args))` which contains the post
list and, on a post page, the post body and replies. This script parses that
state directly -- WebFetch on these URLs returns only the page title.

Usage:
    python3 fetch_talkboard.py --out out/            # fetch + write posts.json
    python3 fetch_talkboard.py --out out/ --snapshot docs/competition/talkboard-snapshot.json
"""
import argparse, html, json, os, re, sys, urllib.request

CPT = "236749"
BASE = f"https://dacon.io/competitions/official/{CPT}"
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36")


def get(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=45) as r:
        return r.read().decode("utf-8", "replace")


def nuxt_body(page: str) -> str:
    i = page.find("window.__NUXT__")
    if i < 0:
        raise RuntimeError("no __NUXT__ state in page (site layout changed?)")
    seg = page[i:page.find("</script>", i)]
    return seg.replace("\\u002F", "/")


def split_args(s: str):
    """Split a JS argument list at top-level commas."""
    out, depth, buf, q, esc = [], 0, [], None, False
    for ch in s:
        if q:
            buf.append(ch)
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == q:
                q = None
            continue
        if ch in "\"'":
            q = ch; buf.append(ch); continue
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            out.append("".join(buf).strip()); buf = []
        else:
            buf.append(ch)
    if buf:
        out.append("".join(buf).strip())
    return out


def lit(tok: str):
    """Best-effort JS literal -> python value."""
    tok = tok.strip()
    if tok.startswith('"'):
        try:
            return json.loads(tok)
        except Exception:
            return tok.strip('"')
    if tok in ("true", "false"):
        return tok == "true"
    if tok in ("null", "void 0", "undefined"):
        return None
    if re.fullmatch(r"-?\d+", tok):
        return int(tok)
    if re.fullmatch(r"-?\d*\.\d+", tok):
        return float(tok)
    return tok


def parse_objects(body: str):
    """Reconstruct the `x.key=value;` object assignments inside the IIFE."""
    m = re.search(r"__NUXT__=\(function\(([^)]*)\)\{", body)
    if not m:
        return {}
    params = [p.strip() for p in m.group(1).split(",")]
    close = body.rfind("}(")
    env = {}
    if close > 0:
        end = body.rfind("))")
        args = split_args(body[close + 2:end if end > close else len(body)])
        for p, a in zip(params, args):
            env[p] = lit(a)

    objs = {}
    for stmt in re.split(r";(?![^\"]*\"(?:[^\"]*\"[^\"]*\")*[^\"]*$)", body):
        mm = re.match(r"\s*([A-Za-z_$][\w$]*)\.([\w]+)=(.*)$", stmt, re.S)
        if not mm:
            continue
        var, key, val = mm.groups()
        val = val.strip()
        v = env[val] if val in env else lit(val)
        objs.setdefault(var, {})[key] = v
    return objs


def strip_html(s: str) -> str:
    s = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", s, flags=re.S | re.I)
    s = re.sub(r"<br\s*/?>|</p>|</li>|</h[1-6]>", "\n", s, flags=re.I)
    s = re.sub(r"<[^>]+>", " ", s)
    s = html.unescape(s)
    s = re.sub(r"[ \t ]+", " ", s)
    return re.sub(r"\n{3,}", "\n\n", s).strip()


def korean_blocks(body: str, minlen: int):
    """All JSON string literals containing Hangul, HTML-stripped, in order."""
    out = []
    for m in re.finditer(r'"((?:[^"\\]|\\.)*)"', body):
        try:
            v = json.loads('"' + m.group(1) + '"')
        except Exception:
            continue
        if len(v) >= minlen and re.search(r"[가-힣]", v):
            t = strip_html(v)
            if t:
                out.append(t)
    return out


def list_posts():
    body = nuxt_body(get(f"{BASE}/talkboard"))
    posts, seen = [], set()
    for var, o in parse_objects(body).items():
        pid = o.get("post_id") or o.get("tb_id")
        title = o.get("title")
        if not isinstance(pid, int) or not isinstance(title, str) or pid in seen:
            continue
        seen.add(pid)
        posts.append({
            "id": pid,
            "title": title,
            "author": o.get("name") or o.get("team_name"),
            "created": o.get("create_time2") or o.get("create_time"),
            "replies": o.get("reply_cnt"),
            "views": o.get("view_cnt"),
            "url": f"{BASE}/talkboard/{pid}",
        })
    posts.sort(key=lambda p: p["id"], reverse=True)
    return posts, set(korean_blocks(body, 40))


def fetch_post(pid: int, boilerplate: set):
    """Post-specific text = Korean blocks on the post page minus those that
    also appear on the index page (competition description, rules, schedule)."""
    body = nuxt_body(get(f"{BASE}/talkboard/{pid}"))
    blocks, seen = [], set()
    for t in korean_blocks(body, 20):
        if t in boilerplate or t in seen:
            continue
        seen.add(t)
        blocks.append(t)
    return blocks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="talkboard_out")
    ap.add_argument("--snapshot", default="docs/competition/talkboard-snapshot.json")
    ap.add_argument("--no-bodies", action="store_true")
    a = ap.parse_args()

    posts, boiler = list_posts()
    if not a.no_bodies:
        for p in posts:
            try:
                p["content"] = fetch_post(p["id"], boiler)
            except Exception as e:
                p["content"] = [f"<fetch failed: {e}>"]

    os.makedirs(a.out, exist_ok=True)
    cur = {"competition": CPT, "posts": posts}
    with open(os.path.join(a.out, "posts.json"), "w", encoding="utf-8") as f:
        json.dump(cur, f, ensure_ascii=False, indent=2)

    old = {}
    if a.snapshot and os.path.exists(a.snapshot):
        with open(a.snapshot, encoding="utf-8") as f:
            old = {p["id"]: p for p in json.load(f).get("posts", [])}

    new_posts = [p for p in posts if p["id"] not in old]
    changed = [p for p in posts if p["id"] in old
               and (p.get("replies") != old[p["id"]].get("replies")
                    or p.get("title") != old[p["id"]].get("title"))]

    print(f"posts: {len(posts)} | new: {len(new_posts)} | reply-count changed: {len(changed)}")
    print(f"snapshot: {'none (first run)' if not old else a.snapshot}")
    for p in new_posts:
        print(f"  NEW      #{p['id']}  r={p['replies']}  {p['title']}")
    for p in changed:
        o = old[p["id"]]
        print(f"  CHANGED  #{p['id']}  replies {o.get('replies')} -> {p.get('replies')}  {p['title']}")
    if not new_posts and not changed:
        print("  no changes")
    print(f"\nwrote {os.path.join(a.out, 'posts.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
