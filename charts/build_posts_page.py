"""Build the shareable page for both posts straight from posts/*.txt, so page and files never drift.

  .venv/bin/python charts/build_posts_page.py OUT.html
"""
import html
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
esc = html.escape


def read(name):
    return open(os.path.join(ROOT, "posts", name)).read()


def parse_x(text):
    single = re.split(r"^=== SINGLE POST.*?===\s*$", text, flags=re.M)[1]
    parts = re.split(r"^=== (?:THREAD|OPTIONAL REPLY).*?===\s*$", single, flags=re.M)
    single, thread = parts[0], (parts[1] if len(parts) > 1 else "")
    replies = [(n, body.strip()) for n, body in re.findall(r"^--- (\d+) ---\n(.*?)(?=^--- \d+ ---|\Z)", thread, flags=re.M | re.S)]
    return single.strip(), replies


def card(text, limit, platform, label, button, step=""):
    ident = re.sub(r"[^a-z0-9]+", "-", f"{label}-{step}".lower()).strip("-")
    step_html = f'<span class="step">{esc(step)}</span>' if step else ""
    return f"""<article class="post" data-platform="{platform}" data-limit="{limit}">
  <div class="post-head">
    <div class="meta">{step_html}<span class="count" id="count-{ident}">0 / {limit:,}</span><span class="meter"><i></i></span></div>
    <button type="button" class="copy" id="copy-{ident}">{esc(button)}</button>
  </div>
  <div class="text" id="text-{ident}" tabindex="0">{esc(text)}</div>
</article>"""


def post_block(key, x_file, li_file):
    single, replies = parse_x(read(x_file))
    out = [f'<section class="group" aria-labelledby="h-{key}-x"><h3 id="h-{key}-x">X <small>one post, attach the media above</small></h3>',
           card(single, 280, "x", f"{key}-x", "Copy post"), "</section>",
           *([f'<section class="group" aria-labelledby="h-{key}-thread"><h3 id="h-{key}-thread">Optional reply <small>receipts under the post; skip it for one post only</small></h3><div class="thread">',
              *[card(body, 280, "x", f"{key}-reply", "Copy reply") for n, body in replies], "</div></section>"] if replies else []),
           f'<section class="group" aria-labelledby="h-{key}-li"><h3 id="h-{key}-li">LinkedIn <small>attach the same media natively</small></h3>',
           card(read(li_file).strip(), 3000, "linkedin", f"{key}-li", "Copy post"), "</section>"]
    return "\n".join(out)


rows1 = [json.loads(l) for l in open(os.path.join(ROOT, "ports", "results.jsonl"))]
rows6 = [json.loads(l) for l in open(os.path.join(ROOT, "ports", "results_six.jsonl"))]
table = [  # emails/s and decisions/s from ports/results.jsonl (1 question) and ports/results_six.jsonl (6 questions)
    ("Jev · cloud API, 32 requests in flight", "161 ms", "99", "102", "610", "100%"),
    ("Jev · cloud API, 128 requests in flight", "–", "110", "124", "746", "100%"),
    ("Laya · fastest (Neural Engine, multilingual)", "6 ms", "176", "30", "178", "55%"),
    ("Laya · most accurate (Neural Engine, typed-decisions)", "15 ms", "65", "11", "69", "82%"),
]
table_html = "".join(f"<tr><th scope=\"row\">{esc(r[0])}</th>" + "".join(f"<td>{v}</td>" for v in r[1:]) + "</tr>" for r in table)

SOURCES = [
    ("Laya model card, including its “Honest Limits” section", "https://github.com/NandhaKishorM/laya"),
    ("Tetris demo source: “The model cannot count or compare…” (static/demos/tetris.js)", "https://github.com/wdobry/laya-playground"),
    ("Snake demo options: “Safe. Best route to food.” (laya-mlx)", "https://github.com/mizorewww/laya-mlx"),
    ("Head-to-head arena: “Laya’s edge here is speed, not judgment”", "https://github.com/PromptEngineer48/laya-vs-jev-arena"),
    ("Chrome T-Rex head-to-head, lockstep results", "https://github.com/virajbhartiya/laya-vs-jev"),
    ("Tetris arena, paced to the slower side", "https://github.com/dante01yoon/laya-jev-arena"),
    ("Core ML (Neural Engine) port used here", "https://huggingface.co/FluidInference/laya-coreml"),
    ("Community catalog of Laya demos", "https://www.madewithlaya.com/"),
]
sources_html = "".join(f'<li><a href="{u}" target="_blank" rel="noopener">{esc(t)}</a></li>' for t, u in SOURCES)

PAGE = r"""<title>Jev Painting Posts</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
  :root {
    --ground: #EEF3F7; --surface: #FFFFFF; --ink: #14233B; --muted: #55657A; --line: #D3DDE7;
    --accent: #0F4C81; --accent-ink: #FFFFFF; --track: #DFE7EF; --over: #C4221B;
    --display: "Bricolage Grotesque", "Avenir Next", "Segoe UI", system-ui, sans-serif;
    --body: system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    --mono: "IBM Plex Mono", ui-monospace, "SF Mono", Menlo, Consolas, monospace;
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      --ground: #0E1726; --surface: #16233A; --ink: #E6EDF5; --muted: #93A4B8; --line: #283A55;
      --accent: #7FB2E5; --accent-ink: #0E1726; --track: #22324B; --over: #FF7A70;
    }
  }
  :root[data-theme="dark"] {
    --ground: #0E1726; --surface: #16233A; --ink: #E6EDF5; --muted: #93A4B8; --line: #283A55;
    --accent: #7FB2E5; --accent-ink: #0E1726; --track: #22324B; --over: #FF7A70;
  }
  body { background: var(--ground); color: var(--ink); font-family: var(--body); font-size: 15px; line-height: 1.55; }
  .page { max-width: 780px; margin-inline: auto; padding-inline: 20px; padding-block: 44px 72px; display: grid; gap: 40px; }
  .masthead { display: grid; gap: 12px; }
  .eyebrow, .count, .step, .copy, h3, .toc a, .facts { font-family: var(--mono); }
  .eyebrow { margin: 0; font-size: 12px; letter-spacing: 0.09em; text-transform: uppercase; color: var(--muted); }
  h1 { margin: 0; font-family: var(--display); font-weight: 700; font-size: clamp(30px, 6vw, 44px); line-height: 1.05; letter-spacing: -0.02em; text-wrap: balance; }
  .lede { margin: 0; max-width: 60ch; color: var(--muted); font-size: 16px; }
  .toc { display: flex; flex-wrap: wrap; gap: 10px; }
  .toc a { font-size: 13px; color: var(--accent); text-decoration: none; border: 1px solid var(--line); border-radius: 4px; padding: 6px 12px; background: var(--surface); }
  .toc a:hover { border-color: var(--accent); }
  .toc a:focus-visible, .copy:focus-visible, .text:focus-visible, summary:focus-visible, .sources a:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
  .post-section { display: grid; gap: 22px; scroll-margin-top: 16px; }
  .post-section > header { display: grid; gap: 6px; padding-top: 24px; border-top: 2px solid var(--ink); }
  h2 { margin: 0; font-family: var(--display); font-weight: 700; font-size: clamp(24px, 4.4vw, 32px); line-height: 1.1; letter-spacing: -0.015em; text-wrap: balance; }
  .facts { margin: 0; font-size: 12.5px; color: var(--muted); font-variant-numeric: tabular-nums; }
  .media { margin: 0; display: grid; gap: 8px; }
  .media video, .media img { display: block; width: 100%; max-width: 100%; height: auto; border: 1px solid var(--line); border-radius: 6px; background: #15100D; }
  .media video { aspect-ratio: 1200 / 1202; max-width: 560px; }
  .media figcaption { font-size: 13px; color: var(--muted); }
  .group { display: grid; gap: 12px; }
  h3 { margin: 0; font-size: 12px; font-weight: 500; letter-spacing: 0.09em; text-transform: uppercase; color: var(--muted); display: flex; align-items: baseline; gap: 10px; flex-wrap: wrap; }
  h3 small { font-size: 12px; letter-spacing: 0; text-transform: none; }
  .post { background: var(--surface); border: 1px solid var(--line); border-radius: 6px; display: grid; }
  .post-head { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 10px 16px; padding: 10px 16px; border-bottom: 1px solid var(--line); }
  .meta { display: flex; align-items: center; gap: 12px; min-width: 0; }
  .step { font-size: 13px; font-weight: 500; color: var(--accent); }
  .count { font-size: 12.5px; color: var(--muted); font-variant-numeric: tabular-nums; white-space: nowrap; }
  .meter { width: 84px; height: 4px; border-radius: 2px; background: var(--track); overflow: hidden; flex: none; }
  .meter i { display: block; height: 100%; width: 0; background: var(--accent); }
  .post.over .count { color: var(--over); }
  .post.over .meter i { background: var(--over); }
  .copy { appearance: none; border: 0; border-radius: 4px; padding: 8px 14px; font-size: 12.5px; font-weight: 500; background: var(--accent); color: var(--accent-ink); cursor: pointer; transition: filter 0.15s ease; }
  .copy:hover { filter: brightness(1.12); }
  .copy.done { background: transparent; color: var(--accent); box-shadow: inset 0 0 0 1px var(--accent); }
  .text { margin: 0; padding: 14px 16px; white-space: pre-wrap; overflow-wrap: anywhere; max-width: 68ch; }
  .thread { position: relative; display: grid; gap: 12px; padding-left: 22px; }
  .thread::before { content: ""; position: absolute; left: 6px; top: 0; bottom: 24px; width: 2px; background: var(--line); }
  .thread .post { position: relative; }
  .thread .post::before { content: ""; position: absolute; left: -21px; top: 18px; width: 10px; height: 10px; border-radius: 50%; background: var(--ground); border: 2px solid var(--accent); box-sizing: border-box; }
  details { background: var(--surface); border: 1px solid var(--line); border-radius: 6px; padding: 10px 16px; }
  summary { cursor: pointer; font-size: 14px; color: var(--accent); }
  .table-wrap { overflow-x: auto; margin-top: 10px; }
  table { border-collapse: collapse; width: 100%; font-size: 13.5px; }
  th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--line); vertical-align: top; }
  td { font-family: var(--mono); font-variant-numeric: tabular-nums; white-space: nowrap; }
  thead th { font-size: 12px; color: var(--muted); font-weight: 500; }
  .notes ul, .sources ul { margin: 0; padding-left: 18px; display: grid; gap: 8px; max-width: 66ch; }
  .notes li::marker, .sources li::marker { color: var(--accent); }
  .sources a { color: var(--accent); }
  .sr { position: absolute; width: 1px; height: 1px; overflow: hidden; clip-path: inset(50%); white-space: nowrap; }
  @media (prefers-reduced-motion: reduce) { .copy { transition: none; } }
</style>

<div class="page">
  <header class="masthead">
    <p class="eyebrow">Ready to post · copy buttons count characters the way X does</p>
    <h1>Jev Painting Posts</h1>
    <p class="lede">Post 1 shows Jev painting a Bob Ross landscape one decision at a time. Post 2 puts Jev next to Laya, its open local rival, and measures where each one wins.</p>
    <nav class="toc" aria-label="Posts"><a href="#post1">Post 1 · Jev paints Bob Ross</a><a href="#post2">Post 2 · Latency is not throughput</a></nav>
  </header>

  <section class="post-section" id="post1" aria-labelledby="h-post1">
    <header>
      <h2 id="h-post1">Post 1 · An AI that can’t draw paints Bob Ross</h2>
      <p class="facts">512 × 512 · 37,856 decisions · 34.9 s · $0.244 · 99.81% of pixels right · video replayed at 4×</p>
    </header>
    <figure class="media">
      <video id="post1-video" controls playsinline muted loop preload="metadata" src="post1_jev_bob_ross_4x.mp4"></video>
      <figcaption>Your screen recording, re-encoded at 60 fps for X (it was 120 fps). 9.7 s, 1200 × 1202. The file is videos/post1_jev_bob_ross_4x.mp4.</figcaption>
    </figure>
    __POST1__
    <section class="group notes" aria-labelledby="h-post1-notes">
      <h3 id="h-post1-notes">Before you post</h3>
      <ul>
        <li><strong>Upload the video natively</strong> on both platforms; the page copy plays inline for review only.</li>
        <li><strong>Tags are verified:</strong> @typesafeai is linked from typesafe.ai; @CompleteSkeptic is Diogo Almeida, TypeSafe’s co-founder and CEO (RuntimeWire’s launch article and his AI Engineer speaker page).</li>
        <li><strong>Every number is from the run log:</strong> 37,856 decisions, 34.9 s, $0.2440, 99.81%.</li>
      </ul>
    </section>
  </section>

  <section class="post-section" id="post2" aria-labelledby="h-post2">
    <header>
      <h2 id="h-post2">Post 2 · Latency is not throughput</h2>
      <p class="facts">Measured 24 Sep 2026 · M3 Pro laptop vs Jev via OpenRouter · 256 emails × 6 questions · all Jev benchmarks: $0.04</p>
    </header>
    <figure class="media">
      <img id="post2-chart" src="post2_latency_vs_throughput.png" alt="Bar chart. One decision: Jev 161 ms, Laya fastest 6 ms, Laya most accurate 15 ms. Bulk job, 256 emails with 6 questions each: Jev 746 decisions per second, Laya fastest 178, Laya most accurate 69. Emails sent to the right team: Jev 100%, Laya fastest 55%, Laya most accurate 82%." width="1600" height="900">
      <figcaption>Attach this image to the X post and the LinkedIn post. The file is posts/media/post2_latency_vs_throughput.png (3200 × 1800).</figcaption>
    </figure>
    <details>
      <summary>The numbers behind the chart</summary>
      <div class="table-wrap">
        <table>
          <thead><tr><th scope="col">Setup</th><th scope="col">One decision, median</th><th scope="col">Emails/s, 1 question each</th><th scope="col">Emails/s, 6 questions each</th><th scope="col">Decisions/s, 6 questions each</th><th scope="col">Right team</th></tr></thead>
          <tbody>__TABLE__</tbody>
        </table>
      </div>
    </details>
    __POST2__
    <section class="group sources" aria-labelledby="h-post2-sources">
      <h3 id="h-post2-sources">Sources for the claims</h3>
      <ul>__SOURCES__</ul>
    </section>
    <section class="group notes" aria-labelledby="h-post2-notes">
      <h3 id="h-post2-notes">Before you post</h3>
      <ul>
        <li><strong>Keep the credit to Laya.</strong> The criticism targets the viral framing, not the model; its README states these limits.</li>
        <li><strong>Jev’s bulk figure is a 2-second burst</strong> at 128 requests in flight, not a ceiling. The 100,000-email line is an estimate at the measured rates, and says so.</li>
        <li><strong>The emails are synthetic,</strong> written for one team each. Real inboxes are messier for both models.</li>
      </ul>
    </section>
  </section>

  <p class="sr" id="status" aria-live="polite"></p>
</div>

<script>
  (function () {
    var ONES = [[0, 4351], [8192, 8205], [8208, 8223], [8242, 8247]];
    function xWeight(text) {
      var total = 0;
      Array.from(text).forEach(function (ch) {
        var cp = ch.codePointAt(0);
        total += ONES.some(function (r) { return cp >= r[0] && cp <= r[1]; }) ? 1 : 2;
      });
      return total;
    }
    function legacyCopy(text) {
      var ta = document.createElement("textarea");
      ta.value = text; ta.setAttribute("readonly", ""); ta.style.position = "fixed"; ta.style.opacity = "0";
      document.body.appendChild(ta); ta.select();
      var ok = false;
      try { ok = document.execCommand("copy"); } catch (e) { ok = false; }
      document.body.removeChild(ta);
      return ok;
    }
    function copyText(text) {
      if (navigator.clipboard && navigator.clipboard.writeText) {
        return navigator.clipboard.writeText(text).then(function () { return true; }, function () { return legacyCopy(text); });
      }
      return Promise.resolve(legacyCopy(text));
    }
    var status = document.getElementById("status");
    document.querySelectorAll(".post").forEach(function (post) {
      var textEl = post.querySelector(".text");
      var text = textEl.textContent.trim();
      var limit = Number(post.dataset.limit);
      var count = post.dataset.platform === "x" ? xWeight(text.replace(/https?:\/\/\S+/g, "xxxxxxxxxxxxxxxxxxxxxxx")) : Array.from(text).length;
      post.querySelector(".count").textContent = count.toLocaleString("en-US") + " / " + limit.toLocaleString("en-US");
      post.querySelector(".meter i").style.width = Math.min(100, Math.round(count / limit * 100)) + "%";
      post.classList.toggle("over", count > limit);
      var button = post.querySelector(".copy");
      var label = button.textContent;
      button.addEventListener("click", function () {
        copyText(text).then(function (ok) {
          if (!ok) {
            var range = document.createRange();
            range.selectNodeContents(textEl);
            var selection = window.getSelection();
            selection.removeAllRanges(); selection.addRange(range);
          }
          button.textContent = ok ? "Copied" : "Selected. Press Cmd+C";
          button.classList.toggle("done", ok);
          status.textContent = ok ? "Copied to the clipboard." : "Text selected. Press Command C to copy.";
          window.setTimeout(function () { button.textContent = label; button.classList.remove("done"); }, 1800);
        });
      });
    });
  })();
</script>
"""

page = (PAGE.replace("__POST1__", post_block("post1", "x_how_it_works.txt", "linkedin_how_it_works.txt"))
            .replace("__POST2__", post_block("post2", "x_latency_vs_throughput.txt", "linkedin_latency_vs_throughput.txt"))
            .replace("__TABLE__", table_html).replace("__SOURCES__", sources_html))
open(sys.argv[1], "w").write(page)
print(f"wrote {sys.argv[1]} ({len(page):,} bytes)")
