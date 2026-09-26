#!/usr/bin/env python3
"""Make the images in docs/media: the README's explainer, the demo video and GIF, screenshots of
the report and the social preview.

    pip install playwright && python scripts/record_demo.py --chrome

Everything comes from a real run of `callreplay demo`. The terminal in the video is the command's
real output, captured through a pseudo-terminal with its timing, progress bar included. The
report is the real HTML report, driven with a visible cursor. Frames come from Chrome's
screencast and are resampled to a constant 30 fps with ffmpeg (which must be on the PATH).
"""
from __future__ import annotations

import argparse
import base64
import fcntl
import html
import json
import os
import pty
import re
import select
import struct
import subprocess
import sys
import tempfile
import termios
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MEDIA = ROOT / 'docs' / 'media'
W, H = 1000, 625                      # the GIF's own size: shown at ~830px on GitHub, text stays readable
EXAMPLE_CALL = 'call-003'             # the regression the explainer shows

CAPTIONS = {
    'run': 'Replay 42 recorded calls against a new prompt',
    'report': 'What broke and what got fixed, grouped by cause',
    'regressed': 'Side by side: the new prompt books without checking the calendar',
    'fixed': 'Fixed: it no longer offers a time the calendar never returned',
    'apart': 'Silent calls and provider errors are kept apart, not counted as failures',
    'end': 'callreplay · open source · MIT',
}

OVERLAY_JS = r"""
(() => {
  if (window.__demo) return;
  const css = document.createElement('style');
  css.textContent = `
    #demo-cursor{position:fixed;left:0;top:0;width:20px;height:20px;z-index:99999;pointer-events:none;transform:translate(-3px,-2px)}
    #demo-cursor svg{width:20px;height:20px;filter:drop-shadow(0 2px 3px rgba(0,0,0,.35))}
    .demo-ripple{position:fixed;z-index:99998;width:30px;height:30px;margin:-15px 0 0 -15px;border-radius:50%;
      pointer-events:none;border:3px solid #3b5bdb;animation:demo-rip .5s ease-out forwards}
    @keyframes demo-rip{from{transform:scale(.3);opacity:1}to{transform:scale(1.4);opacity:0}}
    #demo-caption{position:fixed;left:50%;bottom:26px;transform:translate(-50%,12px);z-index:99990;opacity:0;
      background:rgba(17,17,20,.93);color:#fff;font:600 17px/1.35 Inter,system-ui,sans-serif;letter-spacing:-.01em;
      padding:10px 20px;border-radius:12px;box-shadow:0 18px 50px rgba(0,0,0,.35);transition:all .35s;max-width:90vw;
      text-align:center;white-space:nowrap}
    #demo-caption.on{opacity:1;transform:translate(-50%,0)}`;
  document.head.append(css);
  const cur = document.createElement('div'); cur.id = 'demo-cursor';
  cur.innerHTML = '<svg viewBox="0 0 24 24"><path d="M3 2l7.5 19 2.6-7.4L20.5 11z" fill="#111" stroke="#fff" stroke-width="1.6" stroke-linejoin="round"/></svg>';
  const cap = document.createElement('div'); cap.id = 'demo-caption';
  document.body.append(cur, cap);
  addEventListener('mousemove', e => { cur.style.left = e.clientX + 'px'; cur.style.top = e.clientY + 'px'; }, true);
  addEventListener('mousedown', e => {
    const r = document.createElement('div'); r.className = 'demo-ripple';
    r.style.left = e.clientX + 'px'; r.style.top = e.clientY + 'px';
    document.body.append(r); setTimeout(() => r.remove(), 600);
  }, true);
  window.__demo = { caption(t) { if (!t) return cap.classList.remove('on'); cap.textContent = t; cap.classList.add('on'); } };
})();
"""


# --------------------------------------------------------------------------- the real run
def run_demo(cwd: Path) -> dict:
    """`callreplay demo` in a neutral folder (no local path shows), for the report and the stills."""
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    env.pop('GITHUB_STEP_SUMMARY', None)
    subprocess.run([sys.executable, '-m', 'callreplay', 'demo', '--no-open', '--out', 'callreplay-demo'],
                   capture_output=True, text=True, env=env, cwd=cwd, check=True)
    return json.loads((cwd / 'callreplay-demo' / 'results.json').read_text(encoding='utf-8'))


def run_demo_in_a_terminal(cwd: Path, cols: int = 108, rows: int = 40) -> list[tuple[float, str]]:
    """The same command in a pseudo-terminal: what a person sees, bytes and timing."""
    pid, fd = pty.fork()
    if pid == 0:
        os.chdir(cwd)
        env = dict(os.environ, PYTHONPATH=str(ROOT), TERM='xterm-256color', COLUMNS=str(cols), LINES=str(rows))
        for k in ('NO_COLOR', 'FORCE_COLOR', 'GITHUB_STEP_SUMMARY'):
            env.pop(k, None)
        os.execvpe(sys.executable, [sys.executable, '-m', 'callreplay', 'demo', '--no-open', '--out', 'callreplay-demo'], env)
    fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack('HHHH', rows, cols, 0, 0))
    chunks, t0 = [], time.monotonic()
    while True:
        try:
            ready, _, _ = select.select([fd], [], [], 10)
            data = os.read(fd, 65536) if ready else b''
        except OSError:
            break
        if not data:
            break
        chunks.append((time.monotonic() - t0, data.decode('utf-8', 'replace')))
    os.waitpid(pid, 0)
    return chunks


def screens(chunks: list[tuple[float, str]]) -> list[tuple[float, list[str]]]:
    """Replay the bytes like a terminal would: lines, colours, and the progress bar redrawn in place."""
    lines, out = [''], []
    for t, text in chunks:
        for tok in re.findall(r'\x1b\[[0-9;]*[A-Za-z]|\r|\n|[^\x1b\r\n]+', text):
            if tok == '\n':
                lines.append('')
            elif tok == '\x1b[2K':
                lines[-1] = ''
            elif tok == '\r' or (tok.startswith('\x1b[') and not tok.endswith('m')):
                continue
            else:
                lines[-1] += tok
        out.append((t, [re.sub(r'/\S*/callreplay-demo/', 'callreplay-demo/', x) for x in lines]))
    return out


ANSI = {'1': 'b', '2': 'dim', '31': 'red', '32': 'green', '33': 'yellow', '34': 'blue', '35': 'magenta',
        '36': 'cyan', '90': 'grey'}


def ansi_to_html(line: str) -> str:
    out, open_spans = [], 0
    for part in re.split(r'(\033\[[\d;]*m)', line):
        m = re.fullmatch(r'\033\[([\d;]*)m', part)
        if not m:
            out.append(html.escape(part))
        elif m.group(1) in ('', '0'):
            out.append('</span>' * open_spans)
            open_spans = 0
        else:
            out.append(f'<span class="{" ".join(ANSI.get(c, "") for c in m.group(1).split(";"))}">')
            open_spans += 1
    return ''.join(out) + '</span>' * open_spans


def terminal_page(frames: list[tuple[float, list[str]]]) -> str:
    data = json.dumps([[round(t, 3), [ansi_to_html(x) for x in lines]] for t, lines in frames])
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>
body{{margin:0;background:#e9e9e4;height:100vh;display:flex;align-items:center;justify-content:center;
  font-family:-apple-system,BlinkMacSystemFont,Inter,sans-serif}}
.win{{width:{W - 36}px;height:{H - 34}px;background:#15161a;border-radius:12px;box-shadow:0 24px 60px rgba(0,0,0,.28);overflow:hidden;
  display:flex;flex-direction:column}}
.bar{{height:32px;background:#202127;display:flex;align-items:center;gap:7px;padding:0 14px;flex:none}}
.bar i{{width:11px;height:11px;border-radius:50%;display:block}}
.bar span{{color:#8b8d96;font-size:12px;margin-left:auto;margin-right:auto;transform:translateX(-26px)}}
pre{{margin:0;padding:10px 16px;color:#d8d9de;font:12.2px/1.38 "SF Mono",SFMono-Regular,Menlo,monospace;flex:1;overflow:hidden;
  white-space:pre;display:flex;flex-direction:column;justify-content:flex-start}}
.b{{font-weight:700;color:#fff}} .dim{{color:#80838d}} .grey{{color:#5d6069}} .red{{color:#ff6b8a}} .green{{color:#5ad17a}}
.yellow{{color:#f5c04a}} .blue{{color:#6aa8ff}} .magenta{{color:#b69cff}} .cyan{{color:#5fd3e0}}
.b.red{{color:#ff6b8a}} .b.green{{color:#5ad17a}} .b.magenta{{color:#b69cff}}
.p{{color:#5ad17a}} .c{{display:inline-block;width:8px;height:16px;background:#d8d9de;vertical-align:-3px;animation:blink 1s steps(1) infinite}}
@keyframes blink{{50%{{opacity:0}}}}
</style></head><body><div class="win"><div class="bar"><i style="background:#ff5f57"></i><i style="background:#febc2e"></i>
<i style="background:#28c840"></i><span>~/my-voice-agent</span></div><pre id="t"></pre></div>
<script>
const FRAMES = {data};
const t = document.getElementById('t');
const prompt = '<span class="p">$</span> ';
const fit = lines => {{            // whole lines only, like a terminal that has scrolled
  const cs = getComputedStyle(t);
  const rows = Math.floor((t.clientHeight - parseFloat(cs.paddingTop) - parseFloat(cs.paddingBottom)) / parseFloat(cs.lineHeight));
  return lines.slice(-rows);
}};
window.typeCommand = async (cmd) => {{
  t.innerHTML = '<div>' + prompt + '<span id="cmd"></span><span class="c"></span></div>';
  for (const ch of cmd) {{ document.getElementById('cmd').textContent += ch; await new Promise(r => setTimeout(r, 55)); }}
}};
window.play = async () => {{
  const head = prompt + document.getElementById('cmd').textContent;
  const start = performance.now();
  for (const [at, lines] of FRAMES) {{
    const wait = at * 1000 - (performance.now() - start);
    if (wait > 0) await new Promise(r => setTimeout(r, wait));
    t.innerHTML = fit([head, ...lines]).map(l => '<div>' + (l || ' ') + '</div>').join('');
  }}
}};
</script></body></html>"""


# --------------------------------------------------------------------------- the explainer
def _args(name: str, args: dict) -> str:
    """Arguments as a person reads them: Fri 15:00, not date="2026-09-18", time="15:00"."""
    a = {k: v for k, v in args.items() if k not in ('phone', 'reason')}
    day = date.fromisoformat(a.pop('date')).strftime('%a') if 'date' in a else ''
    when = ' '.join(x for x in (day, a.pop('time', '')) if x)
    if name == 'create_booking':
        a.pop('service', None)
    first = [str(a.pop('name'))] if 'name' in a else []
    return ', '.join(first + ([when] if when else []) + [str(v) for v in a.values()])


def _result(result) -> str:
    if isinstance(result, dict) and 'slots' in result:
        return ' · '.join(result['slots']) or 'no slots'
    if isinstance(result, dict) and result.get('booking_id'):
        return f"booked {result['booking_id']}"
    return ''


def _opening(conv: dict, replay: bool) -> list[str]:
    """The call up to the agent's answer to the caller's request: what the explainer shows."""
    rows, e = [], html.escape
    results = {t['tool_call_id']: t for t in conv['turns'] if t['role'] == 'tool'}
    users = 0
    for t in conv['turns'][1:]:                          # the greeting says nothing here
        if t['role'] == 'user':
            users += 1
            if users > 1:
                break
            same = '<span class="same">same words</span>' if replay else ''
            rows.append(f'<div class="r">Caller</div><div class="u">{e(t["text"])}{same}</div>')
        elif t['role'] == 'agent':
            for c in t.get('tool_calls') or []:
                res = results.get(c['id'])
                out = _result(res['result']) if res else ''
                src = ' <span class="src">recorded result</span>' if replay and res and res.get('source') else ''
                rows.append(f'<div class="r">Tool</div><div class="c"><b>{e(c["name"])}</b><span class="args">({e(_args(c["name"], c["args"]))})</span>'
                            + (f'<span class="res">→ {e(out)}{src}</span>' if out else '') + '</div>')
            if t.get('text'):
                rows.append(f'<div class="r">Agent</div><div class="a">{e(t["text"])}</div>')
    return rows


def explainer_page(data: dict, dark: bool) -> str:
    """One real call of the demo: recorded, replayed, and why the replay regressed."""
    it = next(i for i in data['items'] if i['id'] == EXAMPLE_CALL)
    s, ch = data['summary'], data['summary']['changes']
    rec = _opening(it['reference_conversation'], False)
    rep = _opening(it['replay_conversation'], True)
    missing = [a for a, b in it['tool_diff'] if a and not b]
    for n, name in enumerate(missing):                   # where the recording had it
        rep.insert(1 + n, f'<div class="r">Tool</div><div class="c gap"><b>{html.escape(name)}</b><span class="skip">never called</span></div>')
    finding = next(f for f in it['replay']['findings'] if f['code'] == 'missing_required_tool')
    intent = it['reference']['intent']
    required = ', '.join(f'"{t}"' for t in ('check_availability', 'create_booking'))
    pct = lambda v: f'{round(v * 100)}%'
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>
:root {{
  --bg:#ffffff; --ink:#1f2328; --muted:#59636e; --faint:#8b949e; --line:#d1d9e0; --card:#f6f8fa; --u:#ffffff; --a:#eef3ff;
  --tool:#ffffff; --accent:#3b5bdb; --pass:#1a7f37; --pass-soft:#dafbe1; --fail:#cf222e; --fail-soft:#ffebe9;
  --grey-soft:#eaeef2; --err:#8250df; --err-soft:#fbefff;
}}
.dark {{
  --bg:#0d1117; --ink:#e6edf3; --muted:#9198a1; --faint:#6e7681; --line:#30363d; --card:#151b23; --u:#0d1117; --a:#15213d;
  --tool:#0d1117; --accent:#8ea4ff; --pass:#3fb950; --pass-soft:#12261e; --fail:#ff7b72; --fail-soft:#2d1517;
  --grey-soft:#21262d; --err:#bc8cff; --err-soft:#231a33;
}}
* {{ box-sizing:border-box }}
body {{ margin:0; background:var(--bg); color:var(--ink); font:15px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,sans-serif;
  -webkit-font-smoothing:antialiased }}
.hero {{ width:960px; padding:22px 24px 24px; background:var(--bg) }}
.top {{ display:grid; grid-template-columns:1fr 64px 1fr }}
.head {{ display:flex; align-items:center; gap:10px; margin:0 0 10px }}
.n {{ width:24px; height:24px; border-radius:50%; background:var(--ink); color:var(--bg); font:700 13px/24px -apple-system,sans-serif;
  text-align:center; flex:none }}
.t {{ font-weight:650; font-size:16px; letter-spacing:-.01em }}
.s {{ color:var(--muted); font-size:13px }}
.pill {{ margin-left:auto; font:700 12px/1 -apple-system,sans-serif; text-transform:uppercase; letter-spacing:.05em; padding:6px 10px;
  border-radius:99px }}
.pill.pass {{ background:var(--pass-soft); color:var(--pass) }} .pill.fail {{ background:var(--fail-soft); color:var(--fail) }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:12px; padding:12px 14px; display:grid;
  grid-template-columns:50px 1fr; gap:8px 10px; align-items:start }}
.card.bad {{ border-color:color-mix(in srgb, var(--fail) 45%, var(--line)) }}
.r {{ color:var(--faint); font-size:11px; font-weight:650; text-transform:uppercase; letter-spacing:.05em; padding-top:6px }}
.u, .a {{ padding:6px 10px; border-radius:10px; font-size:14.5px }}
.u {{ background:var(--u); border:1px solid var(--line) }}
.a {{ background:var(--a) }}
.same {{ display:inline-block; margin-left:8px; font-size:11px; font-weight:650; color:var(--muted); background:var(--grey-soft);
  border-radius:5px; padding:1px 6px; vertical-align:1px }}
.c {{ font:13px/1.5 ui-monospace,"SF Mono",SFMono-Regular,Menlo,monospace; background:var(--tool); border:1px solid var(--line);
  border-radius:8px; padding:5px 9px }}
.c b {{ color:var(--accent); font-weight:650 }}
.c .args {{ color:var(--muted) }}
.c .res {{ display:block; color:var(--muted); font-size:12px }}
.c .src {{ font:650 10px -apple-system,sans-serif; text-transform:uppercase; letter-spacing:.05em; color:var(--muted);
  background:var(--grey-soft); border-radius:4px; padding:1px 5px; margin-left:6px; vertical-align:1px }}
.c.gap {{ background:transparent; border:1.5px dashed var(--fail); color:var(--fail) }}
.c.gap b {{ color:var(--fail); text-decoration:line-through; text-decoration-thickness:1.5px }}
.c.gap .skip {{ margin-left:10px; font:700 11px -apple-system,sans-serif; text-transform:uppercase; letter-spacing:.05em }}
.mid {{ display:flex; flex-direction:column; align-items:center; justify-content:center; gap:6px; color:var(--muted); font-size:11px;
  text-align:center; padding-top:34px }}
.mid svg {{ width:40px; height:24px }}
.verdict {{ margin-top:14px; display:flex; gap:12px; align-items:center; border:1px solid var(--line); border-left:4px solid var(--fail);
  border-radius:12px; padding:12px 16px; background:var(--card) }}
.badge {{ display:inline-block; background:var(--fail); color:#fff; font:700 12px/1 -apple-system,sans-serif; letter-spacing:.06em;
  text-transform:uppercase; padding:6px 9px; border-radius:6px; margin-right:8px; vertical-align:1px }}
.dark .badge {{ color:#0d1117 }}
.vt {{ font-size:15.5px; font-weight:600 }}
.rule {{ font-weight:400; color:var(--muted); font-size:13px; margin-top:5px }}
.rule code {{ font:12.5px ui-monospace,"SF Mono",Menlo,monospace; color:var(--ink); background:var(--grey-soft); padding:2px 6px;
  border-radius:5px }}
.all {{ margin-top:14px; display:flex; flex-wrap:wrap; gap:8px; align-items:center; font-size:14px }}
.lbl {{ font-weight:650; margin-right:4px }}
.chip {{ border-radius:99px; padding:4px 11px; font-weight:600; font-size:13.5px; background:var(--grey-soft); color:var(--muted) }}
.chip.red {{ background:var(--fail-soft); color:var(--fail) }} .chip.green {{ background:var(--pass-soft); color:var(--pass) }}
.chip.purple {{ background:var(--err-soft); color:var(--err) }}
.rate {{ margin-left:auto; color:var(--muted) }} .rate b {{ color:var(--ink) }}
</style></head><body class="{'dark' if dark else ''}"><div class="hero">
<div class="top">
  <div>
    <div class="head"><span class="n">1</span><div><div class="t">A recorded call</div><div class="s">what the agent did with the current prompt</div></div>
      <span class="pill pass">{it['reference']['status']}</span></div>
    <div class="card">{''.join(rec)}</div>
  </div>
  <div class="mid"><svg viewBox="0 0 40 24"><path d="M2 12h32m-9-8 9 8-9 8" fill="none" stroke="currentColor" stroke-width="2.2"
    stroke-linecap="round" stroke-linejoin="round"/></svg>replayed<br>on a new<br>prompt</div>
  <div>
    <div class="head"><span class="n">2</span><div><div class="t">The same call, replayed</div><div class="s">the caller's words are fixed, the agent answers anew</div></div>
      <span class="pill fail">{it['replay']['status']}</span></div>
    <div class="card bad">{''.join(rep)}</div>
  </div>
</div>
<div class="verdict"><span class="n">3</span><div class="vt"><span class="badge">✕ Regressed</span>{html.escape(finding['message'])}
  <div class="rule">checked against the contract: <code>[intents.{intent}] requires = [{required}]</code></div></div></div>
<div class="all"><span class="lbl">All {s['total']} calls:</span>
  <span class="chip red">{ch.get('regressed', 0)} regressed</span><span class="chip green">{ch.get('fixed', 0)} fixed</span>
  <span class="chip">{ch.get('better', 0)} better</span><span class="chip">{ch.get('same', 0)} same</span>
  <span class="chip">{ch.get('unscorable', 0)} nothing to judge</span><span class="chip purple">{ch.get('error', 0)} provider error</span>
  <span class="rate">pass rate {pct(s['recorded_pass_rate_same_set'])} → <b>{pct(s['replay_pass_rate'])}</b></span></div>
</div></body></html>"""


# --------------------------------------------------------------------------- video
class Recorder:
    """Chrome DevTools screencast: every painted frame with its timestamp."""

    def __init__(self, page, folder: Path, scale: int):
        self.folder, self.frames = folder, []
        self.cdp = page.context.new_cdp_session(page)
        self.cdp.on('Page.screencastFrame', self.frame)
        self.cdp.send('Page.startScreencast', {'format': 'jpeg', 'quality': 94, 'maxWidth': W * scale,
                                               'maxHeight': H * scale, 'everyNthFrame': 1})

    def frame(self, ev):
        path = self.folder / f'{len(self.frames):06d}.jpg'
        path.write_bytes(base64.b64decode(ev['data']))
        self.frames.append((ev['metadata']['timestamp'], path))
        try:
            self.cdp.send('Page.screencastFrameAck', {'sessionId': ev['sessionId']})
        except Exception:  # noqa: BLE001
            pass

    def stop(self):
        self.cdp.send('Page.stopScreencast')

    def encode(self, out: Path, fps: int = 30, tail: float = 1.2) -> None:
        """Frames only arrive when something changes: resample to a constant rate."""
        self.frames.sort()
        t0 = self.frames[0][0]
        times = [t - t0 for t, _ in self.frames]
        ff = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'image2pipe', '-framerate', str(fps), '-i', '-',
                               '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p', '-c:v', 'libx264',
                               '-preset', 'slow', '-crf', '20', '-movflags', '+faststart', str(out)], stdin=subprocess.PIPE)
        k, cache = 0, {}
        for n in range(int((times[-1] + tail) * fps)):
            while k + 1 < len(times) and times[k + 1] <= n / fps:
                k += 1
            path = self.frames[k][1]
            if path not in cache:
                cache = {path: path.read_bytes()}
            ff.stdin.write(cache[path])
        ff.stdin.close()
        if ff.wait():
            raise RuntimeError('ffmpeg failed')


def gif(mp4: Path, out: Path, width: int = W, fps: int = 12) -> None:
    pal = out.with_suffix('.png')
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', str(mp4), '-vf',
                    f'fps={fps},scale={width}:-1:flags=lanczos,palettegen=stats_mode=diff', str(pal)], check=True)
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', str(mp4), '-i', str(pal), '-lavfi',
                    f'fps={fps},scale={width}:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle',
                    str(out)], check=True)
    pal.unlink()


def record(chrome: bool) -> None:
    from playwright.sync_api import sync_playwright
    work = Path(tempfile.mkdtemp(prefix='callreplay-video-'))
    frames_term = screens(run_demo_in_a_terminal(work))
    (work / 'terminal.html').write_text(terminal_page(frames_term), encoding='utf-8')
    report = work / 'callreplay-demo' / 'report.html'
    frames = work / 'frames'
    frames.mkdir()
    with sync_playwright() as p:
        browser = p.chromium.launch(channel='chrome' if chrome else None)
        page = browser.new_page(viewport={'width': W, 'height': H}, color_scheme='light', device_scale_factor=2)

        def wait(s): page.wait_for_timeout(int(s * 1000))

        def caption(key=None, hold=0.0):
            page.evaluate('t => window.__demo && window.__demo.caption(t)', CAPTIONS[key] if key else '')
            wait(hold)

        def point(selector, steps=18):
            box = page.locator(selector).first.bounding_box()
            page.mouse.move(box['x'] + min(box['width'] / 2, 60), box['y'] + box['height'] / 2, steps=steps)

        def click(selector, pause=0.25):
            point(selector)
            wait(pause)
            page.mouse.down()
            page.mouse.up()

        # 1. the command, and what it prints
        page.goto((work / 'terminal.html').as_uri())
        page.evaluate(OVERLAY_JS)
        page.mouse.move(W - 30, H - 30)
        rec = Recorder(page, frames, 2)
        wait(0.4)
        page.evaluate('typeCommand("callreplay demo")')
        wait(0.5)
        page.evaluate('play()')
        wait(frames_term[-1][0] + 0.8)
        caption('run', 3.6)

        # 2. the report
        page.goto(report.as_uri())
        page.evaluate(OVERLAY_JS)
        page.mouse.move(W / 2, H / 2)
        wait(0.6)
        page.mouse.wheel(0, 330)
        wait(0.5)
        caption('report', 2.4)

        # 3. a regression, side by side
        click(f'#causes .idchip[data-id="{EXAMPLE_CALL}"]')
        wait(0.5)
        caption('regressed', 1.2)
        point('#dbody .tdiff .tcell.only')
        wait(1.2)
        point('#dbody .turn.flag-error .call')
        wait(1.6)
        click('.dhead .x')
        wait(0.4)

        # 4. a fix
        click('#causes .idchip[data-id="call-006"]')
        wait(0.5)
        caption('fixed', 1.0)
        point('#dbody .turn.flag-error .bubble')
        wait(2.4)
        click('.dhead .x')
        wait(0.3)

        # 5. what isn't judged
        page.evaluate("document.getElementById('notjudged').scrollIntoView({behavior: 'smooth', block: 'center'})")
        wait(0.6)
        point('#notjudged .idchip')
        caption('apart', 2.6)
        caption('end', 1.6)
        rec.stop()
        browser.close()
    MEDIA.mkdir(parents=True, exist_ok=True)
    rec.encode(MEDIA / 'demo.mp4')
    gif(MEDIA / 'demo.mp4', MEDIA / 'demo.gif')
    print(MEDIA / 'demo.mp4', f'({len(rec.frames)} frames)')


# --------------------------------------------------------------------------- stills
def stills(chrome: bool) -> None:
    """The explainer and the report screenshots, light and dark, and the 1280x640 social preview."""
    from playwright.sync_api import sync_playwright
    work = Path(tempfile.mkdtemp(prefix='callreplay-stills-'))
    data = run_demo(work)
    url = (work / 'callreplay-demo' / 'report.html').as_uri()
    MEDIA.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(channel='chrome' if chrome else None)
        for scheme in ('light', 'dark'):
            page = browser.new_page(viewport={'width': 960, 'height': 900}, device_scale_factor=2, color_scheme=scheme)
            page.set_content(explainer_page(data, scheme == 'dark'))
            page.locator('.hero').screenshot(path=str(MEDIA / f'how-it-works-{scheme}.png'))

            page = browser.new_page(viewport={'width': 1200, 'height': 1000}, device_scale_factor=2, color_scheme=scheme)
            page.goto(url)
            page.wait_for_timeout(300)
            top = page.locator('header').bounding_box()['y']
            bottom = page.locator('#notjudged').bounding_box()
            page.screenshot(path=str(MEDIA / f'report-{scheme}.png'),
                            clip={'x': 0, 'y': max(0, top - 24), 'width': 1200, 'height': bottom['y'] + bottom['height'] + 24 - top})
            page.set_viewport_size({'width': 1200, 'height': 860})
            page.evaluate(f"openById('{EXAMPLE_CALL}', 'regressed')")
            page.wait_for_timeout(500)
            page.locator('#drawer').screenshot(path=str(MEDIA / f'side-by-side-{scheme}.png'))

        shot = base64.b64encode((MEDIA / 'how-it-works-dark.png').read_bytes()).decode()
        social = browser.new_page(viewport={'width': 1280, 'height': 640})
        social.set_content(f"""<html><body style="margin:0;width:1280px;height:640px;overflow:hidden;background:#0d1117;
          font-family:-apple-system,Inter,sans-serif;color:#e6edf3;position:relative">
          <div style="position:absolute;left:60px;top:64px;width:470px">
            <div style="font:600 18px ui-monospace,Menlo,monospace;color:#8ea4ff">● callreplay</div>
            <h1 style="font-size:50px;line-height:1.08;letter-spacing:-.03em;margin:22px 0 18px">Regression tests for voice agents</h1>
            <p style="font-size:22px;line-height:1.45;color:#9198a1;margin:0">Replay your recorded calls against a new prompt or model,
              check each one against a contract, and see which calls broke and why.</p>
            <p style="font:15px ui-monospace,Menlo,monospace;color:#6e7681;margin-top:30px">github.com/73bruno/callreplay</p>
          </div>
          <img src="data:image/png;base64,{shot}" style="position:absolute;left:572px;top:150px;width:664px;border-radius:14px;
            border:1px solid #30363d;box-shadow:0 30px 80px rgba(0,0,0,.5)"></body></html>""")
        social.screenshot(path=str(MEDIA / 'social-preview.png'))
        browser.close()
    print('stills in', MEDIA)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--chrome', action='store_true', help='use the installed Google Chrome')
    ap.add_argument('--stills', action='store_true', help='only the images, no video')
    a = ap.parse_args()
    stills(a.chrome)
    if not a.stills:
        record(a.chrome)
