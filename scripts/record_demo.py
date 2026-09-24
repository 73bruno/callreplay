#!/usr/bin/env python3
"""Record docs/media/demo.mp4 and demo.gif, plus the screenshots and the social preview.

    pip install playwright && python scripts/record_demo.py --chrome

Runs the real `callreplay demo`, shows its real output in a terminal page, then drives the real
HTML report with a visible cursor. Frames come from Chrome's screencast and are resampled to a
constant 30 fps with ffmpeg (which must be on the PATH).
"""
from __future__ import annotations

import argparse
import base64
import html
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MEDIA = ROOT / 'docs' / 'media'
W, H = 1440, 900

CAPTIONS = {
    'run': 'Replay 42 recorded calls against a new prompt',
    'report': 'Every call checked against a contract: fixed, regressed, not judged',
    'regressed': 'A regression, side by side: it now books without checking the calendar',
    'fixed': 'A fix: it no longer offers a time the calendar never returned',
    'apart': 'Silent calls and provider errors are kept apart from real failures',
    'end': 'callreplay · open source · MIT',
}

OVERLAY_JS = r"""
(() => {
  if (window.__demo) return;
  const css = document.createElement('style');
  css.textContent = `
    #demo-cursor{position:fixed;left:0;top:0;width:22px;height:22px;z-index:99999;pointer-events:none;transform:translate(-3px,-2px)}
    #demo-cursor svg{width:22px;height:22px;filter:drop-shadow(0 2px 3px rgba(0,0,0,.35))}
    .demo-ripple{position:fixed;z-index:99998;width:34px;height:34px;margin:-17px 0 0 -17px;border-radius:50%;
      pointer-events:none;border:3px solid #3b5bdb;animation:demo-rip .5s ease-out forwards}
    @keyframes demo-rip{from{transform:scale(.3);opacity:1}to{transform:scale(1.4);opacity:0}}
    #demo-caption{position:fixed;left:50%;bottom:44px;transform:translate(-50%,12px);z-index:99990;opacity:0;
      background:rgba(17,17,20,.92);color:#fff;font:600 21px/1.35 Inter,system-ui,sans-serif;letter-spacing:-.01em;
      padding:13px 24px;border-radius:14px;box-shadow:0 18px 50px rgba(0,0,0,.35);transition:all .35s;max-width:82vw;
      text-align:center}
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


def terminal_page(lines: list[str]) -> str:
    body = json.dumps([ansi_to_html(x) for x in lines])
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>
body{{margin:0;background:#e9e9e4;height:100vh;display:flex;align-items:center;justify-content:center;
  font-family:-apple-system,BlinkMacSystemFont,Inter,sans-serif}}
.win{{width:1180px;height:760px;background:#15161a;border-radius:14px;box-shadow:0 30px 80px rgba(0,0,0,.28);overflow:hidden;
  display:flex;flex-direction:column}}
.bar{{height:38px;background:#202127;display:flex;align-items:center;gap:8px;padding:0 16px}}
.bar i{{width:12px;height:12px;border-radius:50%;display:block}}
.bar span{{color:#8b8d96;font-size:13px;margin-left:auto;margin-right:auto;transform:translateX(-30px)}}
pre{{margin:0;padding:18px 22px;color:#d8d9de;font:14.5px/1.5 "SF Mono",SFMono-Regular,Menlo,monospace;flex:1;overflow:hidden;
  white-space:pre}}
.b{{font-weight:700;color:#fff}} .dim{{color:#80838d}} .grey{{color:#6f727b}} .red{{color:#ff6b8a}} .green{{color:#5ad17a}}
.yellow{{color:#f5c04a}} .blue{{color:#6aa8ff}} .magenta{{color:#b69cff}} .cyan{{color:#5fd3e0}}
.p{{color:#5ad17a}} .c{{display:inline-block;width:9px;height:18px;background:#d8d9de;vertical-align:-3px;animation:blink 1s steps(1) infinite}}
@keyframes blink{{50%{{opacity:0}}}}
</style></head><body><div class="win"><div class="bar"><i style="background:#ff5f57"></i><i style="background:#febc2e"></i>
<i style="background:#28c840"></i><span>~/my-voice-agent</span></div><pre id="t"></pre></div>
<script>
const LINES = {body};
const t = document.getElementById('t');
window.typeCommand = async (cmd) => {{
  t.innerHTML = '<span class="p">$</span> <span id="cmd"></span><span class="c"></span>';
  for (const ch of cmd) {{ document.getElementById('cmd').textContent += ch; await new Promise(r => setTimeout(r, 45)); }}
}};
window.printAll = async () => {{
  document.querySelector('.c').remove();
  let html = t.innerHTML + '\\n';
  for (const [i, l] of LINES.entries()) {{
    html += l + '\\n'; t.innerHTML = html;
    const lines = html.split('\\n');
    if (lines.length > 36) {{ html = lines.slice(lines.length - 36).join('\\n'); t.innerHTML = html; }}
    await new Promise(r => setTimeout(r, l.includes('/42]') ? 32 : 70));
  }}
}};
</script></body></html>"""


class Recorder:
    """Chrome DevTools screencast: every painted frame with its timestamp."""

    def __init__(self, page, folder: Path):
        self.folder, self.frames = folder, []
        self.cdp = page.context.new_cdp_session(page)
        self.cdp.on('Page.screencastFrame', self.frame)
        self.cdp.send('Page.startScreencast', {'format': 'jpeg', 'quality': 92, 'maxWidth': W, 'maxHeight': H,
                                               'everyNthFrame': 1})

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


def gif(mp4: Path, out: Path, width: int = 960, fps: int = 12) -> None:
    pal = out.with_suffix('.png')
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', str(mp4), '-vf',
                    f'fps={fps},scale={width}:-1:flags=lanczos,palettegen=stats_mode=diff', str(pal)], check=True)
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', str(mp4), '-i', str(pal), '-lavfi',
                    f'fps={fps},scale={width}:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle',
                    str(out)], check=True)
    pal.unlink()


def run_demo(out: Path) -> list[str]:
    """The real command, with colours, from a neutral folder so no local path shows."""
    env = dict(os.environ, FORCE_COLOR='1', PYTHONPATH=str(ROOT))
    res = subprocess.run([sys.executable, '-m', 'callreplay', 'demo', '--no-open', '--out', str(out)],
                         capture_output=True, text=True, env=env, cwd=tempfile.gettempdir(), check=True)
    lines = res.stdout.rstrip('\n').split('\n')
    return [re.sub(r'/\S*/report\.html', 'callreplay-demo/report.html', x) for x in lines]


def record(chrome: bool) -> None:
    from playwright.sync_api import sync_playwright
    work = Path(tempfile.mkdtemp(prefix='callreplay-video-'))
    lines = run_demo(work / 'report')
    (work / 'terminal.html').write_text(terminal_page(lines), encoding='utf-8')
    frames = work / 'frames'
    frames.mkdir()
    with sync_playwright() as p:
        browser = p.chromium.launch(channel='chrome' if chrome else None)
        page = browser.new_page(viewport={'width': W, 'height': H}, color_scheme='light')

        def wait(s): page.wait_for_timeout(int(s * 1000))

        def caption(key=None, hold=0.0):
            page.evaluate('t => window.__demo && window.__demo.caption(t)', CAPTIONS[key] if key else '')
            wait(hold)

        def click(selector, pause=0.25):
            box = page.locator(selector).first.bounding_box()
            page.mouse.move(box['x'] + box['width'] / 2, box['y'] + box['height'] / 2, steps=16)
            wait(pause)
            page.mouse.down()
            page.mouse.up()

        # 1. the command, and its real output
        page.goto((work / 'terminal.html').as_uri())
        page.evaluate(OVERLAY_JS)
        page.mouse.move(W - 40, H - 40)
        rec = Recorder(page, frames)
        wait(0.5)
        page.evaluate('typeCommand("callreplay demo")')
        wait(0.8)
        caption('run')
        page.evaluate('printAll()')
        wait(len(lines) * 0.05 + 1.6)

        # 2. the report
        page.goto((work / 'report' / 'report.html').as_uri())
        page.evaluate(OVERLAY_JS)
        page.mouse.move(700, 300)
        caption('report', 2.6)

        # 3. a regression, side by side
        click('.card[data-f="regressed"]')
        wait(0.9)
        click('#rows tr:nth-child(3)')          # call-005: booked 11:00 without checking
        caption('regressed', 3.2)
        page.mouse.move(1150, 360, steps=14)
        wait(1.6)
        click('.dhead .x')
        wait(0.4)

        # 4. a fix
        click('#filters .chip[data-f="fixed"]')
        wait(0.6)
        click('#rows tr:nth-child(1)')          # call-006: "how about 4:30 pm?" on a full day
        caption('fixed', 3.4)
        click('.dhead .x')
        wait(0.4)

        # 5. what isn't judged
        click('#filters .chip[data-f="error"]')
        wait(0.3)
        caption('apart', 1.2)
        click('#filters .chip[data-f="unscorable"]')
        wait(1.6)
        caption('end', 1.8)
        rec.stop()
        browser.close()
    MEDIA.mkdir(parents=True, exist_ok=True)
    rec.encode(MEDIA / 'demo.mp4')
    gif(MEDIA / 'demo.mp4', MEDIA / 'demo.gif')
    print(MEDIA / 'demo.mp4', f'({len(rec.frames)} frames)')


def stills(chrome: bool) -> None:
    """The side-by-side screenshot for the README and the 1280x640 social preview."""
    from playwright.sync_api import sync_playwright
    work = Path(tempfile.mkdtemp(prefix='callreplay-stills-'))
    run_demo(work / 'report')
    url = (work / 'report' / 'report.html').as_uri()
    with sync_playwright() as p:
        browser = p.chromium.launch(channel='chrome' if chrome else None)
        page = browser.new_page(viewport={'width': W, 'height': H}, color_scheme='light', device_scale_factor=2)
        page.goto(url)
        page.evaluate("setFilter('regressed'); show(2)")
        wait_ms = 600
        page.wait_for_timeout(wait_ms)
        page.screenshot(path=str(MEDIA / 'side-by-side.png'))
        page.goto(url)
        page.wait_for_timeout(300)
        page.screenshot(path=str(MEDIA / 'report.png'))
        shot = base64.b64encode(page.screenshot()).decode()
        social = browser.new_page(viewport={'width': 1280, 'height': 640})
        social.set_content(f"""<html><body style="margin:0;width:1280px;height:640px;overflow:hidden;background:#15161a;
          font-family:-apple-system,Inter,sans-serif;color:#fff;position:relative">
          <div style="position:absolute;left:64px;top:92px;width:520px">
            <div style="font:600 18px ui-monospace,Menlo,monospace;color:#8ea4ff">● callreplay</div>
            <h1 style="font-size:52px;line-height:1.08;letter-spacing:-.03em;margin:22px 0 18px">Regression tests for voice agents</h1>
            <p style="font-size:22px;line-height:1.45;color:#b9bcc6;margin:0">Replay recorded calls against a new prompt or model.
              Check every call against a contract. See what broke.</p>
            <p style="font:15px ui-monospace,Menlo,monospace;color:#80838d;margin-top:34px">pip install git+https://github.com/73bruno/callreplay</p>
          </div>
          <img src="data:image/png;base64,{shot}" style="position:absolute;left:640px;top:70px;width:900px;border-radius:14px;
            box-shadow:0 30px 80px rgba(0,0,0,.5)"></body></html>""")
        social.screenshot(path=str(MEDIA / 'social-preview.png'))
        browser.close()
    print('stills in', MEDIA)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--chrome', action='store_true', help='use the installed Google Chrome')
    ap.add_argument('--stills', action='store_true', help='only the screenshots and the social preview')
    a = ap.parse_args()
    if not a.stills:
        record(a.chrome)
    stills(a.chrome)
