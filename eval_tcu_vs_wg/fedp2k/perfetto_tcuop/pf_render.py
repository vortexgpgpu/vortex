#!/usr/bin/env python3
"""Headless screenshots of the real Perfetto UI (ui.perfetto.dev) via system Chrome.

usage: pf_render.py TRACE.json SHOTS.json OUTDIR
SHOTS.json: [{"name": "01_overview", "start": 0, "end": 81500, "caption": "..."}, ...]
start/end are in trace time units. With the 1 ns = 1 cycle mapping used by
tcuop_engine_tracks.py they are core cycles.
"""
import json, sys, time
from pathlib import Path
from playwright.sync_api import sync_playwright

trace, shots_f, outdir = sys.argv[1], sys.argv[2], Path(sys.argv[3])
W, H = int(sys.argv[4]) if len(sys.argv) > 4 else 1900, int(sys.argv[5]) if len(sys.argv) > 5 else 1500
outdir.mkdir(parents=True, exist_ok=True)
shots = json.load(open(shots_f))

JS_SETUP = """() => {
  const a = window.app, t = a.trace, log = [];
  // timestamps as raw trace ns with separators == core cycles
  try { if (a.timestampFormat && a.timestampFormat.set) { a.timestampFormat.set('traceNsLocale'); log.push('tsfmt:set'); }
        else { t.timeline.timestampFormat = 'traceNsLocale'; log.push('tsfmt:assign'); } } catch (e) { log.push('tsfmt-err:' + e); }
  try { if (a.sidebar.visible) { a.commands.runCommand('dev.perfetto.ToggleLeftSidebar'); } log.push('sidebar:' + a.sidebar.visible); } catch (e) { log.push('sb-err:' + e); }
  try { a.commands.runCommand('dev.perfetto.ExpandAllGroups'); } catch (e) { log.push('exp-err:' + e); }
  // cookie consent toast: remove the smallest element that holds its text
  try { const els = [...document.querySelectorAll('body *')].filter(el => /This site uses cookies/.test(el.textContent || ''));
        els.sort((x, y) => (x.textContent.length - y.textContent.length));
        // climb only while the ancestor still holds nothing but the toast (short text)
        let el = els[0];
        while (el && el.parentElement && el.parentElement !== document.body &&
               (el.parentElement.textContent || '').length < 200) el = el.parentElement;
        if (el && (el.textContent || '').length < 200) { el.style.display = 'none'; log.push('cookie:hidden'); } } catch (e) { log.push('cookie-err:' + e); }
  const top = t.currentWorkspace.children || [];
  log.push('groups:' + top.map(n => n.name + '=' + n.expanded).join(' | '));
  return log;
}"""

with sync_playwright() as p:
    b = p.chromium.launch(executable_path="/usr/bin/google-chrome", headless=True, args=["--no-sandbox"])
    pg = b.new_page(viewport={"width": W, "height": H}, device_scale_factor=1)
    pg.goto("https://ui.perfetto.dev/", wait_until="networkidle", timeout=120000)
    time.sleep(2)
    try:
        pg.get_by_role("button", name="OK").click(timeout=3000)
    except Exception:
        pass
    pg.query_selector_all("input[type=file]")[0].set_input_files(trace)
    pg.wait_for_function("() => window.app && window.app.trace && !window.app.isLoadingTrace", timeout=300000)
    time.sleep(5)
    for _ in range(2):
        try:
            pg.get_by_role("button", name="OK").click(timeout=2000)
        except Exception:
            pass
    print("setup:", pg.evaluate(JS_SETUP))
    time.sleep(2)
    for s in shots:
        pg.evaluate("""([s, e]) => window.app.trace.timeline.panSpanIntoView(BigInt(s), BigInt(e),
                        {align: 'zoom', margin: 0.0, animation: 'step'})""", [int(s["start"]), int(s["end"])])
        pg.evaluate("""(c) => { let d = document.getElementById('__cap');
            if (!d) { d = document.createElement('div'); d.id = '__cap'; document.body.appendChild(d); }
            d.style.cssText = 'position:fixed;top:4px;left:8px;z-index:99999;background:#1b2a3a;color:#fff;' +
              'font:600 15px/1.35 system-ui,sans-serif;padding:7px 12px;border-radius:4px;max-width:1400px';
            d.textContent = c; }""", s.get("caption", s["name"]))
        time.sleep(float(s.get("settle", 2.5)))
        path = outdir / f"{s['name']}.png"
        pg.screenshot(path=str(path), full_page=False)
        print("wrote", path)
    b.close()
