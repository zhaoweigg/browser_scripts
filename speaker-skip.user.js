// ==UserScript==
// @name         Speaker Skip
// @namespace    https://github.com/zhaoweigg/browser_scripts
// @version      1.1.0
// @description  Adds a button to YouTube's player that skips every part where one chosen person talks. Needs the speaker-skip helper running on this Mac (see speaker-skip/README.md).
// @match        https://www.youtube.com/*
// @run-at       document-idle
// @grant        GM_xmlhttpRequest
// @grant        GM_getValue
// @grant        GM_setValue
// @connect      127.0.0.1
// @updateURL    https://raw.githubusercontent.com/zhaoweigg/browser_scripts/main/speaker-skip.user.js
// @downloadURL  https://raw.githubusercontent.com/zhaoweigg/browser_scripts/main/speaker-skip.user.js
// ==/UserScript==

(function () {
  'use strict';

  const HELPER = 'http://127.0.0.1:8787';

  // Channels (by @handle) the button shows up on, and whose voice it skips there.
  // The name must match a voiceprint enrolled in the helper.
  const CHANNELS = {
    '@三個水槍手': { skip: '伍雷' },
  };

  const POLL_MS = 2000;
  const MAX_FAILS = 8;
  // Skip a little before the end of a range so the next speaker's first word isn't clipped.
  const END_MARGIN = 0.3;

  let cur = null; // { vid, cfg, enabled, ranges, state, progress, message, timer }

  // ---------- helper API ----------
  function api(path) {
    return new Promise((resolve, reject) => {
      GM_xmlhttpRequest({
        method: 'GET',
        url: HELPER + path,
        headers: { 'X-Speaker-Skip': '1' },
        timeout: 8000,
        responseType: 'json',
        onload: (r) => (r.status === 200 ? resolve(r.response) : reject(new Error((r.response && r.response.error) || `HTTP ${r.status}`))),
        onerror: () => reject(new Error('helper not running')),
        ontimeout: () => reject(new Error('helper not responding')),
      });
    });
  }

  // ---------- page access ----------
  // Only the DOM is used: Tampermonkey may run this in a sandbox where YouTube's own
  // player functions (getPlayerResponse etc.) aren't visible.
  const playerEl = () => document.getElementById('movie_player');
  const videoEl = () => document.querySelector('#movie_player video');

  // What the script is doing, readable from the page for troubleshooting:
  // document.documentElement.dataset.speakerSkip
  function debug(msg) {
    document.documentElement.dataset.speakerSkip = `${GM_info.script.version} ${msg}`;
  }

  function videoInfo() {
    const urlVid = new URLSearchParams(location.search).get('v');
    const flexy = document.querySelector('ytd-watch-flexy');
    // After in-app navigation the metadata lags the URL; wait until it's for this video.
    if (!urlVid || !flexy || flexy.getAttribute('video-id') !== urlVid) return debug('waiting: watch page'), null;
    const owner = document.querySelector('ytd-watch-metadata #owner a[href^="/@"]');
    if (!owner) return debug('waiting: channel link'), null;
    return { vid: urlVid, channel: decodeURIComponent(owner.getAttribute('href').slice(1).split('/')[0]) };
  }

  // ---------- UI ----------
  const STYLE = `
    .ss-btn { width:auto !important; padding:0 10px !important; margin:0 4px; font:500 13px/1 Roboto,Arial,sans-serif;
      color:#fff; border-radius:14px; height:28px !important; align-self:center; cursor:pointer;
      background:rgba(255,255,255,.15); border:none; white-space:nowrap; opacity:1 !important; }
    .ss-btn:hover { background:rgba(255,255,255,.28); }
    .ss-btn.ss-on { background:#2e7d32; }
    .ss-btn.ss-busy { background:#b26a00; }
    .ss-btn.ss-err { background:#b3261e; }
    .ss-marks { position:absolute; inset:0; pointer-events:none; z-index:33; }
    .ss-marks > div { position:absolute; top:0; bottom:0; background:rgba(255,193,7,.75); }
    .ss-toast { position:absolute; left:50%; top:12%; transform:translateX(-50%); z-index:80; pointer-events:none;
      background:rgba(0,0,0,.7); color:#fff; font:500 15px Roboto,Arial,sans-serif; padding:6px 14px;
      border-radius:16px; transition:opacity .3s; }
  `;

  function ensureStyle() {
    if (document.getElementById('ss-style')) return;
    const s = document.createElement('style');
    s.id = 'ss-style';
    s.textContent = STYLE;
    document.head.appendChild(s);
  }

  function button() {
    let b = document.querySelector('#movie_player .ss-btn');
    const bar = document.querySelector('#movie_player .ytp-right-controls');
    if (!b && bar) {
      b = document.createElement('button');
      b.className = 'ytp-button ss-btn';
      b.addEventListener('click', (e) => {
        e.stopPropagation();
        toggle();
      });
      bar.prepend(b);
    }
    return b;
  }

  function removeUI() {
    document.querySelectorAll('.ss-btn, .ss-marks').forEach((n) => n.remove());
  }

  function render() {
    if (!cur) return removeUI();
    const b = button();
    if (!b) return;
    const who = cur.cfg.skip;
    b.classList.remove('ss-on', 'ss-busy', 'ss-err');
    b.title = '';
    if (!cur.enabled) {
      b.textContent = `Skip ${who}: off`;
      b.title = `Click to skip every part where ${who} talks`;
    } else if (cur.state === 'error') {
      b.classList.add('ss-err');
      b.textContent = `Skip ${who}: error`;
      b.title = `${cur.message}\nClick to retry.` +
        (cur.message === 'helper not running' ? '\nStart it: see speaker-skip/README.md' : '');
    } else if (cur.ranges && cur.partial) {
      // Still analyzing, but the part done so far is already being skipped.
      b.classList.add('ss-on');
      b.textContent = `Skipping ${who} · ${Math.round((cur.progress || 0) * 100)}%`;
      b.title = `Analyzed up to ${fmtTime(cur.upto)}; skipping ${who} there already. Click to turn off.`;
    } else if (cur.ranges) {
      b.classList.add('ss-on');
      const mins = cur.ranges.reduce((t, [s, e]) => t + (e - s), 0) / 60;
      b.textContent = `Skipping ${who} (${Math.round(mins)}m)`;
      b.title = `${cur.ranges.length} parts, ${mins.toFixed(1)} min skipped. Click to turn off.`;
    } else {
      b.classList.add('ss-busy');
      const label = { queued: 'Waiting', downloading: 'Starting', decoding: 'Decoding', analyzing: 'Analyzing' }[cur.state] || 'Starting';
      b.textContent = cur.state === 'analyzing' ? `${label} ${Math.round((cur.progress || 0) * 100)}%` : `${label}…`;
      b.title = `Finding where ${who} talks. Click to cancel.`;
    }
    renderMarks();
  }

  function fmtTime(sec) {
    const t = Math.floor(sec || 0);
    return `${Math.floor(t / 60)}:${String(t % 60).padStart(2, '0')}`;
  }

  function renderMarks() {
    document.querySelectorAll('.ss-marks').forEach((n) => n.remove());
    const v = videoEl();
    const bar = document.querySelector('#movie_player .ytp-progress-bar');
    if (!cur || !cur.enabled || !cur.ranges || !v || !bar || !v.duration) return;
    const wrap = document.createElement('div');
    wrap.className = 'ss-marks';
    for (const [s, e] of cur.ranges) {
      const d = document.createElement('div');
      d.style.left = `${(s / v.duration) * 100}%`;
      d.style.width = `${((e - s) / v.duration) * 100}%`;
      wrap.appendChild(d);
    }
    bar.appendChild(wrap);
  }

  function toast(msg) {
    const host = document.getElementById('movie_player');
    if (!host) return;
    let t = host.querySelector('.ss-toast');
    if (!t) {
      t = document.createElement('div');
      t.className = 'ss-toast';
      host.appendChild(t);
    }
    t.textContent = msg;
    t.style.opacity = '1';
    clearTimeout(t._h);
    t._h = setTimeout(() => (t.style.opacity = '0'), 1500);
  }

  // ---------- logic ----------
  function stopPolling() {
    if (cur && cur.timer) clearTimeout(cur.timer);
    if (cur) cur.timer = null;
  }

  async function poll(start) {
    const me = cur;
    if (!me || !me.enabled) return;
    try {
      const r = await api(`/analyze?v=${me.vid}${start ? '&start=1' : ''}`);
      if (cur !== me || !me.enabled) return;
      me.state = r.state;
      me.progress = r.progress;
      me.message = r.message;
      if (r.state === 'done') {
        me.partial = false;
        me.ranges = (r.result.skip && r.result.skip[me.cfg.skip]) || [];
        if (!r.result.skip || !(me.cfg.skip in r.result.skip)) {
          me.state = 'error';
          me.message = `No voiceprint named ${me.cfg.skip} in the helper`;
        }
      } else if (r.state !== 'error') {
        // The helper analyzes while it downloads and shares what it has so far.
        if (r.partial && r.partial.skip && me.cfg.skip in r.partial.skip) {
          me.partial = true;
          me.upto = r.partial.upto;
          me.ranges = r.partial.skip[me.cfg.skip];
        }
        me.timer = setTimeout(() => poll(false), POLL_MS);
      }
      me.fails = 0;
    } catch (err) {
      if (cur !== me || !me.enabled) return;
      // A busy helper can miss a request or two; only give up after several in a row.
      me.fails = (me.fails || 0) + 1;
      if (me.fails < MAX_FAILS) {
        me.timer = setTimeout(() => poll(start), POLL_MS);
      } else {
        me.state = 'error';
        me.message = err.message;
      }
    }
    render();
  }

  function toggle() {
    if (!cur) return;
    // Clicking an errored button retries instead of switching off.
    cur.enabled = cur.enabled && cur.state === 'error' ? true : !cur.enabled;
    GM_setValue('enabled', cur.enabled);
    stopPolling();
    if (cur.enabled) {
      cur.state = null;
      cur.ranges = null;
      cur.partial = false;
      cur.fails = 0;
      poll(true);
    }
    render();
  }

  function checkSkip() {
    if (!cur || !cur.enabled || !cur.ranges || !cur.ranges.length) return;
    const p = playerEl();
    const v = videoEl();
    if (!p || !v || p.classList.contains('ad-showing')) return;
    const t = v.currentTime;
    for (const [s, e] of cur.ranges) {
      if (t >= s && t < e - END_MARGIN) {
        v.currentTime = e;
        toast(`Skipped ${cur.cfg.skip} · ${Math.round(e - s)}s`);
        return;
      }
      if (s > t) break;
    }
  }

  function onPage() {
    try {
      onPageInner();
    } catch (err) {
      debug(`error: ${err && err.stack}`);
    }
  }

  function onPageInner() {
    const info = videoInfo();
    const cfg = info && CHANNELS[info.channel];
    if (info) debug(cfg ? `active on ${info.channel}` : `not configured: ${info.channel}`);
    if (!info || !cfg) {
      if (!info && location.pathname === '/watch') return void setTimeout(onPage, 500); // player not ready yet
      stopPolling();
      cur = null;
      return render();
    }
    if (cur && cur.vid === info.vid) return render();
    stopPolling();
    cur = { vid: info.vid, cfg, enabled: GM_getValue('enabled', false), ranges: null, state: null };
    ensureStyle();
    if (cur.enabled) poll(true);
    else api('/health').catch(() => {}); // lets the helper's menu show the browser is connected
    render();
  }

  debug('loaded');
  document.addEventListener('yt-navigate-finish', () => setTimeout(onPage, 300));
  setInterval(checkSkip, 250);
  // The player's control bar gets rebuilt sometimes; re-attach the button and marks.
  setInterval(() => {
    if (cur && !document.querySelector('#movie_player .ss-btn')) render();
    if (cur && cur.ranges && !document.querySelector('.ss-marks')) renderMarks();
  }, 1000);
  onPage();
})();
