// ==UserScript==
// @name         Quiet Video Controls
// @namespace    https://github.com/zhaoweigg/browser_scripts
// @version      1.0.0
// @description  Stop streaming players (HBO Max, etc.) from popping up their control overlay and dimming the video whenever the mouse moves. Controls only appear after you click (or press a key).
// @match        *://*.max.com/*
// @match        *://*.hbomax.com/*
// @run-at       document-start
// @grant        GM_registerMenuCommand
// @grant        GM_getValue
// @grant        GM_setValue
// ==/UserScript==

(function () {
  'use strict';

  // How long (ms) mouse movement is let through after a click/keypress.
  // Each allowed move extends the window, so you can use the controls freely
  // after clicking; once the mouse sits still this long, blocking resumes.
  const GRACE_MS = 3000;

  // Hover-type events that players use to decide "the user is active, show the UI".
  const BLOCKED_EVENTS = [
    'mousemove', 'pointermove',
    'mouseover', 'pointerover',
    'mouseenter', 'pointerenter',
  ];

  // Events that count as deliberate interaction and open the grace window.
  const UNLOCK_EVENTS = ['mousedown', 'pointerdown', 'touchstart', 'keydown', 'wheel'];

  let enabled = GM_getValue('enabled', true);
  let unlockedAt = 0;

  function isVideoPlaying() {
    for (const v of document.querySelectorAll('video')) {
      if (!v.paused && !v.ended && v.readyState > 2 && v.offsetWidth > 200) {
        return true;
      }
    }
    return false;
  }

  function shouldBlock() {
    if (!enabled) return false;
    const now = Date.now();
    if (now - unlockedAt < GRACE_MS) {
      unlockedAt = now; // keep the window open while the user is actively using the controls
      return false;
    }
    // When paused/not on a player page, behave normally so menus and hover work.
    return isVideoPlaying();
  }

  function blocker(e) {
    if (shouldBlock()) {
      e.stopImmediatePropagation();
    }
  }

  function unlock() {
    unlockedAt = Date.now();
  }

  // Capture phase on window runs before any page listener (React's root listeners included).
  for (const type of BLOCKED_EVENTS) {
    window.addEventListener(type, blocker, { capture: true });
  }
  for (const type of UNLOCK_EVENTS) {
    window.addEventListener(type, unlock, { capture: true, passive: true });
  }

  // ---- Toggle (Tampermonkey menu + Alt+Shift+Q) ----
  function toast(msg) {
    const el = document.createElement('div');
    el.textContent = msg;
    Object.assign(el.style, {
      position: 'fixed', top: '20px', left: '50%', transform: 'translateX(-50%)',
      zIndex: 2147483647, padding: '8px 16px', borderRadius: '6px',
      background: 'rgba(0,0,0,0.8)', color: '#fff', font: '14px sans-serif',
      pointerEvents: 'none',
    });
    (document.fullscreenElement || document.body || document.documentElement).appendChild(el);
    setTimeout(() => el.remove(), 1500);
  }

  function toggle() {
    enabled = !enabled;
    GM_setValue('enabled', enabled);
    toast(`Quiet Video Controls: ${enabled ? 'ON' : 'OFF'}`);
  }

  GM_registerMenuCommand('Toggle Quiet Video Controls', toggle);
  window.addEventListener('keydown', (e) => {
    if (e.altKey && e.shiftKey && e.code === 'KeyQ') {
      e.preventDefault();
      e.stopImmediatePropagation();
      toggle();
    }
  }, { capture: true });
})();
