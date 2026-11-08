// ==UserScript==
// @name         Quiet Video Controls
// @namespace    https://github.com/zhaoweigg/browser_scripts
// @version      2.0.0
// @description  Stop HBO Max from dimming the video when the player controls pop up. Controls still appear as usual; the video just stays at full brightness.
// @match        *://*.max.com/*
// @match        *://*.hbomax.com/*
// @run-at       document-start
// @grant        GM_registerMenuCommand
// @grant        GM_getValue
// @grant        GM_setValue
// @updateURL    https://raw.githubusercontent.com/zhaoweigg/browser_scripts/main/quiet-video-controls.user.js
// @downloadURL  https://raw.githubusercontent.com/zhaoweigg/browser_scripts/main/quiet-video-controls.user.js
// ==/UserScript==

(function () {
  'use strict';

  // HBO Max dims the picture with two full-width gradient layers that fade in
  // alongside the controls. Their class names come from styled-components, so
  // match on the stable prefix rather than the hashed suffix.
  const CSS = `
    [class*="TopGradient-"],
    [class*="BottomGradient-"] {
      background: none !important;
      background-image: none !important;
    }
  `;

  const style = document.createElement('style');
  style.id = 'quiet-video-controls';
  style.textContent = CSS;

  let enabled = GM_getValue('noDim', true);
  const apply = () => { style.media = enabled ? 'all' : 'not all'; };
  apply();

  (function inject() {
    const parent = document.head || document.documentElement;
    if (parent) parent.appendChild(style);
    else requestAnimationFrame(inject);
  })();

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
    apply();
    GM_setValue('noDim', enabled);
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
