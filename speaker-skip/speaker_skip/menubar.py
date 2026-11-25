"""macOS menu-bar (status bar) item for the helper, so it never runs invisibly.

Shows an icon while idle and a percentage while a video is being analyzed. The menu
lists what it's doing, whether the browser script has been in touch, and a Quit item.
"""

import subprocess
import time

import AppKit
import Foundation
import objc

from . import pipeline, server

LOG = pipeline.DATA / "helper.log"
STATE_LABEL = {"queued": "Waiting", "downloading": "Downloading", "decoding": "Decoding", "analyzing": "Analyzing"}
# Rough share of total time per stage, to turn stage progress into one number.
STAGE = {"queued": (0.0, 0.0), "downloading": (0.0, 0.1), "decoding": (0.1, 0.02), "analyzing": (0.12, 0.88)}

_controller = None  # keep a reference so the menu's target isn't garbage-collected


def _symbol(name):
    img = AppKit.NSImage.imageWithSystemSymbolName_accessibilityDescription_(name, "Speaker Skip")
    if img is not None:
        img.setTemplate_(True)  # follows light/dark menu bar
    return img


def _ago(ts):
    if not ts:
        return "not seen since the helper started"
    d = int(time.time() - ts)
    if d < 5:
        return "connected just now"
    if d < 90:
        return f"last seen {d}s ago"
    if d < 5400:
        return f"last seen {d // 60} min ago"
    return f"last seen {d // 3600} h ago"


class Controller(AppKit.NSObject):
    def init(self):
        self = objc.super(Controller, self).init()
        if self is None:
            return None
        bar = AppKit.NSStatusBar.systemStatusBar()
        self.item = bar.statusItemWithLength_(AppKit.NSVariableStatusItemLength)
        self.idle_icon = _symbol("forward.end")
        self.busy_icon = _symbol("waveform")
        self.error_icon = _symbol("exclamationmark.triangle")
        self.item.button().setToolTip_("Speaker Skip helper")

        menu = AppKit.NSMenu.alloc().init()
        menu.setAutoenablesItems_(False)

        def add(title, action=None):
            mi = menu.addItemWithTitle_action_keyEquivalent_(title, action, "")
            mi.setTarget_(self)
            mi.setEnabled_(action is not None)
            return mi

        add(f"Speaker Skip helper · 127.0.0.1:{server.PORT}")
        menu.addItem_(AppKit.NSMenuItem.separatorItem())
        self.status_item = add("Idle")
        self.error_item = add("")
        self.error_item.setHidden_(True)
        self.browser_item = add("")
        self.voices_item = add("")
        self.count_item = add("")
        menu.addItem_(AppKit.NSMenuItem.separatorItem())
        add("Open Log", "openLog:")
        add("Show Results in Finder", "openResults:")
        menu.addItem_(AppKit.NSMenuItem.separatorItem())
        add("Quit Speaker Skip", "quit:")
        self.item.setMenu_(menu)

        self.tick_(None)
        timer = Foundation.NSTimer.timerWithTimeInterval_target_selector_userInfo_repeats_(
            1.0, self, "tick:", None, True)
        # Common modes: keeps updating while the menu is open.
        Foundation.NSRunLoop.currentRunLoop().addTimer_forMode_(timer, Foundation.NSRunLoopCommonModes)
        return self

    def tick_(self, _timer):
        snap = server.snapshot()
        button = self.item.button()
        if not getattr(self, "placed", False) and button.window() is not None and _timer is not None:
            f = button.window().frame()
            self.placed = True
            print(time.strftime("%F %T"), f"menu-bar icon placed at x={f.origin.x:.0f} y={f.origin.y:.0f} "
                  f"({f.size.width:.0f}x{f.size.height:.0f}), visible={bool(self.item.isVisible())}", flush=True)
        active = snap["active"]
        if active:
            vid, job = active
            state = job.get("state")
            base, span = STAGE.get(state, (0.0, 0.0))
            pct = round(100 * (base + span * (job.get("progress") or 0.0)))
            button.setImage_(self.busy_icon)
            button.setTitle_(f" {pct}%" if self.busy_icon else f"Skip {pct}%")
            more = f" (+{snap['queued']} waiting)" if snap["queued"] else ""
            self.status_item.setTitle_(f"{STATE_LABEL.get(state, state)} {vid}: {pct}%{more}")
        else:
            err = snap["error"]
            icon = self.error_icon if err else self.idle_icon
            button.setImage_(icon)
            button.setTitle_("" if icon else "Skip")
            self.status_item.setTitle_("Idle, waiting for the browser")

        err = snap["error"]
        self.error_item.setHidden_(err is None)
        if err:
            self.error_item.setTitle_(f"Last error ({err[0]}): {str(err[1].get('message', ''))[:80]}")
        self.browser_item.setTitle_(f"Browser script: {_ago(snap['last_seen'])}")
        self.voices_item.setTitle_("Voices: " + (", ".join(snap["voices"]) or "none enrolled"))
        self.count_item.setTitle_(f"Videos analyzed: {snap['analyzed']}")

    def openLog_(self, _sender):
        subprocess.Popen(["/usr/bin/open", "-a", "Console", str(LOG)])

    def openResults_(self, _sender):
        pipeline.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        subprocess.Popen(["/usr/bin/open", str(pipeline.RESULTS_DIR)])

    def quit_(self, _sender):
        print(time.strftime("%F %T"), "quit from menu bar", flush=True)
        AppKit.NSApp.terminate_(None)  # exit code 0: launchd leaves it stopped until next login


def run():
    global _controller
    app = AppKit.NSApplication.sharedApplication()
    app.setActivationPolicy_(AppKit.NSApplicationActivationPolicyAccessory)  # no Dock icon
    _controller = Controller.alloc().init()
    print(time.strftime("%F %T"), "menu-bar item ready", flush=True)
    app.run()
