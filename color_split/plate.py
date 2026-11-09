# SPDX-License-Identifier: GPL-3.0-or-later
"""Put a file on OrcaSlicer's plate from inside OrcaSlicer (Windows).

The plugin API has no call to add objects. OrcaSlicer itself, though, listens on its main
window for ``WM_COPYDATA`` messages carrying a command line: that is how a second OrcaSlicer
started with a file hands it to the running one (GUI_App.cpp, ``MSWRegisterMessageHandler``),
and the message handler loads the files with ``Plater::load_files``, exactly as a drag and
drop does. The main window is marked with the window property ``Instance_Hash_Minor``
(InstanceCheck.cpp). The handler is registered whatever the single-instance setting is.

The command line must be written the way OrcaSlicer writes it (``escape_strings_cstyle``):
words separated by ';'. Words separated by spaces are rejected ("message from other instance
is incorrectly formatted" in OrcaSlicer's log).

This is not a documented plugin API: if it is not available, callers fall back to telling the
user how to import the file.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

last_note = ""  # what the last add_to_plate did, for the log
WM_COPYDATA = 0x004A
_COPYDATA_LOAD = 1  # dwData OrcaSlicer expects for a command line
_MARKER = "Instance_Hash_Minor"
_SPECIAL = set(" \t;\\\"\r\n")


def escape_strings_cstyle(words) -> str:
    """OrcaSlicer's escape_strings_cstyle (libslic3r/Config.cpp): words joined by ';'; a word
    holding a space, tab, ';', backslash, quote or line break is quoted, with backslash and
    quote escaped by a backslash."""
    out = []
    for word in words:
        if (len(words) == 1 and not word) or _SPECIAL & set(word):
            body = (word.replace("\\", "\\\\").replace("\"", "\\\"")
                    .replace("\r", "\\r").replace("\n", "\\n"))
            out.append("\"" + body + "\"")
        else:
            out.append(word)
    return ";".join(out)


def unescape_strings_cstyle(text: str) -> list[str] | None:
    """OrcaSlicer's unescape_strings_cstyle, to check messages; None where OrcaSlicer would
    reject the text as incorrectly formatted."""
    out: list[str] = []
    i, n = 0, len(text)
    if n == 0:
        return out
    while True:
        while text[i] in " \t":
            i += 1
            if i == n:
                return out
        buf = []
        if text[i] == "\"":
            i += 1
            while i < n:
                c = text[i]
                if c == "\"":
                    break
                if c == "\\":
                    i += 1
                    if i == n:
                        return None
                    c = {"r": "\r", "n": "\n"}.get(text[i], text[i])
                buf.append(c)
                i += 1
            if i == n:
                return None
            i += 1
        else:
            while i < n and text[i] != ";":
                buf.append(text[i])
                i += 1
        out.append("".join(buf))
        if i == n:
            return out
        while text[i] in " \t":
            i += 1
            if i == n:
                return out
        if text[i] != ";":
            return None
        i += 1
        if i == n:
            out.append("")
            return out


def command_line(paths) -> str:
    """The message OrcaSlicer parses: the program, then the files (the first word is skipped)."""
    return escape_strings_cstyle(["orca-slicer"] + [str(Path(p).resolve()) for p in paths])


def _main_window():
    import ctypes
    from ctypes import wintypes
    user32 = ctypes.windll.user32
    user32.GetPropW.restype = wintypes.HANDLE
    user32.GetPropW.argtypes = [wintypes.HWND, wintypes.LPCWSTR]
    user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    found = []
    pid = os.getpid()

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def visit(hwnd, _):
        owner = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
        if owner.value == pid and user32.GetPropW(hwnd, _MARKER):
            found.append(hwnd)
            return False
        return True
    user32.EnumWindows(visit, 0)
    return found[0] if found else None


def add_to_plate(paths) -> bool:
    """Ask the OrcaSlicer we run in to load ``paths`` onto the plate. True if the request was
    delivered (OrcaSlicer may still ask how to import a project file)."""
    global last_note
    if sys.platform != "win32":
        last_note = "not Windows"
        return False
    try:
        import ctypes
        from ctypes import wintypes
        hwnd = _main_window()
        if not hwnd:
            last_note = "main window not found"
            return False

        class COPYDATASTRUCT(ctypes.Structure):
            _fields_ = [("dwData", ctypes.c_size_t), ("cbData", wintypes.DWORD),
                        ("lpData", ctypes.c_void_p)]
        text = ctypes.create_unicode_buffer(command_line(paths))
        data = COPYDATASTRUCT(_COPYDATA_LOAD, ctypes.sizeof(text), ctypes.cast(text, ctypes.c_void_p))
        user32 = ctypes.windll.user32
        user32.SendMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
        user32.SendMessageW.restype = ctypes.c_ssize_t
        result = user32.SendMessageW(hwnd, WM_COPYDATA, 0, ctypes.addressof(data))
        last_note = f"hwnd {hwnd}, result {result}, message {text.value}"
        return True
    except Exception as exc:  # noqa: BLE001 - not allowed by the host, or not OrcaSlicer: fall back
        last_note = f"failed: {exc!r}"
        return False
