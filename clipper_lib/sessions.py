"""Navegador propio de la app para subir clips, y si hay sesión en cada red.

Es un Brave (o, si no está, Chrome o Edge) normal con un perfil aparte, fuera del proyecto: el usuario
inicia sesión y publica a mano, aquí no se maneja ninguna página. Para la luz
verde/roja solo se mira si existe la cookie de sesión y cuándo caduca (nombre y
fecha); su valor no se lee nunca.

Windows no deja leer ese archivo mientras el navegador está abierto: entonces se
enseña lo último que se supo.
"""
import os
import sqlite3
import subprocess
import time

PROFILE = os.path.join(os.environ.get("LOCALAPPDATA", os.path.expanduser("~")), "ClaudeClipper", "brave")
NETWORKS = {
    "tiktok": {"name": "TikTok", "host": ".tiktok.com", "cookie": "sessionid",
               "login": "https://www.tiktok.com/login", "upload": "https://www.tiktok.com/tiktokstudio/upload"},
    "instagram": {"name": "Instagram", "host": ".instagram.com", "cookie": "sessionid",
                  "login": "https://www.instagram.com/accounts/login/", "upload": "https://www.instagram.com/"},
    "youtube": {"name": "YouTube", "host": ".youtube.com", "cookie": "LOGIN_INFO",
                "login": "https://accounts.google.com/ServiceLogin?service=youtube",
                "upload": "https://www.youtube.com/upload"},
}
_EPOCH = 11644473600  # segundos entre 1601 (Chrome) y 1970
_last = {}  # lo último que se pudo leer


def browser():
    """Ruta de Brave o, si no está, de Chrome o Edge."""
    bases = [os.environ.get(k, "") for k in ("ProgramFiles", "ProgramFiles(x86)", "LOCALAPPDATA")]
    for rel in ("BraveSoftware/Brave-Browser/Application/brave.exe",
                "Google/Chrome/Application/chrome.exe", "Microsoft/Edge/Application/msedge.exe"):
        for b in bases:
            p = os.path.normpath(os.path.join(b, rel))
            if b and os.path.exists(p):
                return p
    return None


def open_url(url):
    exe = browser()
    if not exe:
        raise RuntimeError("no encuentro Brave, Chrome ni Edge")
    os.makedirs(PROFILE, exist_ok=True)
    subprocess.Popen([exe, f"--user-data-dir={PROFILE}", "--no-first-run", "--no-default-browser-check",
                      "--disable-background-mode", url],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _cookies_file():
    for rel in (("Default", "Network", "Cookies"), ("Default", "Cookies")):
        p = os.path.join(PROFILE, *rel)
        if os.path.exists(p):
            return p
    return None


def status():
    """({red: True/False/None}, abierto). None = aún no se sabe."""
    path = _cookies_file()
    if not path:
        return {k: False for k in NETWORKS}, False
    try:
        # mode=ro: ni se copia ni se toca el archivo.
        con = sqlite3.connect("file:" + path.replace("\\", "/").replace("#", "%23") + "?mode=ro", uri=True, timeout=1)
        try:
            rows = con.execute("SELECT host_key, name, expires_utc, has_expires FROM cookies").fetchall()
        finally:
            con.close()
    except sqlite3.Error:
        return {k: _last.get(k) for k in NETWORKS}, True
    now = time.time()
    for k, n in NETWORKS.items():
        _last[k] = any(host == n["host"] and name == n["cookie"]
                       and (not has or exp / 1e6 - _EPOCH > now) for host, name, exp, has in rows)
    return dict(_last), False
