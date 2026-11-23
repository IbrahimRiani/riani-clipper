"""Iconos del League of Legends al mencionarlos (objetos grandes y hechizos).

Los iconos son los oficiales de Riot (Data Dragon, ddragon.leagueoflegends.com,
pensados para contenido de la comunidad). Se descargan la primera vez a
assets/lol/<versión>/ y no se suben a GitHub.

La detección es local (sin DeepSeek): se buscan los nombres en la
transcripción, en inglés, en español y con la jerga de partida ("rabadon",
"bc", "ie", "flash", "tp"...). Los nombres en español de algunos hechizos son
palabras normales ("curar", "barrera", "limpiar", "prender"), así que de esos
solo vale el nombre inglés.
"""
import json
import os
import re
import unicodedata

import httpx

DD = "https://ddragon.leagueoflegends.com"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Jerga y abreviaturas habituales -> id de Data Dragon (objeto) o de hechizo.
SLANG = {
    # hechizos de invocador
    "flash": "SummonerFlash", "flas": "SummonerFlash", "destello": "SummonerFlash",
    "ignite": "SummonerDot", "ignait": "SummonerDot",
    "smite": "SummonerSmite", "smait": "SummonerSmite", "castigo": "SummonerSmite",
    "tp": "SummonerTeleport", "teleport": "SummonerTeleport", "teletransporte": "SummonerTeleport",
    "heal": "SummonerHeal", "exhaust": "SummonerExhaust", "exhaus": "SummonerExhaust",
    "barrier": "SummonerBarrier", "cleanse": "SummonerBoost", "ghost": "SummonerHaste",
    # objetos
    "rabadon": "3089", "rabadón": "3089", "deathcap": "3089",
    "ie": "3031", "infinity": "3031", "infinity edge": "3031", "filo infinito": "3031",
    "bc": "3071", "black cleaver": "3071", "cuchilla negra": "3071", "cleaver": "3071",
    "zhonya": "3157", "zhonyas": "3157", "zonya": "3157",
    "trinity": "3078", "trinidad": "3078", "trinity force": "3078",
    "kraken": "6672", "sterak": "3053", "steraks": "3053", "warmog": "3083", "warmogs": "3083",
    "thornmail": "3075", "espinas": "3075", "guardian angel": "3026", "ga": "3026",
    "angel guardian": "3026", "shojin": "3161", "collector": "6676", "recolector": "6676",
    "luden": "6655", "ludens": "6655", "liandry": "6653", "liandrys": "6653",
    "rylai": "3116", "rylais": "3116", "void staff": "3135", "baculo del vacio": "3135",
    "morello": "3165", "titanic": "3748", "titanica": "3748", "eclipse": "6692",
    "youmuu": "3142", "youmus": "3142", "mercurial": "3139", "qss": "3139",
    "randuin": "3143", "randuins": "3143", "frozen heart": "3110", "corazon helado": "3110",
    "sunfire": "3068", "spirit visage": "3065", "banshee": "3102", "banshees": "3102",
    "botrk": "3153", "bork": "3153", "rey arruinado": "3153", "lord dominik": "3036",
    "dominik": "3036", "seryldas": "6694", "hullbreaker": "3181", "navori": "6675",
    "nashor": "3115", "diente de nashor": "3115", "riftmaker": "4633", "cosmic drive": "4629",
    "hydra": "3074", "ravenous": "3074", "hidra": "3074", "jak sho": "6665", "jaksho": "6665",
}
# Nombres oficiales que NO se usan (palabras corrientes en español o inglés).
STOP_NAMES = {"curar", "barrera", "limpiar", "prender", "aplastar", "fantasmal",
              "extenuacion", "teleportar", "heal", "eclipse"}


def norm(text):
    text = unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode().lower()
    text = text.replace("'s ", " ").replace("'s", "")
    return re.sub(r"[^a-z0-9 ]+", " ", text).split()


def _get(url):
    r = httpx.get(url, timeout=30, follow_redirects=True)
    r.raise_for_status()
    return r


def load(cache_dir=None):
    """Índice {alias normalizado: {"id", "name", "kind", "png"}} (descarga lo que falte)."""
    cache_dir = cache_dir or os.path.join(ROOT, "assets", "lol")
    os.makedirs(cache_dir, exist_ok=True)
    idx_file = os.path.join(cache_dir, "index.json")
    try:
        version = _get(f"{DD}/api/versions.json").json()[0]
    except Exception:
        if os.path.exists(idx_file):  # sin internet: usar lo que hay
            with open(idx_file, encoding="utf-8") as f:
                return json.load(f)
        raise
    vdir = os.path.join(cache_dir, version)
    if os.path.exists(idx_file):
        with open(idx_file, encoding="utf-8") as f:
            data = json.load(f)
        if data.get("_version") == version:
            return data
    os.makedirs(vdir, exist_ok=True)
    entries = {}

    def add(alias, entry):
        key = " ".join(norm(alias))
        if key and key not in STOP_NAMES and key not in entries:
            entries[key] = entry

    # Hechizos de invocador.
    sp_en = _get(f"{DD}/cdn/{version}/data/en_US/summoner.json").json()["data"]
    sp_es = _get(f"{DD}/cdn/{version}/data/es_ES/summoner.json").json()["data"]
    spells = {}
    for k, d in sp_en.items():
        if "CLASSIC" not in d.get("modes", []):
            continue
        png = os.path.join(vdir, d["image"]["full"])
        spells[k] = {"id": k, "name": d["name"], "kind": "spell", "png": png,
                     "url": f"{DD}/cdn/{version}/img/spell/{d['image']['full']}"}
        add(d["name"], spells[k])
        add(sp_es[k]["name"], spells[k])
    # Objetos grandes: completos (no se mejoran a otro), comprables en la Grieta, >= 2000 de oro.
    it_en = _get(f"{DD}/cdn/{version}/data/en_US/item.json").json()["data"]
    it_es = _get(f"{DD}/cdn/{version}/data/es_ES/item.json").json()["data"]
    items = {}
    for k, d in sorted(it_en.items(), key=lambda kv: int(kv[0])):
        if int(k) >= 10000 or d.get("into") or not d["gold"]["purchasable"] \
                or not d.get("maps", {}).get("11") or d["gold"]["total"] < 2000 \
                or "Consumable" in d.get("tags", []):
            continue
        png = os.path.join(vdir, f"item_{k}.png")
        items[k] = {"id": k, "name": d["name"], "kind": "item", "png": png,
                    "url": f"{DD}/cdn/{version}/img/item/{k}.png"}
        add(d["name"], items[k])
        add(it_es.get(k, {}).get("name", ""), items[k])
    for alias, target in SLANG.items():
        e = spells.get(target) or items.get(target)
        if e:
            key = " ".join(norm(alias))
            entries.setdefault(key, e)
    entries["_version"] = version
    with open(idx_file, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=1)
    return entries


def ensure_png(entry):
    if not os.path.exists(entry["png"]):
        os.makedirs(os.path.dirname(entry["png"]), exist_ok=True)
        with open(entry["png"], "wb") as f:
            f.write(_get(entry["url"]).content)
    return entry["png"]


def mentions(transcript, start, end, index, cfg=None):
    """Menciones dentro del clip: [{"t": s, "png": ruta, "name": ..., "lol": True}]."""
    cfg = cfg or {}
    words = []
    for seg in transcript.get("segments", []):
        for w in seg.get("words", []):
            if start <= float(w["start"]) < end:
                for tok in norm(w["word"]):
                    words.append((tok, float(w["start"]) - start))
    found, i = [], 0
    while i < len(words):
        hit = None
        for n in (4, 3, 2, 1):  # la frase más larga primero ("black cleaver" antes que "black")
            if i + n <= len(words):
                key = " ".join(t for t, _ in words[i:i + n])
                e = index.get(key)
                if isinstance(e, dict):
                    hit = (n, e)
                    break
        if hit:
            n, e = hit
            found.append({"t": round(words[i][1], 2), "id": e["id"], "name": e["name"],
                          "kind": e["kind"], "entry": e})
            i += n
        else:
            i += 1
    # Sin repetir el mismo icono en <10 s, separados >=3 s, como mucho N por clip.
    out, last_t, last_seen = [], -99.0, {}
    for m in found:
        if m["t"] - last_t < cfg.get("min_gap", 3.0) or m["t"] - last_seen.get(m["id"], -99) < 10:
            continue
        try:
            m["png"] = ensure_png(m.pop("entry"))
        except Exception:
            continue
        out.append(m)
        last_t = last_seen[m["id"]] = m["t"]
        if len(out) >= cfg.get("max_per_clip", 4):
            break
    return out


def merge_with_emojis(emojis, icons, window=1.5):
    """Los iconos del LoL sustituyen a los emojis que caigan cerca."""
    keep = [e for e in emojis if all(abs(e["t"] - i["t"]) > window for i in icons)]
    return sorted(keep + [{"t": i["t"], "png": i["png"], "name": i["name"]} for i in icons],
                  key=lambda x: x["t"])
