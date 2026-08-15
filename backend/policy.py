import os
import re
import pandas as pd

POLICY_XLSX = os.getenv(
    'POLICY_XLSX',
    os.path.join(os.path.dirname(__file__), 'assets', 'ASI_Yanit_Politikasi_Matrisi.xlsx'),
)

TUTUM_MAP = {
    "kaygılı-açık": "kaygılı-açık", "kaygılı": "kaygılı-açık", "korku": "kaygılı-açık",
    "kaygi": "kaygılı-açık", "endişeli": "kaygılı-açık",
    "ambivalan": "ambivalan", "kararsız": "ambivalan", "kararsiz": "ambivalan",
    "dirençli": "dirençli", "direncli": "dirençli", "öfkeli": "dirençli", "reddeden": "dirençli",
    "kabul": "kabul", "kabullenen": "kabul",
}

AFEKT_TON = {
    "korku": "Baskın duygu korku → sakinleştirici, güven verici, telaşsız ton; 'yalnız değilsin'.",
    "kaygı": "Baskın duygu kaygı → yavaş, net, güven verici ton; endişeyi önce tanı.",
    "kaygi": "Baskın duygu kaygı → yavaş, net, güven verici ton; endişeyi önce tanı.",
    "öfke": "Baskın duygu öfke/güvensizlik → savunmaya geçme; meşru zemini tanı; suçlamayı kişiselleştirme.",
    "ofke": "Baskın duygu öfke/güvensizlik → savunmaya geçme; meşru zemini tanı.",
    "suçluluk": "Baskın duygu suçluluk → utandırma yok, normalize et.",
    "sucluluk": "Baskın duygu suçluluk → utandırma yok, normalize et.",
    "merak": "Baskın duygu merak → doğrudan, bilgilendirici, açık ton.",
    "şüphe": "Baskın duygu şüphe → kanıtı şeffaf göster, kaynağı belirt, dayatma yok.",
    "suphe": "Baskın duygu şüphe → kanıtı şeffaf göster, kaynağı belirt, dayatma yok.",
    "nötr": "", "notr": "",
}

_MTX = None
_STN = None
_LOAD_ERR = None

def _load():
    global _MTX, _STN, _LOAD_ERR
    if _MTX is not None or _LOAD_ERR is not None:
        return
    try:
        xl = pd.ExcelFile(POLICY_XLSX)
        _MTX = xl.parse('Yanıt Politikası Matrisi')
        _STN = xl.parse('Tutum Modülasyonu')
    except Exception as e:
        _LOAD_ERR = str(e)

def _cat_id_of(val) -> int:
    m = re.match(r'\s*(\d+)', str(val))
    return int(m.group(1)) if m else -1

def policy_available() -> bool:
    _load()
    return _MTX is not None and _STN is not None

def build_policy_block(kategori_id: int, tutum: str, afekt: str) -> str:
    _load()
    parts = ["## YANIT POLİTİKASI (bu mesaja özel — uy)"]

    ad = AFEKT_TON.get((afekt or "").strip().lower(), "")
    if ad:
        parts.append(f"- Ton: {ad}")

    tkey = TUTUM_MAP.get((tutum or "").strip().lower(), "ambivalan")

    if _STN is not None:
        srow = _STN[_STN["Tutum"].astype(str).str.strip() == tkey]
        if len(srow):
            s = srow.iloc[0]
            parts.append(
                f"- Tutum ({tkey}): {s['Yaklaşım']} | Bilgi dozu: {s['Bilgi dozu']} "
                f"| Amaç: {s['Birincil amaç']} | Tutum-YASAK: {s['Yasak hamleler']}"
            )

    if _MTX is not None and kategori_id and kategori_id > 0:
        krow = _MTX[_MTX["Kategori"].apply(_cat_id_of) == kategori_id]
        if len(krow):
            c = krow.iloc[0]
            parts.append(f"- Kategori yaklaşımı: {c['Ton / Yaklaşım']}")
            parts.append(f"- İzinli hamleler: {c['İzinli hamleler']}")
            parts.append(f"- KATEGORİYE-ÖZEL YASAK: {c['Kategoriye-özel YASAK hamleler']}")
            parts.append(f"- Yönlendirme: {c['Yönlendirme / kırmızı-bayrak kuralı']}")

    return "\n".join(parts) if len(parts) > 1 else ""

def policy_label(kategori_id: int, tutum: str, afekt: str) -> str:
    tkey = TUTUM_MAP.get((tutum or "").strip().lower(), "ambivalan")
    return f"Kategori{kategori_id} × {tkey} × afekt:{afekt}"

if __name__ == "__main__":
    print("POLICY_XLSX:", POLICY_XLSX)
    print("yüklendi mi:", policy_available(), "| hata:", _LOAD_ERR)
    for kat, tut, af in [(3, "kaygılı-açık", "korku"), (7, "dirençli", "öfke"),
                          (11, "kaygılı-açık", "korku"), (0, "kabul", "merak")]:
        print("\n=====", policy_label(kat, tut, af), "=====")
        print(build_policy_block(kat, tut, af))
