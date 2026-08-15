import os
import re
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv('DEEPSEEK_API_KEY'),
    base_url=os.getenv('DEEPSEEK_BASE_URL', 'https://api.deepseek.com'),
)

CLASSIFIER_MODEL = os.getenv('CLASSIFIER_MODEL', 'deepseek-chat')
CLASSIFIER_TEMPERATURE = float(os.getenv('CLASSIFIER_TEMPERATURE', '0.3'))
CLASSIFIER_ATTEMPTS = int(os.getenv('CLASSIFIER_ATTEMPTS', '6'))

KATEGORILER = """1=Genel güvenlik/yan etki, 2=Otizm/nörogelişim,
3=Aşı içeriği (cıva/alüminyum/formaldehit/DNA), 4=İmmün yük / "çok fazla aşı",
5=İnfertilite/gebelik, 6=Doğal bağışıklık & complacency,
7=Devlet/firma/sistem güvensizliği, 8=Dini/etik kaygı,
9=Sosyal medya/aile/norm, 10=Aşı etkililiği/koruma şüphesi (koruma oranı/süresi, breakthrough, sürü bağışıklığı),
11=Önceki reaksiyon / özel tıbbi durum, 12=Karar çatışması/kararsızlık"""

CLASSIFIER_PROMPT = f"""Sen bir sınıflandırıcısın. Kullanıcının aşıyla ilgili mesajını analiz et
ve SADECE geçerli bir JSON nesnesi döndür (başka hiçbir metin yazma).

Kategoriler:
{KATEGORILER}

KIRMIZI BAYRAK = tıbbi olarak acil/özel değerlendirme gerektiren, asla "güvenli, yaptır"
denmemesi gereken durumlar:
- anafilaksi: geçmişte aşıya/bileşene ciddi alerjik reaksiyon (nefes darlığı, yüz/dil
  şişmesi, yaygın kurdeşen, tansiyon düşmesi, bayılma)
- bilesen_alerjisi: yumurta, jelatin, lateks, antibiyotik vb. aşı bileşenine bilinen alerji
- immun_yetmezlik: bağışıklık sistemi baskılanmış kişi/çocuk (kemoterapi, immün yetmezlik,
  organ nakli, steroid) veya canlı aşı sorusu
- ciddi_yan_etki: geçmiş aşı sonrası ciddi/şüpheli olay (nöbet, ensefalit, yüksek ateş + döküntü)
ŞÜPHEDEYSEN kırmızı bayrağı VAR olarak işaretle (güvenlik önceliklidir).

"kim" = mesaj kimin hakkında: "kendisi" (kişi AÇIKÇA kendi durumundan bahsediyor:
"ben", "bana", "hamileyim", "alerjim"), "cocugu" (kendi çocuğu/bebeği için),
"baskasi" (başkası ör. komşunun çocuğu), "belirsiz".
ÖNEMLİ: Açık bir kişi-işareti YOKSA "belirsiz" ver — varsayılan olarak "kendisi" SEÇME.
Genel/teorik soru ("aşılar riskli mi?") → belirsiz.

TUTUM = kişinin aşıya YAKLAŞIMI (DUYGU DEĞİL — afekt ayrı alan!). Yalnız bu 4 değerden biri.
VARSAYILAN "ambivalan"dır; kaygılı-açık/dirençli/kabul'ü ANCAK aşağıdaki açık işaret varsa seç.
- "ambivalan" (VARSAYILAN): kararsız; soruyor/tartıyor ama karar vermemiş; nötr bilgi arıyor.
  Tek başına "güvenli mi / zarar verir mi / doğru mu / gerek var mı" SORMAK ambivalandır.
- "kaygılı-açık": YALNIZ açık KORKU/ENDİŞE DUYGUSU dili varsa ("korkuyorum", "endişeliyim",
  "içim rahat değil", "ürküyorum", "çok kaygılıyım"). Sadece "güvenli mi" demek YETMEZ → ambivalan.
- "dirençli": reddeden/iddia eden/şüpheci/komplocu/öfkeli sabit duruş ("gerek yok", "tezgah",
  "doğal daha iyi", "olmayacağım", "bana ne derseniz deyin", "zaten biliyorum").
- "kabul": YALNIZ açıkça aşı-yanlısı, yaptırmaya kararlı, sadece lojistik/bilgi isteyen
  ("yaptıracağım, dozu kaçırdık telafi?"). Tereddüt/şüphe varsa kabul DEĞİL → ambivalan.

KARAR SIRASI: önce dirençli (ret/iddia/komplo) → sonra kaygılı-açık (açık korku duygusu)
→ sonra kabul (net olumlu+lojistik) → hiçbiri yoksa ambivalan.
Örnekler: "Aşı otizm yapar mı?"→ambivalan | "Yaptırmaya korkuyorum"→kaygılı-açık |
"Bu bir tezgah, neden olayım"→dirençli | "Yaptıracağım, dozu kaçırdık telafi?"→kabul.

AFEKT = baskın DUYGU (tutumdan farklı): korku, kaygı, öfke, şüphe, suçluluk, merak, nötr...
ÖNEMLİ: afekt'i tutum alanına YAZMA; tutum DAİMA yukarıdaki 4 değerden biri olmalı.

Çıktı şeması (tam olarak bu alanlar):
{{"birincil_kategori": <0-12 tamsayı>,
 "kategoriler": [<tamsayı listesi>],
 "tutum": "<kaygılı-açık|ambivalan|dirençli|kabul>",
 "kirmizi_bayrak": {{"var": <true|false>, "tur": "<anafilaksi|bilesen_alerjisi|immun_yetmezlik|ciddi_yan_etki|null>"}},
 "afekt": "<korku|kaygı|öfke|şüphe|suçluluk|merak|nötr>",
 "kim": "<kendisi|cocugu|baskasi|belirsiz>"}}"""

TUTUM_ENUM = {"kaygılı-açık", "ambivalan", "dirençli", "kabul"}
_TUTUM_COERCE = {
    "kaygılı": "kaygılı-açık", "kaygi": "kaygılı-açık", "korku": "kaygılı-açık",
    "endişeli": "kaygılı-açık", "anxious": "kaygılı-açık",
    "kararsız": "ambivalan", "kararsiz": "ambivalan", "tereddütlü": "ambivalan",
    "öfkeli": "dirençli", "reddeden": "dirençli", "direnç": "dirençli",
    "kabullenen": "kabul", "meraklı": "ambivalan", "nötr": "ambivalan",
}

def _norm_tutum(t):
    t = (t or "").strip().lower()
    if t in TUTUM_ENUM:
        return t
    return _TUTUM_COERCE.get(t)

REDFLAG_PATTERNS = [
    (r'anafila|anaphyla', 'anafilaksi'),
    (r'nefes (darlığı|alamı)|boğaz(ım|da) (şiş|sıkış)|soluk|wheez|short(ness)? of breath|throat (clos|swell)', 'anafilaksi'),
    (r'(yüz|dil|dudak).{0,12}şiş|swelling of (the )?(face|tongue|lip)|angioedema|ödem', 'anafilaksi'),
    (r'kurdeşen|ürtiker|hives|urticaria', 'anafilaksi'),
    (r'alerji[mn]|alerjik reaksiyon|allergic reaction|(alerji|allerg\w*).{0,18}(oldu|var\b|yaşa|geçir)'
     r'|(önceki|geçmiş|prior|previous).{0,18}(alerji|reaksiyon|reaction)', 'bilesen_alerjisi'),
    (r'(yumurta|jelatin|gelatin|lateks|latex|neomisin|antibiyotik).{0,30}(alerj|allerg|tepki|reaksiyon|reaction)'
     r'|(alerj|allerg).{0,30}(yumurta|jelatin|gelatin|lateks|latex)', 'bilesen_alerjisi'),
    (r'bağışıklığı (baskı|zayıf)|immün ?yetmezlik|immunocompromis|immunodefic|kemoterapi|chemotherap|organ nakli|transplant', 'immun_yetmezlik'),
    (r'nöbet|havale geçir|seizure|konvülsiyon|ensefalit|encephalit', 'ciddi_yan_etki'),
]

_GEBE = r'hamile|gebe\b|gebelik|pregnan'
_CANLI_ASI = (r'\bmmr\b|kızamık|kabakulak|kızamıkçık|suçiçeği|su çiçeği|su çiçegi|'
              r'\bbcg\b|oral polio|\bopv\b|sarı humma|yellow fever|canlı aşı|live (attenuated )?vaccine')

def _keyword_redflag(message: str):
    low = (message or '').lower()
    if re.search(_GEBE, low) and re.search(_CANLI_ASI, low):
        return "gebelikte_canli_asi"
    for pat, tur in REDFLAG_PATTERNS:
        if re.search(pat, low):
            return tur
    return None

def _safe_default(message: str) -> dict:
    tur = _keyword_redflag(message)
    return {
        "birincil_kategori": 12,
        "kategoriler": [12],
        "tutum": "ambivalan",
        "kirmizi_bayrak": {"var": tur is not None, "tur": tur},
        "afekt": "nötr",
        "kim": "belirsiz",
        "_kaynak": "safe_default",
    }

def _validate(obj: dict) -> bool:
    if not isinstance(obj, dict):
        return False
    if not isinstance(obj.get("kirmizi_bayrak"), dict):
        return False
    if "var" not in obj["kirmizi_bayrak"]:
        return False
    return "birincil_kategori" in obj

def classify(message: str, lang: str = "tr") -> dict:
    result = None
    for attempt in range(CLASSIFIER_ATTEMPTS):
        try:
            temp = min(CLASSIFIER_TEMPERATURE + 0.15 * attempt, 1.0)
            resp = client.chat.completions.create(
                model=CLASSIFIER_MODEL,
                messages=[
                    {"role": "system", "content": CLASSIFIER_PROMPT},
                    {"role": "user", "content": message},
                ],
                response_format={"type": "json_object"},
                max_tokens=300,
                temperature=temp,
            )
            content = (resp.choices[0].message.content or "").strip()
            if not content:
                continue
            obj = json.loads(content)
            if _validate(obj):
                nt = _norm_tutum(obj.get("tutum"))
                obj["tutum"] = nt if nt is not None else "ambivalan"
                obj["_kaynak"] = f"llm(attempt{attempt+1})" + ("" if nt else "/tutum-default")
                result = obj
                break
        except Exception:
            result = None

    if result is None:
        result = _safe_default(message)

    kw = _keyword_redflag(message)
    rb = result.get("kirmizi_bayrak") or {"var": False, "tur": None}
    if kw and not rb.get("var"):
        rb = {"var": True, "tur": kw}
        result["kirmizi_bayrak"] = rb
        result["_kw_override"] = True

    if rb.get("var") and rb.get("tur") in ("anafilaksi", "bilesen_alerjisi",
                                           "immun_yetmezlik", "ciddi_yan_etki"):
        result["birincil_kategori"] = 11
        cats = result.get("kategoriler") or []
        result["kategoriler"] = [11] + [c for c in cats if c != 11]
        result["_kat11_align"] = True

    return result
