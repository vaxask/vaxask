import os
import re
import json
from datetime import datetime
from openai import OpenAI
from ingest import get_chroma_client, get_collection, get_embedder
from models import ArticleMetadata
from classifier import classify
import policy
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv('DEEPSEEK_API_KEY'),
    base_url=os.getenv('DEEPSEEK_BASE_URL', 'https://api.deepseek.com')
)

SYSTEM_PROMPT_TR = """Sen VaxAsk'sın — aşılar konusunda kafası karışmış, endişeli ya da meraklı
kişilere yardım eden, sıcak ama dürüst bir rehbersin. Bilimsel makalelerden öğrendiğin
bilgilerle konuşursun; karşındakinin gerçek bir insan olduğunu ve kendi sağlık kararını
KENDİSİNİN vereceğini asla unutmazsın.

KİM OLDUĞUN:
- Doktor ya da bilim insanı değilsin. Bilimsel makaleleri okumuş, anlamış ve sade bir
  dille aktarabilen güvenilir bir rehbersin.
- Karşındakini ASLA yargılama, küçümseme ya da "cahil" hissettirme. Soru sorması değerlidir.

KİME KONUŞTUĞUN (çok önemli):
- Soruyu SORANIN durumuna konuş. Kişinin KENDİSİ mi, ÇOCUĞU mu, bir başkası mı için
  sorduğunu mesajdan anla — VARSAYMA.
- Soru kişinin kendisi hakkındaysa ona kendi sağlığı için konuş; ebeveynlik/çocuk
  çerçevesi UYDURMA ("harika bir anne/baba", "sizi ve çocuğunuzu" gibi ifadeler kurma).
- Yalnızca soru açıkça bir çocuğa dairse çocuk çerçevesini kullan.
- DAİMA nazik/resmi "siz" diye hitap et; teklifsiz "sen"e ASLA geçme. "Gel, seni, sana,
  bebeğini, yapmalısın, danış, sorman" gibi sen biçimleri YASAK — yerine "Gelin, sizi, size,
  bebeğinizi, yapmalısınız, danışın, sormanız" kullan. "canım, tatlım" gibi takma ad / sevgi
  sözcüğü de KULLANMA; sıcaklık saygıdan ve samimiyetten gelir, lakaptan değil.

KONUŞMA TARZI:
- Sade, 8. sınıf seviyesinde Türkçe. Kısa cümleler, kısa paragraflar.
- Empatiyle başla: "Bu soruyu sormanız çok doğal" gibi. Kişinin kaygısını ciddiye al.
- Teknik terimi hemen parantezle açıkla: "antikor (vücudun savunma proteini)".
- Gerektiğinde günlük hayattan benzetme yap.
- Kesinleşmemiş konularda dürüst ol: "Bu konuda araştırma sürüyor, kesin bir şey
  söylemek zor."

YASAK HAMLELER (bunları ASLA yapma):
- KORKUTMA / İTME YOK: "aşı olmazsan korumasız kalırsın", "şöyle kötü olur" gibi
  korku-temelli ikna ya da kişiyi-ikna-etme refleksi kullanma. Karar kişinindir; sen
  bilgi verirsin, yarar/riski kişi adına TARTMAZSIN, baskı yapmazsın.
- DRAMATİZE ETME: Risk karşılaştırmalarını abartma. "çok daha büyük risk taşır",
  "rulet oynamak gibi", "felakete davetiye" gibi ikna edici/duygu-yüklü çerçeveler kurma.
  Riskleri NÖTR ve kaynağıyla aktar (örn. "X çalışması şu riski bildiriyor [n]"), tartmayı
  ve sonuç çıkarmayı kişiye bırak. Kalın/vurgulu cümlelerle bir tarafı öne çıkarma.
- SELF-TRIAGE YOK: Kullanıcıya "şu belirtiyse sorun değil, bu belirtiyse tehlikeli" gibi
  kendi durumunu sınıflandıracağı eşik/belirti listesi VERME. Bir reaksiyonun şiddetini
  değerlendirmek HEKİMİN işidir, kullanıcının değil.
- TANI / REÇETE YOK: Teşhis koyma, ilaç ya da doz önerme.
- UYDURMA YOK: Sadece sana verilen makale parçalarındaki bilgiyi kullan; bilmiyorsan
  "Bu konuda elimde yeterli bilgi yok" de.

KAYNAK KULLANIMI:
- Her önemli bilgiyi metin içinde [1], [2] ile bir kaynağa bağla.
- Farklı makalelerden yararlan; her paragrafta en az bir kaynak numarası olsun.
- Kaynak numaraları sana verilen MAKALE numaralarına karşılık gelmeli.
- "Araştırmalar gösteriyor" gibi belirsiz ifade kullanma; hangi makale ise [n] ile belirt.
- Yanıt sonunda kaynak listesi YAZMA; sadece metin içinde [n] ver.

KAPANIŞ:
- Sıcak ve kişinin kararına saygılı bir cümleyle bitir, örn. "Başka merak ettiğin olursa
  çekinmeden sorabilirsin." Emoji kullanma.

Yanıtının TAMAMINI TÜRKÇE yaz. (Kullanıcı açıkça başka bir dilde yazıyorsa o dile uyabilirsin.)"""

SYSTEM_PROMPT_EN = """You are VaxAsk — a warm but honest guide who helps confused, anxious or
curious people with questions about vaccines. You speak from what you've learned in
scientific papers; you never forget that the person in front of you is a real human who
will make their OWN health decision.

WHO YOU ARE:
- You are not a doctor or a scientist. You are a reliable guide who has read scientific
  papers, understood them, and can explain them in plain language.
- Never judge, belittle or make the person feel "ignorant." Asking is valuable.

WHO YOU ARE TALKING TO (very important):
- Speak to the situation of the PERSON ASKING. Work out from the message whether they are
  asking about THEMSELVES, their CHILD, or someone else — do NOT assume.
- If the question is about the person themselves, speak to their own health; do NOT invent
  a parent/child frame ("what a wonderful parent", "you and your child").
- Use a child frame only if the question is clearly about a child.
- Avoid pet names or terms of endearment — warmth comes from sincerity, not nicknames.

TONE:
- Plain English, 8th-grade level. Short sentences, short paragraphs.
- Start with empathy, e.g. "It's completely normal to ask this." Take the worry seriously.
- Explain any technical term in parentheses: "antibody (a defense protein the body makes)."
- Use everyday analogies when helpful.
- Be honest about uncertainty: "Research on this is ongoing, so it's hard to be definitive."

FORBIDDEN MOVES (never do these):
- NO FEAR / NO PUSHING: don't use fear-based persuasion ("if you don't vaccinate you'll be
  unprotected") or a reflex to talk the person into it. The decision is theirs; you give
  information, you do NOT weigh the risk/benefit on their behalf or apply pressure.
- NO DRAMATIZING: don't exaggerate risk comparisons ("a much bigger risk", "like playing
  roulette", "a recipe for disaster"). State risks NEUTRALLY with their source (e.g. "study
  X reports this risk [n]") and leave the weighing and conclusion to the person. Don't use
  bold/emphasis to push one side.
- NO SELF-TRIAGE: do not give the user a checklist of thresholds/symptoms to classify their
  own case ("if it's mild it's fine, if it's severe it's dangerous"). Judging the severity
  of a reaction is the CLINICIAN's job, not the user's.
- NO DIAGNOSIS / NO PRESCRIBING: don't diagnose or recommend a drug or dose.
- NO MAKING THINGS UP: use only the article excerpts given; if you don't know, say
  "I don't have enough information on this."

SOURCE USE:
- Back every important claim with an inline [1], [2] source number.
- Draw on different articles; at least one source number per paragraph.
- Source numbers must correspond to the ARTICLE numbers given to you.
- Don't say "studies show" — point to which article with [n].
- Do NOT write a reference list at the end — only cite inline like [n].

CLOSING:
- End warmly and respecting the person's autonomy, e.g. "If there's anything else you're
  wondering about, feel free to ask." Do not use emojis.

Write your ENTIRE answer in ENGLISH. (If the user clearly writes in another language, you may follow theirs.)"""

SAFETY_GATE = {
    "tr": (
        "\n\n⚠️ GÜVENLİK KAPISI — Bu mesajda tıbbi bir KIRMIZI BAYRAK var (tür: {tur}). Kurallar:\n"
        "- ASLA \"güvenli\", \"rahatça yaptırabilirsiniz\", \"endişelenme\" gibi güvence VERME.\n"
        "- ŞİDDET AYRIMI YAPMA: \"hafifse sorun değil / engel değildir\", \"ciddiyse farklıdır\" gibi "
        "reaksiyonu HAFİF–CİDDİ diye ayıran HİÇBİR cümle kurma. Kaynaklarda (ör. Kelso) bu ayrım ya da "
        "\"kademeli doz\" geçse bile kullanıcıya bir KENDİNİ-DEĞERLENDİRME ölçütü olarak SUNMA. "
        "Reaksiyonun türünü ve şiddetini değerlendirmek TAMAMEN uzmanın işidir.\n"
        "- Yanıtı \"Eğer şuysa… / Eğer buysa…\" gibi KOŞULLU DALLARA bölme; tek ve net mesaj: durumu "
        "uzmana anlat, kararı onunla ver.\n"
        "- Net biçimde bir HEKİME ya da ALERJİ-İMMÜNOLOJİ UZMANINA başvurmasını söyle; kararın "
        "uzmanla, gerekirse kontrollü ortamda verilmesi gerektiğini belirt.\n"
        "- Acil belirti (nefes darlığı, dil/dudak şişmesi) varsa 112'yi aramasını söyle.\n"
        "- Kişinin kaygısını ciddiye al; ama yarar/risk dengesini kişi adına tartma — uzmana bırak."
    ),
    "en": (
        "\n\n⚠️ SAFETY GATE — This message contains a medical RED FLAG (type: {tur}). Rules:\n"
        "- NEVER reassure with \"it's safe\", \"you can go ahead\", \"don't worry\".\n"
        "- NO SEVERITY SPLIT: do not write any sentence that sorts the reaction into MILD vs SEVERE "
        "(\"if it's mild it's not a problem / not a barrier\", \"if it's serious it's different\"). Even if "
        "the sources (e.g. Kelso) contain this distinction or \"graded dosing\", do NOT present it to the "
        "user as a SELF-ASSESSMENT criterion. Judging the type and severity of the reaction is ENTIRELY "
        "the specialist's job.\n"
        "- Do NOT split the answer into CONDITIONAL BRANCHES (\"if it's X… / if it's Y…\"); one clear "
        "message: describe it to a specialist and decide with them.\n"
        "- Clearly tell them to see a DOCTOR or an ALLERGY/IMMUNOLOGY SPECIALIST; the decision "
        "should be made with the specialist, in a supervised setting if needed.\n"
        "- If there are emergency symptoms (trouble breathing, swelling of lips/tongue), tell "
        "them to call their local emergency number.\n"
        "- Take the worry seriously, but do not weigh risk/benefit on their behalf — leave that to the specialist."
    ),
}

AUDIENCE = {
    "tr": {
        "kendisi": "\n[Bağlam: Kişi KENDİ durumunu soruyor. Ona kendi sağlığı için konuş; ebeveyn/çocuk çerçevesi kurma.]",
        "cocugu": "\n[Bağlam: Kişi kendi ÇOCUĞU için soruyor.]",
        "baskasi": "\n[Bağlam: Kişi bir başkası için soruyor; varsayım yapma.]",
        "belirsiz": "\n[Bağlam: Kimin için sorulduğu belirsiz; ebeveyn/çocuk VARSAYMA, nötr konuş.]",
    },
    "en": {
        "kendisi": "\n[Context: The person is asking about THEMSELVES. Speak to their own health; do not build a parent/child frame.]",
        "cocugu": "\n[Context: The person is asking about their own CHILD.]",
        "baskasi": "\n[Context: The person is asking on behalf of someone else; do not assume.]",
        "belirsiz": "\n[Context: It's unclear who this is about; do NOT assume parent/child, stay neutral.]",
    },
}

SYSTEM_PROMPTS = {"tr": SYSTEM_PROMPT_TR, "en": SYSTEM_PROMPT_EN}

NO_SOURCES_MSG = {
    "tr": ("Henüz sisteme yüklenmiş bilimsel makale bulunmuyor. "
           "Admin panelinden PDF yükledikten sonra sorularınızı yanıtlayabilirim."),
    "en": ("There are no scientific papers loaded into the system yet. "
           "Once PDFs are uploaded from the admin panel, I can answer your questions."),
}

USER_INSTRUCTION = {
    "tr": ("Asagidaki bilimsel makale parcalarini kullanarak soruyu yanitla.\n"
           "Yanitinda kaynak gosterirken makale numaralarini kullan: [1], [2] gibi.\n\n"
           "{context}\n\nSORU: {message}"),
    "en": ("Answer the question using the scientific article excerpts below.\n"
           "When citing, use the article numbers: [1], [2], etc.\n\n"
           "{context}\n\nQUESTION: {message}"),
}

def _norm_lang(lang: str) -> str:
    l = (lang or "tr").lower()[:2]
    return l if l in SYSTEM_PROMPTS else "tr"

def embed_query(query: str) -> list:
    return get_embedder().encode(f"query: {query}", normalize_embeddings=True).tolist()

def _unpack(results) -> list[dict]:
    out = []
    if not results.get('ids') or not results['ids'][0]:
        return out
    for i in range(len(results['ids'][0])):
        out.append({
            'id': results['ids'][0][i],
            'text': results['documents'][0][i],
            'metadata': results['metadatas'][0][i],
            'distance': (results.get('distances') or [[None]])[0][i],
        })
    return out

def _diversity(chunks: list[dict], max_per_source: int = 2) -> list[dict]:
    cnt, out = {}, []
    for c in chunks:
        sid = c['metadata'].get('source_id')
        cnt[sid] = cnt.get(sid, 0) + 1
        if cnt[sid] <= max_per_source:
            out.append(c)
    return out

def retrieve_free(query_emb, k: int = 15, max_per_source: int = 2) -> list[dict]:
    col = get_collection(get_chroma_client())
    if col.count() == 0:
        return []
    res = col.query(query_embeddings=[query_emb], n_results=min(k, col.count()),
                    include=['documents', 'metadatas', 'distances'])
    return _diversity(_unpack(res), max_per_source)

def retrieve_guided(query_emb, kategori_id: int, k: int = 15, secondary=None,
                    max_per_source: int = 2) -> tuple[list[dict], bool]:
    col = get_collection(get_chroma_client())
    if col.count() == 0:
        return [], False

    cats = [kategori_id] + list(secondary or [])
    where = {"kategori_id": {"$in": cats}} if len(cats) > 1 else {"kategori_id": kategori_id}
    res = col.query(query_embeddings=[query_emb], n_results=k, where=where,
                    include=['documents', 'metadatas', 'distances'])
    chunks = _diversity(_unpack(res), max_per_source)

    have = {c['id'] for c in chunks}
    for rank in (1, 2):
        anc = col.query(query_embeddings=[query_emb], n_results=2,
                        where={"$and": [{"kategori_id": kategori_id}, {"anchor_rank": rank}]},
                        include=['documents', 'metadatas', 'distances'])
        added = False
        for a in _unpack(anc):
            if a['id'] not in have:
                chunks.insert(0, a)
                have.add(a['id'])
                added = True
        if added:
            break

    chunks = _diversity(chunks, max_per_source)[:k]

    if len(chunks) < max(4, k // 3):
        for c in retrieve_free(query_emb, k, max_per_source):
            if c['id'] not in have:
                chunks.append(c); have.add(c['id'])
        chunks = chunks[:k]

    return chunks, True

def retrieve_relevant_chunks(query: str, n_results: int = 15, max_per_source: int = 2) -> list[dict]:
    return retrieve_free(embed_query(query), n_results, max_per_source)

def group_chunks_by_source(chunks: list[dict]) -> tuple[list[dict], list[list[str]]]:
    source_order = []
    source_meta = {}
    source_texts = {}

    for chunk in chunks:
        sid = chunk['metadata']['source_id']
        if sid not in source_meta:
            source_order.append(sid)
            source_meta[sid] = chunk['metadata']
            source_texts[sid] = []
        source_texts[sid].append(chunk['text'])

    sources = [source_meta[sid] for sid in source_order]
    grouped_texts = [source_texts[sid] for sid in source_order]

    return sources, grouped_texts

def build_context(sources: list[dict], grouped_texts: list[list[str]]) -> str:
    context_parts = []
    for i, (meta, texts) in enumerate(zip(sources, grouped_texts)):
        combined_text = "\n\n".join(texts)
        if meta.get('kunye_dogrulandi') and meta.get('citation_string'):
            kunye_satiri = f"Kunye (dogrulanmis): {meta['citation_string']}\n"
        else:
            kunye_satiri = (
                f"Baslik: {meta.get('title', '')}\n"
                f"Yazarlar: {meta.get('authors', '')}\n"
                f"Yil: {meta.get('year', '')}\n"
                f"Dergi: {meta.get('journal', '')}\n"
                f"(kunye otomatik, dogrulanmadi)\n"
            )
        context_parts.append(
            f"[MAKALE {i+1}]\n"
            f"{kunye_satiri}"
            f"Icerik:\n{combined_text}\n"
        )
    return "\n---\n".join(context_parts)

def remap_citations(answer: str, total_sources: int, max_sources: int = 5) -> tuple[str, list[int]]:
    order = []
    for m in re.findall(r'\[(\d+)\]', answer):
        n = int(m)
        if 1 <= n <= total_sources and n not in order:
            order.append(n)

    if not order:
        return answer, list(range(min(total_sources, max_sources)))

    order = order[:max_sources]

    remap = {old_ref: new_idx for new_idx, old_ref in enumerate(order, 1)}

    def replace_ref(match):
        old_num = int(match.group(1))
        if old_num in remap:
            return f"[{remap[old_num]}]"
        return ""

    remapped_answer = re.sub(r'\[(\d+)\]', replace_ref, answer)

    used_indices = [o - 1 for o in order]

    return remapped_answer, used_indices

def _cp(pattern, replacement):
    rx = re.compile(pattern, re.I)
    def f(m):
        s = m.group(0)
        return replacement[:1].upper() + replacement[1:] if s[:1].isupper() else replacement
    return (rx, f)

_TR_REGISTER_FIXES = [
    (re.compile(r'(^|[.!?\n]\s*)Gel,?\s+'), r'\1Gelin, '),
    _cp(r'\bseninle\b', 'sizinle'),
    _cp(r'\bsenin\b', 'sizin'),
    _cp(r'\bsenden\b', 'sizden'),
    _cp(r'\bsende\b', 'sizde'),
    _cp(r'\bseni\b', 'sizi'),
    _cp(r'\bsana\b', 'size'),
    _cp(r'\bsence\b', 'sizce'),
    _cp(r'\bsen\b', 'siz'),
]

def fix_register_tr(text: str) -> str:
    for pat, rep in _TR_REGISTER_FIXES:
        text = pat.sub(rep, text)
    return text

def strip_stray_markdown(text: str) -> str:
    if not text:
        return text
    t = text.rstrip()
    if t.count("**") % 2 == 1:
        i = t.rfind("**")
        t = (t[:i] + t[i + 2:]).rstrip()
    return t

def clean_answer(answer: str) -> str:
    cleaned = re.sub(r'\n*KULLANILAN_KAYNAKLAR:.*', '', answer)
    cleaned = re.sub(r'\n*Kaynaklar?:?\s*\n(\[\d+\].*\n?)+', '', cleaned)
    return strip_stray_markdown(cleaned.strip())

_REFERRAL_RE = re.compile(r'hekim|doktor|uzman|alerji|imm\u00fcnolog|immunolog|112|acil|sa\u011fl\u0131k kurulu|specialist|doctor|emergency', re.I)
_REASSURE_RE = re.compile(r'\brahat\u00e7a\b|endi\u015felenme\b|risksiz|sorun de\u011fil|kesinlikle yapt\u0131r|it.?s safe|don.?t worry|you can (safely )?go ahead|no risk', re.I)
_SELFTRIAGE_RE = re.compile(
    r'hafif[a-z\u00e7\u011f\u0131\u00f6\u015f\u00fc]*.{0,60}(engel de\u011fil|sorun de\u011fil|normaldir|yap\u0131labilir)'
    r'|(mild|minor).{0,60}(not a (barrier|problem|concern)|usually fine|generally fine)',
    re.I | re.S)

SAFETY_APPEND = {
    "tr": ("\n\nBu konuda en do\u011fru ad\u0131m, durumunuzu bir hekime ya da alerji-imm\u00fcnoloji uzman\u0131na "
           "anlat\u0131p karar\u0131 birlikte vermeniz. Nefes darl\u0131\u011f\u0131, dil/dudak \u015fi\u015fmesi gibi acil bir belirti "
           "ya\u015farsanız vakit kaybetmeden 112'yi arayın."),
    "en": ("\n\nThe right step here is to discuss your situation with a doctor or an allergy/immunology "
           "specialist and decide together. If you have an emergency symptom such as trouble breathing or "
           "swelling of the lips/tongue, call your local emergency number right away."),
}

def _strip_selftriage(text: str) -> str:
    segs = re.split(r'(\n+|(?<=[.!?])\s+)', text)
    out = [s for s in segs if not (s and _SELFTRIAGE_RE.search(s))]
    return re.sub(r'[ \t]{2,}', ' ', ''.join(out)).strip()

def _enforce_safety_gate(answer: str, lang: str) -> tuple[str, dict]:
    flags = {
        "reassurance_detected": bool(_REASSURE_RE.search(answer)),
        "self_triage_detected": bool(_SELFTRIAGE_RE.search(answer)),
        "self_triage_stripped": False,
        "appended_referral": False,
    }
    if flags["self_triage_detected"]:
        answer = _strip_selftriage(answer)
        flags["self_triage_stripped"] = True
        flags["self_triage_detected"] = bool(_SELFTRIAGE_RE.search(answer))

    flags["referral_present"] = bool(_REFERRAL_RE.search(answer))
    if not flags["referral_present"]:
        answer = answer.rstrip() + SAFETY_APPEND[lang]
        flags["appended_referral"] = True
        flags["referral_present"] = True
    return answer, flags

def _log_turn(rec: dict):
    try:
        log_dir = os.path.join(os.path.dirname(__file__), 'data', 'runs')
        os.makedirs(log_dir, exist_ok=True)
        rec = {"ts": datetime.now().isoformat(), **rec}
        with open(os.path.join(log_dir, 'turns.jsonl'), 'a', encoding='utf-8') as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass

def chat(message: str, conversation_history: list[dict] = None, lang: str = "tr") -> dict:
    if conversation_history is None:
        conversation_history = []
    lang = _norm_lang(lang)

    cls = classify(message, lang)
    red = bool((cls.get("kirmizi_bayrak") or {}).get("var"))
    kim = cls.get("kim", "belirsiz")
    if kim not in AUDIENCE[lang]:
        kim = "belirsiz"
    kategori = cls.get("birincil_kategori") or 0
    tutum = cls.get("tutum", "ambivalan")
    afekt = cls.get("afekt", "nötr")
    secondary = [k for k in (cls.get("kategoriler") or []) if isinstance(k, int) and k > 0 and k != kategori][:1]

    kategori_filtre = False
    emb = embed_query(message)
    if kategori and kategori > 0:
        chunks, kategori_filtre = retrieve_guided(emb, kategori, k=15, secondary=secondary)
    else:
        chunks = retrieve_free(emb, k=15)
    if not chunks:
        _log_turn({"lang": lang, "message": message[:500], "classification": cls, "no_sources": True})
        return {"answer": NO_SOURCES_MSG[lang], "sources": [], "classification": cls, "safety": {}}

    sources, grouped = group_chunks_by_source(chunks)
    context = build_context(sources, grouped)

    sys_prompt = SYSTEM_PROMPTS[lang]
    sys_prompt += AUDIENCE[lang][kim]
    policy_used = None
    pb = policy.build_policy_block(kategori, tutum, afekt)
    if pb:
        sys_prompt += "\n\n" + pb
        policy_used = policy.policy_label(kategori, tutum, afekt)
    if red:
        tur = (cls.get("kirmizi_bayrak") or {}).get("tur") or "?"
        sys_prompt += SAFETY_GATE[lang].format(tur=tur)

    messages = [{"role": "system", "content": sys_prompt}]
    messages.extend(conversation_history[-6:])
    messages.append({"role": "user",
                     "content": USER_INSTRUCTION[lang].format(context=context, message=message)})

    response = client.chat.completions.create(
        model=os.getenv('RESPONDER_MODEL', os.getenv('DEEPSEEK_MODEL', 'deepseek-chat')),
        messages=messages,
        max_tokens=1500,
        temperature=float(os.getenv('RESPONDER_TEMPERATURE', '0.3')),
        extra_body={"thinking": {"type": "disabled"}},
    )
    cleaned = clean_answer(response.choices[0].message.content)
    if lang == "tr":
        cleaned = fix_register_tr(cleaned)

    safety = {"red_flag": red, "tur": (cls.get("kirmizi_bayrak") or {}).get("tur")}
    if red:
        cleaned, gate = _enforce_safety_gate(cleaned, lang)
        safety.update(gate)

    remapped_answer, used_indices = remap_citations(cleaned, len(sources))
    used_sources = [sources[i] for i in used_indices]

    _log_turn({
        "lang": lang, "message": message[:500],
        "kategori": kategori, "tutum": tutum, "afekt": afekt, "kim": kim,
        "kirmizi_bayrak": cls.get("kirmizi_bayrak"),
        "kategori_filtre_uygulandi": kategori_filtre,
        "retrieved_source_ids": [s.get("source_id") for s in sources],
        "used_source_ids": [m.get("source_id") for m in used_sources],
        "policy_used": policy_used,
        "safety": safety,
    })

    return {
        "answer": remapped_answer,
        "sources": [ArticleMetadata(**m) for m in used_sources],
        "classification": cls,
        "safety": safety,
        "kategori_filtre_uygulandi": kategori_filtre,
        "policy_used": policy_used,
        "retrieved_source_ids": [s.get("source_id") for s in sources],
    }
