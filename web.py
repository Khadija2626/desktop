# ai_study_assistant.py
import streamlit as st
import re
import math
from collections import Counter, defaultdict
import heapq

# Optional OpenAI support (only used if key provided)
try:
    import openai
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

st.set_page_config(page_title="AI Study Assistant", layout="centered")

# --------------------------
# Utilities / Simple NLP
# --------------------------
STOPWORDS = {
    "a","an","the","and","or","but","if","while","with","to","of","in","on","for",
    "is","are","was","were","be","been","has","have","had","do","does","did","that",
    "this","these","those","by","as","at","from","it","its","he","she","they","we",
    "you","I","me","my","mine","your","yours","their","theirs","our","ours"
}

sentence_split_re = re.compile(r'(?<=[.!?])\s+')

def split_sentences(text):
    # naive sentence splitter
    sents = [s.strip() for s in sentence_split_re.split(text.strip()) if s.strip()]
    return sents

def tokenize_words(text):
    # lower, keep only letters and apostrophes
    tokens = re.findall(r"[A-Za-z']+", text.lower())
    return tokens

def sentence_score_by_freq(text):
    sents = split_sentences(text)
    words = tokenize_words(text)
    freqs = Counter(w for w in words if w not in STOPWORDS)
    if not freqs:
        return {i:0 for i in range(len(sents))}
    # normalize freq
    maxf = max(freqs.values())
    for w in list(freqs.keys()):
        freqs[w] /= maxf
    scores = {}
    for i, s in enumerate(sents):
        ws = tokenize_words(s)
        score = sum(freqs.get(w, 0) for w in ws)
        scores[i] = score
    return scores

def extractive_summary(text, max_sentences=3):
    sents = split_sentences(text)
    if not sents:
        return ""
    scores = sentence_score_by_freq(text)
    # choose top sentences by score but preserve original order
    top_n = heapq.nlargest(min(max_sentences, len(sents)), scores, key=lambda i: scores[i])
    top_n_sorted = sorted(top_n)
    summary = " ".join(sents[i] for i in top_n_sorted)
    return summary

def top_keywords(text, n=10):
    words = tokenize_words(text)
    freqs = Counter(w for w in words if w not in STOPWORDS and len(w)>1)
    return [w for w,_ in freqs.most_common(n)]

def approximate_syllables(word):
    # rough heuristic: count vowel groups
    word = word.lower()
    groups = re.findall(r'[aeiouy]+', word)
    count = len(groups)
    if word.endswith("e"):
        count = max(1, count - 1)
    return max(1, count)

def flesch_reading_ease(text):
    sents = split_sentences(text)
    words = tokenize_words(text)
    if len(words) == 0 or len(sents) == 0:
        return None
    syllables = sum(approximate_syllables(w) for w in words)
    asl = len(words) / len(sents)  # average sentence length
    asw = syllables / len(words)   # average syllables per word
    # Flesch Reading Ease formula
    score = 206.835 - (1.015 * asl) - (84.6 * asw)
    return score

# Passive voice heuristic
def detect_passive_sentences(text):
    sents = split_sentences(text)
    passive = []
    # naive: look for forms of 'be' + past participle (ending with -ed) or 'been' + verb
    be_forms = r'\b(am|is|are|was|were|be|been|being)\b'
    for s in sents:
        if re.search(be_forms + r'\s+\w+ed\b', s, re.I) or re.search(r'\bwas\b.*\bby\b', s, re.I):
            passive.append(s)
    return passive

# --------------------------
# Heuristic generators
# --------------------------
def generate_flashcards_from_text(text, n=8):
    # produce cloze flashcards and short Q/A where possible
    sents = split_sentences(text)
    kw = top_keywords(text, n=n*3)
    flashcards = []
    used = set()
    # create cloze cards from sentences containing keywords
    for k in kw:
        if len(flashcards) >= n:
            break
        for s in sents:
            if k in tokenize_words(s) and s not in used:
                # cloze: replace keyword with blank in the sentence
                cloze = re.sub(r'\b'+re.escape(k)+r'\b', '_____', s, flags=re.I)
                question = f"Fill the blank: {cloze}"
                answer = k
                flashcards.append({"q": question, "a": answer})
                used.add(s)
                break
    # If not enough, create Q/A by turning definitions or clauses into question form
    i = 0
    while len(flashcards) < n and i < len(sents):
        s = sents[i]
        i += 1
        # short sentences make decent cards
        if len(tokenize_words(s)) < 6:
            continue
        # find a subject (first noun-ish word) and ask "What/Explain"
        q = f"Explain: {s}"
        a = s
        flashcards.append({"q": q, "a": a})
    return flashcards

def generate_questions_from_text(text, n=6):
    sents = split_sentences(text)
    questions = []
    for s in sents:
        if len(questions) >= n:
            break
        words = tokenize_words(s)
        if not words:
            continue
        # pattern: "X is Y" -> "What is X?"
        m = re.match(r'^(?P<subj>[\w\s\'\-]+?)\s+(is|are|was|were)\s+(?P<pred>.+)', s, re.I)
        if m:
            subj = m.group('subj').strip()
            q = f"What is {subj}?"
            questions.append({"q": q, "ref": s})
            continue
        # pattern: "X causes Y" -> "What causes Y?"
        m2 = re.match(r'^(?P<subj>.+?)\s+(causes|leads to|results in)\s+(?P<obj>.+)', s, re.I)
        if m2:
            obj = m2.group('obj').strip()
            q = f"What causes {obj}?"
            questions.append({"q": q, "ref": s})
            continue
        # fallback: create a "Summarize/Explain" prompt
        questions.append({"q": f"Explain: {s}", "ref": s})
    # trim to n
    return questions[:n]

# --------------------------
# OpenAI helpers (optional)
# --------------------------
def call_openai(prompt, api_key, model="gpt-4o-mini", max_tokens=256, temperature=0.2):
    if not OPENAI_AVAILABLE:
        return None, "openai package not installed"
    try:
        openai.api_key = api_key
        resp = openai.ChatCompletion.create(
            model=model,
            messages=[{"role":"user","content":prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        txt = resp["choices"][0]["message"]["content"].strip()
        return txt, None
    except Exception as e:
        return None, str(e)

# --------------------------
# Streamlit UI
# --------------------------
st.title("📚 AI Study Assistant")
st.markdown(
    "Features: **Summarize text**, **Generate Flashcards**, **Essay Checker**, **Question Generator**.\n\n"
    "You can use the built-in local heuristics (no API key needed) or paste your OpenAI API key in the sidebar to get higher-quality results."
)

with st.sidebar:
    st.header("Options / Settings")
    api_key = st.text_input("OpenAI API Key (optional)", type="password")
    use_openai = st.checkbox("Use OpenAI (if key provided)", value=False)
    if use_openai and api_key.strip() == "":
        st.warning("OpenAI usage enabled but no key provided.")
    st.markdown("---")
    max_summary_sents = st.number_input("Summary sentences", min_value=1, max_value=10, value=3)
    flashcard_count = st.number_input("Flashcards to generate", min_value=1, max_value=30, value=8)
    question_count = st.number_input("Questions to generate", min_value=1, max_value=30, value=6)
    st.markdown("---")
    st.write("Tip: For best OpenAI results, set Use OpenAI and paste your API key above.")

text = st.text_area("Paste your text, passage, or essay here:", height=300)

col1, col2, col3, col4 = st.columns(4)
with col1:
    if st.button("🔎 Summarize"):
        if not text.strip():
            st.warning("Please paste some text first.")
        else:
            if use_openai and api_key and OPENAI_AVAILABLE:
                prompt = f"Summarize the following text into {max_summary_sents} concise sentences:\n\n{text}"
                out, err = call_openai(prompt, api_key, model="gpt-4o-mini", max_tokens=300)
                if err:
                    st.error("OpenAI call failed: " + err)
                else:
                    st.subheader("Summary (OpenAI)")
                    st.write(out)
            else:
                s = extractive_summary(text, max_sentences=max_summary_sents)
                st.subheader("Summary (extractive)")
                st.write(s)

with col2:
    if st.button("🃏 Flashcards"):
        if not text.strip():
            st.warning("Please paste some text first.")
        else:
            if use_openai and api_key and OPENAI_AVAILABLE:
                prompt = (
                    "Generate a list of flashcards (question and short answer) from the text below. "
                    f"Return them as numbered Q / A pairs, up to {flashcard_count} cards.\n\n{text}"
                )
                out, err = call_openai(prompt, api_key, model="gpt-4o-mini", max_tokens=400)
                if err:
                    st.error("OpenAI call failed: " + err)
                else:
                    st.subheader("Flashcards (OpenAI)")
                    st.markdown(out)
            else:
                cards = generate_flashcards_from_text(text, n=flashcard_count)
                st.subheader("Flashcards (heuristic)")
                for i, c in enumerate(cards, 1):
                    st.markdown(f"**Q{i}. {c['q']}**")
                    st.write(f"Answer: {c['a']}")
                    st.write("---")

with col3:
    if st.button("📝 Essay Check"):
        if not text.strip():
            st.warning("Please paste some text first.")
        else:
            if use_openai and api_key and OPENAI_AVAILABLE:
                prompt = "Act as an essay editor. Provide: 1) a short critique with strengths and weaknesses, 2) a list of suggested edits, and 3) a score out of 100 for clarity and grammar. Text:\n\n" + text
                out, err = call_openai(prompt, api_key, model="gpt-4o-mini", max_tokens=500)
                if err:
                    st.error("OpenAI call failed: " + err)
                else:
                    st.subheader("Essay Check (OpenAI)")
                    st.write(out)
            else:
                # local checks
                sents = split_sentences(text)
                words = tokenize_words(text)
                total_words = len(words)
                total_sents = len(sents)
                fre = flesch_reading_ease(text)
                passive = detect_passive_sentences(text)
                # repeated words
                cw = Counter(words)
                repeats = [w for w,c in cw.items() if c>5 and w not in STOPWORDS]
                avg_sent_len = (total_words / total_sents) if total_sents>0 else 0
                st.subheader("Essay Check (heuristic)")
                st.write(f"Sentences: {total_sents}  |  Words: {total_words}  |  Avg sentence length: {avg_sent_len:.1f} words")
                if fre is not None:
                    st.write(f"Approx. Flesch Reading Ease: {fre:.1f} (higher = easier)")
                if passive:
                    st.warning(f"Passive voice detected in {len(passive)} sentence(s). Consider rewriting. Examples:")
                    for p in passive[:3]:
                        st.write("-", p)
                else:
                    st.success("No obvious passive voice patterns detected.")
                if repeats:
                    st.warning("Repeated words detected (might reduce clarity): " + ", ".join(repeats[:10]))
                else:
                    st.info("No strongly repeated words detected.")
                # suggestions
                st.markdown("**Suggestions:**")
                sug = []
                if total_words < 200:
                    sug.append("The essay is short — consider expanding your arguments with examples.")
                if avg_sent_len > 25:
                    sug.append("Some sentences are long. Try splitting long sentences to improve clarity.")
                if fre is not None and fre < 50:
                    sug.append("Text is relatively hard to read. Use simpler words and shorter sentences.")
                if passive:
                    sug.append("Rewrite passive sentences in active voice when possible.")
                if not sug:
                    st.write("Looks pretty good! Small stylistic tweaks may still help.")
                else:
                    for s0 in sug:
                        st.write("-", s0)

with col4:
    if st.button("❓ Generate Questions"):
        if not text.strip():
            st.warning("Please paste some text first.")
        else:
            if use_openai and api_key and OPENAI_AVAILABLE:
                prompt = f"From the text below generate {question_count} clear study questions (mix of factual, short-answer, and discussion prompts):\n\n{text}"
                out, err = call_openai(prompt, api_key, model="gpt-4o-mini", max_tokens=400)
                if err:
                    st.error("OpenAI call failed: " + err)
                else:
                    st.subheader("Questions (OpenAI)")
                    st.markdown(out)
            else:
                qs = generate_questions_from_text(text, n=question_count)
                st.subheader("Questions (heuristic)")
                for i,q in enumerate(qs,1):
                    st.write(f"{i}. {q['q']}")

st.markdown("---")
st.markdown("### How it works")
st.markdown(
    "- The app uses **simple extractive** methods locally (no API key required). These are fast and private but heuristic.\n"
    "- If you supply an **OpenAI API key** in the sidebar and check *Use OpenAI*, the app will call GPT for much higher-quality summaries, flashcards, and questions.\n"
    "- Local heuristics are transparent and work offline; they provide a good baseline for studying and quick iteration."
)

st.markdown("---")
st.caption("Built with simple Python heuristics + optional OpenAI. This is a starter app — you can improve the heuristics or lock to OpenAI prompts for better results.")
