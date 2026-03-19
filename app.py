import streamlit as st
from transformers import pipeline
from deep_translator import GoogleTranslator
from utils import clean_text, detect_language

st.title("Sexism Content Detection")
st.write("Multilingual AI for Toxicity & Sexism Detection")
@st.cache_resource
def load_models():
    toxicity_model = pipeline(
        "text-classification",
        model="unitary/toxic-bert",
        top_k=None
    )
    sexism_model = pipeline(
        "zero-shot-classification",
        model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
    )
    return toxicity_model, sexism_model

toxicity_model, sexism_model = load_models()
def get_toxicity(text):
    results = toxicity_model(text)[0]
    return {r['label']: r['score'] for r in results}

def get_sexism(text):
    labels = ["sexist", "non-sexist", "gender discrimination"]
    result = sexism_model(text, labels)
    return dict(zip(result['labels'], result['scores']))

def translate_to_english(text, lang): #translation for easy analysis
    if lang != "en":
        try:
            return GoogleTranslator(source='auto', target='en').translate(text)
        except:
            return text
    return text

def get_language_name(lang):
    lang_map = {
        "en": "English",
        "es": "Spanish",
        "fr": "French",
        "hi": "Hindi",
        "ar": "Arabic",
        "de": "German",
        "ne": "Hindi"
    }
    return lang_map.get(lang, lang)

#UI
text = st.text_area("Enter your comment:")

if st.button("Analyze"):
    if text.strip() == "":
        st.warning("Please enter text")
    else:
        clean = clean_text(text)
        lang, conf = detect_language(clean)
        language_name = get_language_name(lang)
        confidence_percent = (1 / (1 + abs(conf))) * 100
        st.subheader("Language Detection")
        st.write(f"{language_name} ({confidence_percent:.1f}%)")
        translated_text = translate_to_english(clean, lang)
        st.subheader("Translated Text")
        st.info(translated_text)
        #running translated text on model
        tox = get_toxicity(translated_text)

        st.subheader("Toxicity Scores")
        st.metric("Overall Toxicity", f"{tox['toxic']*100:.1f}%")
        for label, score in tox.items():
            percent = score * 100
            st.write(f"{label}: {percent:.1f}%")
            st.progress(float(score))
        sex = get_sexism(translated_text)

        st.subheader("Sexism Detection")
        for label, score in sex.items():
            percent = score * 100
            st.write(f"{label}: {percent:.1f}%")
            st.progress(float(score))

        st.subheader("Final Decision")
        sexist_score = sex.get("sexist", 0)
        gender_score = sex.get("gender discrimination", 0)
        combined_score = max(sexist_score, gender_score)
        keywords = ["women", "girls", "female", "ladki", "mahila"]
        text_lower = translated_text.lower()
        keyword_flag = any(word in text_lower for word in keywords)

        if combined_score > 0.3 or (keyword_flag and combined_score > 0.2):
            st.error("Potentially Sexist Content")
        elif tox["toxic"] > 0.5:
            st.warning("Toxic Content Detected")
        else:
            st.success("Safe Content")