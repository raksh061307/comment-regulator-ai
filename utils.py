import re
import langid

def clean_text(text):
    text = re.sub(r'<.*?>', '', text)  
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower().strip()
    return text[:1000]

def detect_language(text):
    lang, confidence = langid.classify(text)
    return lang, confidence