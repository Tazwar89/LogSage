import re

PATTERNS = {
    "ip": re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b'),
    "email": re.compile(r'\b[\w.+-]+@[\w-]+\.[\w.-]+\b'),
    "api_key": re.compile(r'\b(sk|api|key)[-_][A-Za-z0-9]{16,}\b', re.IGNORECASE),
}

def redact(text):
    for label, pattern in PATTERNS.items():
        text = pattern.sub(f"[REDACTED_{label.upper()}]", text)

    return text