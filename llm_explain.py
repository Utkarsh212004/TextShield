import os
import anthropic
import streamlit as st


def get_key():
    try:
        return st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        return os.environ.get("ANTHROPIC_API_KEY")


def explain(text, label, confidence, kind):
    key = get_key()
    if not key:
        return None

    client = anthropic.Anthropic(api_key=key)
    prompt = (
        f"A {kind} classifier labelled this text as '{label}' "
        f"with {confidence}% confidence.\n"
        "In 2-3 simple sentences, explain which words or writing patterns "
        f"make it look {label}. Do not say whether the label is right or wrong.\n\n"
        f"Text:\n{text[:3000]}"
    )

    try:
        msg = client.messages.create(
            model="claude-haiku-5-5",
            max_tokens=250,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text
    except Exception as e:
        return f"AI explanation unavailable ({type(e).__name__})."
