# PolicyLens | Document Intelligence

Search fictional policy documents using TF-IDF and cosine similarity. Returns source-linked passages and abstains when evidence is insufficient. Optional AI drafting requires OPENAI_API_KEY and OPENAI_MODEL.

## Quick start
```bash
pip install -r requirements.txt
python rag.py "When must damage reports be submitted?"
python -m pytest -q
```
Independent AI-assisted demonstration; fictional documents, not official policy. Live API mode requires separate verification. Never submit confidential information.
