# Web UI package (`ui/`)

Python helpers and static frontend for the dashboard. **Start the server from the repo root:**

```bash
cd ..   # project root
pip install -r requirements.txt
python app.py
```

Open **http://127.0.0.1:5001**

## Contents

| Path | Role |
|------|------|
| `model_pipeline.py` | Registry-driven checkpoint loader |
| `data_pipeline.py` | Unseen CSV → SKAB column mapping |
| `torch_models.py` | PyTorch inference |
| `constants.py` | 8 SKAB feature columns |
| `frontend/` | `index.html`, `script.js`, `style.css`, `docs.html` |
| `test_data/` | Sample CSVs |

Project docs: [../README.md](../README.md)
