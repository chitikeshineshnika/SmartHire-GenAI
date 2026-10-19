import os
from src.parsing import loader

print("Functions in loader.py:", [n for n in dir(loader) if not n.startswith("_")])

PATH = "tests/fixtures/sample_resume.pdf"
print("File exists:", os.path.exists(PATH))

from src.parsing.loader import load_document   # change if your function has another name

result = load_document(PATH)
print("Type:", type(result))
print("Length:", len(result) if hasattr(result, "__len__") else "n/a")
print("Preview:", repr(result)[:300])