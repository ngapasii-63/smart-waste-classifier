import os

folders = [
    "data/raw",
    "data/processed",
    "docs",
    "models",
    "notebooks",
    "src",
    "tests",
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)
    print(f"Folder siap: {folder}/")

gitignore_content = """.venv/
__pycache__/
*.pyc
data/raw/*
data/processed/*
models/*.h5
models/*.keras
.DS_Store
"""

if not os.path.exists(".gitignore"):
    with open(".gitignore", "w", encoding="utf-8") as f:
        f.write(gitignore_content)
    print("File .gitignore berhasil dibuat.")

print("\nStruktur folder proyek selesai dibuat!")