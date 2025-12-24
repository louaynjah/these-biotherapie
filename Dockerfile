# Utilisation de Python 3.11 (compatible avec tes modèles)
FROM python:3.11-slim

# Définition du dossier de travail
WORKDIR /app

# Copie et installation des bibliothèques
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie de tout ton code (app.py + modeles_ia)
COPY . .

# Commande de lancement de l'app
CMD ["streamlit", "run", "app.py"]
