import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, classification_report
)

# Avaliação do modelo
def avaliar_modelo(nome, modelo, X_train, X_test, y_train, y_test):
    print(f"\n===== {nome} =====")

    start = time.time()
    modelo.fit(X_train, y_train)
    pred = modelo.predict(X_test)
    end = time.time()

    acc = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred, average="macro", zero_division=0)
    recall = recall_score(y_test, pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, pred, average="macro", zero_division=0)
    tempo = end - start

    print(f"Acurácia: {acc:.4f}")
    print(f"Precisão (macro): {precision:.4f}")
    print(f"Recall (macro): {recall:.4f}")
    print(f"F1-score (macro): {f1:.4f}")
    print(f"Tempo total (s): {tempo:.2f} s\n")

    print(classification_report(y_test, pred, zero_division=0))


# Carrega o arquivo como texto
with open("high_popularity_spotify_data.csv", "r", encoding="utf-8") as f:
    text = f.read()

# Corrige problema de músicas com mais de um artista, identificado por aspas duplas
text = text.replace('""', '"')

# Cria o novo arquivo com os dados ajustados para anállise
with open("spotify_clean.csv", "w", encoding="utf-8") as f:
    f.write(text)

# Faz a leitura do arquivo
df = pd.read_csv(
    "spotify_clean.csv",
    sep=",",
    quotechar='"',
    engine="python"
)

# Contabiliza as ocorrências da coluna de gênero musical, identificando casos com menos de 15 ocorrências e classificando-os juntos como 'OUTROS'
vc = df['playlist_genre'].value_counts()
rare_classes = vc[vc < 15].index

df['playlist_genre'] = df['playlist_genre'].replace(rare_classes, 'OUTROS')

# Realiza agrupamento geral de gêneros musicais semelhantes, buscando melhor padronização para os modelos
agrupamento = {
    "rock": "rock",
    "metal": "rock",
    "punk": "rock",
    "blues": "rock",
    
    "pop": "pop",
    "latin": "pop",
    "r&b": "pop",
    "k-pop": "pop",
    "j-pop": "pop",

    "hip-hop": "hip-hop",
    "afrobeats": "hip-hop",

    "electronic": "eletrônica",
    "gaming": "eletrônica",
    "ambient": "eletrônica",

    "folk": "folk",
    "indie": "folk",
    "country": "folk",

    "classical": "clássica",
    "arabic": "mundo",
    "indian": "mundo",
    "turkish": "mundo",
    "reggae": "mundo",

    "OUTROS": "OUTROS"
}

df["playlist_genre"] = df["playlist_genre"].map(agrupamento).fillna("OUTROS")

# Define o atributo alvo como gênero da playlist
target = "playlist_genre"

X = df.drop(columns=[target])
y = df[target]

# Define as colunas numéricas que serão normalizadas/analisadas
### Inicialmente utilizava dados categóricos (nomes, artistas, etc), o que retornava uma acurácia muito alta
### Com o intuito de realizar um teste mais fiel a situações reais, foram utilizados apenas dados numéricos das músicas
numeric_features = [
    "energy", "tempo", "danceability", "loudness",
    "liveness", "valence", "time_signature", "speechiness",
    "track_popularity", "instrumentalness", "mode", "key",
    "duration_ms", "acousticness"
]

# Normalização dos dados numéricos
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features)
    ]
)

# Modelos
models = {
    "RandomForest": Pipeline([
        ("preprocess", preprocessor),
        ("clf", RandomForestClassifier(n_estimators=200, random_state=42))
    ]),
    "SVM_RBF": Pipeline([
        ("preprocess", preprocessor),
        ("clf", SVC(kernel="rbf", gamma="scale"))
    ]),
    "MLP": Pipeline([
        ("preprocess", preprocessor),
        ("clf", MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42))
    ])
}

# Definição de dados para teste e treino
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Execução e avaliação dos modelos
for nome, modelo in models.items():
    avaliar_modelo(nome, modelo, X_train, X_test, y_train, y_test)
