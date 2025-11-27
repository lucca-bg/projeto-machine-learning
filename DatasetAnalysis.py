import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Carrega o arquivo como texto
with open("high_popularity_spotify_data.csv", "r", encoding="utf-8") as f:
    text = f.read()

# Corrige problema de músicas com mais de um artista
text = text.replace('""', '"')

# Salva arquivo limpo
with open("spotify_clean.csv", "w", encoding="utf-8") as f:
    f.write(text)

df = pd.read_csv(
    "spotify_clean.csv",
    sep=",",
    quotechar='"',
    engine="python"
)
df.head()
df.info()
df.describe()

vc = df['playlist_genre'].value_counts()
rare_classes = vc[vc < 15].index

df['playlist_genre'] = df['playlist_genre'].replace(rare_classes, 'OUTROS')

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

    "electronic": "electronic",
    "gaming": "electronic",
    "ambient": "electronic",

    "folk": "folk",
    "indie": "folk",
    "country": "folk",

    "classical": "classical",
    "arabic": "world",
    "indian": "world",
    "turkish": "world",
    "reggae": "world",

    "OUTROS": "OUTROS"
}
df["playlist_genre"] = df["playlist_genre"].map(agrupamento).fillna("OUTROS")


target = "playlist_genre"

X = df.drop(columns=[target])
y = df[target]

numeric_features = [
    "energy", "tempo", "danceability", "loudness",
    "liveness", "valence", "time_signature", "speechiness",
    "track_popularity", "instrumentalness", "mode", "key",
    "duration_ms", "acousticness"
]

categorical_features = []

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

model = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=200, random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

print("Acurácia:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))


