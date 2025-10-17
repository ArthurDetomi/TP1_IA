import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder

games = pd.read_csv("../data/games.csv")

print(games.columns)

# Etapa 1 limpeza dos dados

# Removendo dados irrelevantes
games = games.drop(columns=['AppID', 'Screenshots', 'Movies', 'Metacritic url', 'Header image', 'Website', 'Support url', 'Support email', 'DiscountDLC count', 'Achievements', 'Notes'])
# Removendo dados duplicados
games = games.drop_duplicates()

# Removendo dados que possuem valores ausentes
games.dropna(inplace=True)

print(games.columns)


df_sample = games.sample(frac=0.3, random_state=42)

print(df_sample)

games['Release date'] = pd.to_datetime(games['Release date'], errors='coerce')

print(games["Genres"])


# Etapa 2 Transformação dos dados
