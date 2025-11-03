import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

## Carregar os Dados
file_path = 'dataset/games.csv'
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Erro: O arquivo '{file_path}' não foi encontrado.")
    exit()
except Exception as e:
    print(f"Ocorreu um erro ao carregar o arquivo: {e}")
    exit()

## Coleta e Pré-processamento dos Dados

target_column = 'User Rating'
drop_cols = [target_column, 'Game Title', 'User Review Text', 'Developer', 'Publisher']

X = df.drop(columns=drop_cols)
y = df[target_column]

# Mapeamento para colunas ordinais
graphics_mapping = {
    'Low': 0,
    'Medium': 1,
    'High': 2,
    'Ultra': 3
}

quality_mapping = {
    'Poor': 0,
    'Average': 1,
    'Good': 2,
    'Excellent': 3
}

try:
    X['Graphics Quality'] = X['Graphics Quality'].map(graphics_mapping)
    X['Soundtrack Quality'] = X['Soundtrack Quality'].map(quality_mapping)
    X['Story Quality'] = X['Story Quality'].map(quality_mapping)
except KeyError as e:
    print(f"Erro no mapeamento: Valor inesperado encontrado. {e}")

# Features numéricas
numeric_features = ['Price', 'Game Length (Hours)', 'Release Year', 'Min Number of Players',
                    'Graphics Quality', 'Soundtrack Quality', 'Story Quality']

# Features categóricas
categorical_features = ['Age Group Targeted', 'Platform', 'Requires Special Device', 'Genre', 'Multiplayer', 'Game Mode']

# Divisão dos Dados (80% treino, 20% teste)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Dados divididos: {len(X_train)} para treino, {len(X_test)} para teste.\n")


#StandardScaler
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

#OneHotEncoder
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])


## Desenvolvimento do Modelo

# Regressão Linear
lr_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('regressor', LinearRegression())])
lr_pipeline.fit(X_train, y_train)

# Random Forest Regressor
rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                             ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))])
rf_pipeline.fit(X_train, y_train)


## Análise dos Resultados

y_pred_lr = lr_pipeline.predict(X_test)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
mae_lr = mean_absolute_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

print(f"Resultados (Regressão Linear):")
print(f"  R-squared (R2): {r2_lr:.4f}")
print(f"  Mean Absolute Error (MAE): {mae_lr:.4f}")
print(f"  Root Mean Squared Error (RMSE): {rmse_lr:.4f}")

y_pred_rf = rf_pipeline.predict(X_test)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
mae_rf = mean_absolute_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print(f"Resultados (Random Forest):")
print(f"  R-squared (R2): {r2_rf:.4f}")
print(f"  Mean Absolute Error (MAE): {mae_rf:.4f}")
print(f"  Root Mean Squared Error (RMSE): {rmse_rf:.4f}")

try:
    preprocessor_trained = rf_pipeline.named_steps['preprocessor']
    model_trained = rf_pipeline.named_steps['regressor']
    
    all_feature_names = preprocessor_trained.get_feature_names_out()
    
    importances = model_trained.feature_importances_
    
    feature_importance_df = pd.DataFrame({
        'Feature': all_feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    print("\nTop 10 features mais importantes:")
    print(feature_importance_df.head(10).to_markdown(index=False, numalign="left", stralign="left"))

except Exception as e:
    print(f"Ocorreu um erro ao extrair a importância das features: {e}")