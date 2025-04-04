import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from category_encoders import CatBoostEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.model_selection import cross_val_predict
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from imblearn.under_sampling import RandomUnderSampler
import pickle

## CARREGAR O ARQUIVO ##
df = pd.read_csv('datalab_churn.csv', encoding='windows-1252')

## TRATAR OS DADOS DATAFRAME ##

df.columns = [
    'Indice',
    'Falha de chamada',
    'Reclamacoes',
    'Duracao da assinatura',
    'Valor da cobranca',
    'Segundos de uso',
    'Frequência de uso',
    'Frequência de SMS',
    'Números distintos chamados',
    'Faixa etaria',
    'Plano tarifario',
    'Status',
    'Idade',
    'Valor do cliente',
    'Churn']

df = df.drop('Indice', axis=1)

df.dropna(inplace=True)

df_concatenado = df

df_concatenado['Churn'] = df_concatenado['Churn'].replace(
    {'Não churn': 0, 'Churn': 1})  # binário

X = df_concatenado[['Falha de chamada', 'Reclamacoes', 'Duracao da assinatura', 'Valor da cobranca', 'Segundos de uso',
                    'Frequência de uso', 'Frequência de SMS', 'Números distintos chamados', 'Faixa etaria', 'Plano tarifario', 'Status',
                    'Idade', 'Valor do cliente']]  # caracteristicas

y = df_concatenado['Churn']  # target

# Separar as variáveis numéricas e categóricas
categorical_cols = ['Reclamacoes', 'Faixa etaria', 'Plano tarifario', 'Status']
numerical_cols = ['Falha de chamada', 'Duracao da assinatura', 'Valor da cobranca', 'Segundos de uso',
                  'Frequência de uso', 'Frequência de SMS', 'Números distintos chamados',
                  'Idade', 'Valor do cliente']

# Criar o pipeline
pipeline = Pipeline([
    ('encoder', CatBoostEncoder(cols=categorical_cols, random_state=42)),
    ('scaler', MinMaxScaler())
])

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Aplicar o pipeline aos dados de treino
X_train_transformed = pipeline.fit_transform(X_train, y_train)

# Aplicar o pipeline aos dados de teste (usando o fit do treino)
X_test_transformed = pipeline.transform(X_test)


def treinar_e_avaliar_modelo(modelo, X_train_transformed, y_train, X_test_transformed, y_test):
    # Treinar o modelo com validação cruzada
    # Validação cruzada estratificada
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = cross_val_predict(modelo, X_train_transformed, y_train, cv=cv)

    # Calcular a curva ROC na base de treino
    y_prob = cross_val_predict(
        modelo, X_train_transformed, y_train, cv=cv, method='predict_proba')[:, 1]
    fpr_train, tpr_train, thresholds_train = roc_curve(y_train, y_prob)
    roc_auc_train = auc(fpr_train, tpr_train)

    # Avaliar o modelo na base de teste
    y_pred_test = modelo.predict(X_test_transformed)
    print("\nRelatório de Classificação (Teste):")
    print(classification_report(y_test, y_pred_test))

    # Calcular a curva ROC na base de teste
    y_prob_test = modelo.predict_proba(X_test_transformed)[:, 1]
    fpr_test, tpr_test, thresholds_test = roc_curve(y_test, y_prob_test)
    roc_auc_test = auc(fpr_test, tpr_test)

    # Plotar a curva ROC (treino e teste)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_train, tpr_train, color='darkorange', lw=2,
             label=f'Curva ROC (Treino) (área = {roc_auc_train:.2f})')
    plt.plot(fpr_test, tpr_test, color='green', lw=2,
             label=f'Curva ROC (Teste) (área = {roc_auc_test:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de Falsos Positivos')
    plt.ylabel('Taxa de Verdadeiros Positivos')
    plt.title('Curva ROC (Treino e Teste)')
    plt.legend(loc="lower right")
    plt.show()


# Inicializa o RandomUnderSampler
rus = RandomUnderSampler(random_state=42)

# Aplica o undersampling no conjunto de dados
X_resampled, y_resampled = rus.fit_resample(
    df_concatenado.drop('Churn', axis=1), df_concatenado['Churn'])

# Cria um novo DataFrame com os dados balanceados
df_balanced = pd.concat([pd.DataFrame(X_resampled), pd.DataFrame(
    y_resampled, columns=['Churn'])], axis=1)


X = df_balanced[['Falha de chamada', 'Reclamacoes', 'Duracao da assinatura', 'Valor da cobranca', 'Segundos de uso',
                 'Frequência de uso', 'Frequência de SMS', 'Números distintos chamados', 'Faixa etaria', 'Plano tarifario', 'Status',
                 'Idade', 'Valor do cliente']]  # caracteristicas

y = df_balanced['Churn']  # target

# Separar as variáveis numéricas e categóricas
categorical_cols = ['Reclamacoes', 'Faixa etaria', 'Plano tarifario', 'Status']
numerical_cols = ['Falha de chamada', 'Duracao da assinatura', 'Valor da cobranca', 'Segundos de uso',
                  'Frequência de uso', 'Frequência de SMS', 'Números distintos chamados',
                  'Idade', 'Valor do cliente']

# Criar o pipeline
pipeline = Pipeline([
    ('encoder', CatBoostEncoder(cols=categorical_cols, random_state=42)),
    ('scaler', MinMaxScaler())
])

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Aplicar o pipeline aos dados de treino
X_train_transformed = pipeline.fit_transform(X_train, y_train)

# Aplicar o pipeline aos dados de teste (usando o fit do treino)
X_test_transformed = pipeline.transform(X_test)

modelo_upgrade = RandomForestClassifier(random_state=42)
modelo_upgrade.fit(X_train_transformed, y_train)

# Inicializa o RandomUnderSampler
rus = RandomUnderSampler(random_state=42)

# Aplica o undersampling no conjunto de dados
X_resampled, y_resampled = rus.fit_resample(
    df_concatenado.drop('Churn', axis=1), df_concatenado['Churn'])

# Cria um novo DataFrame com os dados balanceados
df_balanced = pd.concat([pd.DataFrame(X_resampled), pd.DataFrame(
    y_resampled, columns=['Churn'])], axis=1)

X = df_balanced[['Falha de chamada', 'Reclamacoes', 'Duracao da assinatura', 'Valor da cobranca', 'Segundos de uso',
                 'Frequência de uso', 'Frequência de SMS', 'Números distintos chamados', 'Faixa etaria', 'Plano tarifario', 'Status',
                 'Idade', 'Valor do cliente']]  # caracteristicas

y = df_balanced['Churn']  # target

# Separar as variáveis numéricas e categóricas
categorical_cols = ['Reclamacoes', 'Faixa etaria', 'Plano tarifario', 'Status']
numerical_cols = ['Falha de chamada', 'Duracao da assinatura', 'Valor da cobranca', 'Segundos de uso',
                  'Frequência de uso', 'Frequência de SMS', 'Números distintos chamados',
                  'Idade', 'Valor do cliente']

# Criar o pipeline
pipeline = Pipeline([
    ('encoder', CatBoostEncoder(cols=categorical_cols, random_state=42)),
    ('scaler', MinMaxScaler())
])

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Aplicar o pipeline aos dados de treino
X_train_transformed = pipeline.fit_transform(X_train, y_train)

# Aplicar o pipeline aos dados de teste (usando o fit do treino)
X_test_transformed = pipeline.transform(X_test)

modelo_upgrade = RandomForestClassifier(random_state=42)
modelo_upgrade.fit(X_train_transformed, y_train)

# Salve a pipeline
filename = 'modelo_upgrade.pkl'
pickle.dump(modelo_upgrade, open(filename, 'wb'))
