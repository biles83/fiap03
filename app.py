from azure.storage.blob import BlobServiceClient
import pandas as pd
import pandas as pd
import pickle
from modelo import pipeline

## CARREGAR O MODELO DE IA ##

with open("modelo_upgrade.pkl", "rb") as arquivo:
    modelo_upgrade = pickle.load(arquivo)

## CARREGAR O ARQUIVO ##

df_validacao = pd.read_csv(
    'novos_dados.csv',  encoding='windows-1252')

df_validacao.columns = [
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

df_validacao = df_validacao.drop('Indice', axis=1)

# Selecione as features relevantes para a predição
X_validacao = df_validacao[['Falha de chamada', 'Reclamacoes', 'Duracao da assinatura', 'Valor da cobranca', 'Segundos de uso',
                            'Frequência de uso', 'Frequência de SMS', 'Números distintos chamados', 'Faixa etaria', 'Plano tarifario', 'Status',
                            'Idade', 'Valor do cliente']]

# Separar as variáveis numéricas e categóricas
categorical_cols = ['Reclamacoes', 'Faixa etaria', 'Plano tarifario', 'Status']
numerical_cols = ['Falha de chamada', 'Duracao da assinatura', 'Valor da cobranca', 'Segundos de uso',
                  'Frequência de uso', 'Frequência de SMS', 'Números distintos chamados',
                  'Idade', 'Valor do cliente']

# Aplicar o pipeline aos dados de teste (usando o fit do treino)
X_validacao_transformed = pipeline.transform(X_validacao)

# Faça a predição usando o pipeline treinado
y_pred = modelo_upgrade.predict(X_validacao_transformed)

# Adicione as predições ao DataFrame de validação
df_validacao['Predição'] = y_pred

# Exporta arquivo com a Predição
df_validacao.to_csv('predicao.csv', index=False)

## EVIAR O ARQUIVO PARA A CLOUD ##

storage_account_key = "T5NMdY7rnwKb+k8Qsr2FFLdovFnMsg3lyrlOR/CteRL5pDLuhpPpth6OAoYqZf23GGEKA/e5uc31+ASt4KOlWw=="
storage_account_name = "fiaptechchallengefase03"
connection_string = "DefaultEndpointsProtocol=https;AccountName=fiaptechchallengefase03;AccountKey=T5NMdY7rnwKb+k8Qsr2FFLdovFnMsg3lyrlOR/CteRL5pDLuhpPpth6OAoYqZf23GGEKA/e5uc31+ASt4KOlWw==;EndpointSuffix=core.windows.net"
container_name = "dados"


def uploadToBlobStorage(file_path, file_name):
    blob_service_client = BlobServiceClient.from_connection_string(
        connection_string)
    blob_client = blob_service_client.get_blob_client(
        container=container_name, blob=file_name)
    with open(file_path, "rb") as data:
        blob_client.upload_blob(data)
        print(f"Uploaded {file_name}.")


# calling a function to perform upload
uploadToBlobStorage(
    'predicao.csv', 'predicao.csv')
