## Predizendo a evasão de clientes de Telecom - CHURN -


Redução do churn de clientes da empresa.
O conjunto de dados inclui diversas informações sobre o uso do plano por parte dos clientes. O problema é formulado como uma tarefa de classificação de duas categorias (Churn e Não Churn).

Criar modelos uma pipeline preditiva de um modelo de classificação 🎯


## 📁 Estrutura do Projeto

```bash

FIAP03/
  ├── __init__.py
  ├── requirements.txt
  ├── app.py
  ├── modelo.py
  ├── datalab_churn.csv
  ├── novos_dados.csv
  ├── FIAP_Modelo_Churn.ipynb
  ├── gitignore.txt
  ├── modelo_upgrade.pkl
  ├── arquitetura.jpg
  ├── Documentação_API.docx
  ├── Churn.pbix
  └── README.md
```

- **`FIAP03/`**: Diretório principal do aplicativo.
- **`app.py`**: Fonte para rodar o modelo treinado e enviar para a nuvem.
- **`modelo.py`**: Fonte para treinamento e deração do modelo de churn.
- **`requirements.txt`**: Lista de dependências do projeto.
- **`FIAP_Modelo_Churn.ipynb`**: Documentação com os testes e códigos.
- **`arquitetura.jpg`**: Desenho da Arquitetura do Projeto.
- **`novos_dados.csv`**: Dados novos a serem rodados pelo modelo.
- **`datalab_churn.csv`**: Dados dos clientes.
- **`modelo_upgrade.pkl`**: Modelo Exportado.
- **`Churn.pbix`**: Relatório contendo as informações sobre Churn geradas pelo Modelo Preditivo.
- **`README.md`**: Documentação do projeto.

## 🛠️ Como Executar o Projeto

### 1. Clone o Repositório

```bash
git clone https://github.com/biles83/fiap03
```

### 2. Crie um Ambiente Virtual

```bash
python -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate
```

### 3. Instale as Dependências

```bash
pip install -r requirements.txt
```

### 4. Execute o Código

```bash
python modelo.py
python app.py
```

### 5. Execute o Relatório

Churn.pbix => Microsoft Power BI.


## 📖 Documentação do Projeto

A documentação do projeto, assim como o código usado, análise e conclusões, está disponível no notebook "FIAP_Modelo_Churn.ipynb", na raiz do projeto.

```bash
FIAP03/
  ├── FIAP_Modelo_Churn.ipynb
```

## 🤝 Contribuindo

1. Fork este repositório.
2. Crie sua branch (`git checkout -b feature/nova-funcionalidade`).
3. Faça commit das suas alterações (`git commit -m 'Adiciona nova funcionalidade'`).
4. Faça push para sua branch (`git push origin feature/nova-funcionalidade`).
5. Abra um Pull Request.
instalar, configurar e usar o projeto. Ele também cobre contribuições, contato, licença e agradecimentos, tornando-o completo e fácil de entender para novos desenvolvedores.