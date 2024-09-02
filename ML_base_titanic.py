# Manipulação e Tratamento de dados
import openpyxl
import pandas as pd
import numpy as np
from numpy import NaN

#ignorando Warning inuteis
import warnings 
from pandas.errors import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
warnings.filterwarnings(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

#################################################################################

# Carregar os dados
url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
df = pd.read_csv(url)

# Identificar colunas a serem removidas # Remover colunas inúteis
columns_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin']
df = df.drop(columns=columns_to_drop)
df = df.dropna()

#################################################################################

colunas_cat = ['Pclass','Sex','Embarked']
for coluna in colunas_cat:
    df[coluna] = df[coluna].astype('O')

colunas_categoricas = []
for coluna in df.columns:
    if df[coluna].dtype == 'O':
        categorias = df[coluna].unique()
        if len(categorias) == 2:
            colunas_categoricas.append(coluna)
        else:
            colunas_categoricas.append(coluna)

#################################################################################

x = df.drop('Survived', axis=1)
y = df['Survived']

#################################################################################

# DUMMYRISAÇÃO
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder #transformando colunas com 2 categorias em 0 e 1

coluna = x.columns
one_hot = make_column_transformer((
    OneHotEncoder(drop='if_binary'), #caso a coluna tenha apenas 2 categorias 
    colunas_categoricas), #passando quais são essas colunas
    remainder = 'passthrough', sparse_threshold=0) #oque deve ser feito com as outras

#Aplicando transformação
x = one_hot.fit_transform(x)

#Os novos nomes das colunas #'onehotencoder=transformadas; 'remainder'=não transformadas
novos_nomes_colunas = one_hot.get_feature_names_out(coluna)

########################################################################################

# PADRONIZAÇÃO DOS DADOS
from sklearn.preprocessing import MinMaxScaler

normalizacao = MinMaxScaler()
#x = normalizacao.fit_transform(x)

x = pd.DataFrame(x, columns = novos_nomes_colunas) #alterando de volta
x_columns = x.columns.tolist() 

#################################################################################

x.reset_index(drop=True, inplace=True)
y.reset_index(drop=True, inplace=True)
x_backup = x
y_backup = y