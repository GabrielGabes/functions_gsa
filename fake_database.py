#ignorando Warning inuteis
import warnings 
from pandas.errors import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
warnings.filterwarnings(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

from sklearn.datasets import make_classification
import pandas as pd
import numpy as np

# DADOS FICTICIOS 1
def fake_database(n_samples=1000, seed=3141592):
    np.random.seed(1)

    Tamanho_da_amostra = n_samples
    n = Tamanho_da_amostra/2 ; n = int(n)
    df = pd.DataFrame()

    df['desfecho'] = ['não'] * n + ['sim'] * n
    filtro_1 = df['desfecho'] == 'não'
    filtro_0 = df['desfecho'] == 'sim'

    lista_col = ['idade','sexo','tabagismo','avc_previo','hipo','has',
                'dm','doenca_cardiaca','var_num1','var_num2']
    for coluna in lista_col: df[coluna] = np.nan

    df['sexo'][filtro_0] = np.random.choice(['F', 'M'], size=n, p=[0.4, 0.6])
    df['sexo'][filtro_1] = np.random.choice(['F', 'M'], size=n, p=[0.7, 0.3])

    df['idade'][filtro_0] = np.random.normal(loc=30, scale=5, size=n).round().astype(int)
    df['idade'][filtro_1] = np.random.normal(loc=23, scale=4, size=n).round().astype(int)

    df['tabagismo'][filtro_0] = np.random.choice(['não', 'sim'], size=n, p=[0.3, 0.7])
    df['tabagismo'][filtro_1] = np.random.choice(['não', 'sim'], size=n, p=[0.4, 0.6])

    df['avc_previo'][filtro_0] = np.random.choice(['não', 'sim'], size=n, p=[0.3, 0.7])
    df['avc_previo'][filtro_1] = np.random.choice(['não', 'sim'], size=n, p=[0.5, 0.5])

    df['hipo'][filtro_0] = np.random.choice(['não', 'sim'], size=n, p=[0.2, 0.8])
    df['hipo'][filtro_1] = np.random.choice(['não', 'sim'], size=n, p=[0.3, 0.7])

    df['has'][filtro_0] = np.random.choice(['não', 'sim'], size=n, p=[0.5, 0.5])
    df['has'][filtro_1] = np.random.choice(['não', 'sim'], size=n, p=[0.2, 0.8])

    df['dm'][filtro_0] = np.random.choice(['não', 'sim'], size=n, p=[0.8, 0.2])
    df['dm'][filtro_1] = np.random.choice(['não', 'sim'], size=n, p=[0.3, 0.7])

    df['doenca_cardiaca'][filtro_0] = np.random.choice(['não', 'sim'], size=n, p=[0.2, 0.8])
    df['doenca_cardiaca'][filtro_1] = np.random.choice(['não', 'sim'], size=n, p=[0.6, 0.4])

    df['var_num1'][filtro_0] = np.random.poisson(12,n)
    df['var_num1'][filtro_1] = np.random.poisson(20,n)

    df['var_num2'][filtro_0] = np.random.uniform(0,100,n)
    df['var_num2'][filtro_1] = np.random.uniform(0,100,n)
    
    return df

# DADOS FICTICIOS 2
def fake_database2(n_samples=1000, seed=3141592):
    np.random.seed(seed)

    # Gerando dados numéricos e a variável dependente 'y'
    x, y = make_classification(n_samples=n_samples, 
                            n_features=10, #total de colunas
                            n_informative=5, #colunas informativas
                            n_redundant=5, #colunas redundantes
                            n_classes=2, #numero de classes y
                            random_state=n_samples)

    df = pd.DataFrame(x)
    df.columns = ['x_num'+str(i) for i in range(10)]

    df['y'] = y.astype('O')

    # Adicionando 10 variáveis categóricas binarias

    for i in range(5):
        if i >= 3:
            df['x_bin'+str(i)] = np.random.choice(['sim','não'], size=len(df))
        else:
            df['x_bin'+str(i)] = np.nan
            df['x_bin'+str(i)][df['y'] == 0] = np.random.choice(['sim','não'], size=len(df), p=[.7,.3])
            df['x_bin'+str(i)][df['y'] == 1] = np.random.choice(['sim','não'], size=len(df), p=[.3,.7])

    for i in range(3):
        if i >= 2:
            df['x_cat'+str(i)] = np.random.choice(['A','B','C',], size=len(df))
        else:
            df['x_cat'+str(i)] = np.nan
            df['x_cat'+str(i)][df['y'] == 0] = np.random.choice(['A','B','C',], size=len(df), p=[.3,.3,.4])
            df['x_cat'+str(i)][df['y'] == 1] = np.random.choice(['A','B','C',], size=len(df), p=[.2,.7,.1])

    for i in range(2):
        if i >= 1:
            df['x_cat_'+str(i)] = np.random.choice(['A','B','C','D'], size=len(df))
        else:
            df['x_cat_'+str(i)] = np.nan
            df['x_cat_'+str(i)][df['y'] == 0] = np.random.choice(['A','B','C','D'], size=len(df), p=[.4,.3,.2,.1])
            df['x_cat_'+str(i)][df['y'] == 1] = np.random.choice(['A','B','C','D'], size=len(df), p=[.1,.2,.3,.4])

    return df