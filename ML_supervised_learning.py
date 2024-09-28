import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, RocCurveDisplay

# Função para avaliação de modelos exibindo metricas de avaliação
def avaliar_modelo(y_verdadeiro, y_teste, conjunto_nome="", plotar_grafico=False):
    """
    Função para avaliar e imprimir o relatório de classificação e a matriz de confusão.
    """
    #if conjunto_nome == "Treino":
     #   pass
    #else:
    # Metricas
    print('*' * 70)
    # Matriz
    print("Matriz de Confusão:", conjunto_nome,":\n")
    print(confusion_matrix(y_verdadeiro, y_predito))
    
    print('*' * 50)
    
    print("Relatório de Classificação:", conjunto_nome,":\n")
    print(classification_report(y_verdadeiro, y_predito))
    print('*' * 70)

    if plotar_grafico == True:
        display(RocCurveDisplay.from_predictions(y_verdadeiro, y_predito, name = conjunto_nome))
########################################################################################################################################################

# Função que cria tabela com todas metricas de avaliação em cada ponto de threshold
def aval_modelo_corte_tabela(x_teste, y_teste, classificador, beta = 1):
        
    previsoes_proba = classificador.predict_proba(x_teste)
    probs_positivas = previsoes_proba[:, 1]

    # Inicializando um DataFrame para armazenar as métricas
    fd = pd.DataFrame(columns=['threshold', 'tn', 'fp', 'fn', 'tp'])

    # Pontos de corte de 0 a 1, com passos de 0.10
    pontos_de_corte = np.arange(0, 1.1, 0.10)

    # Loop pelos pontos de corte
    for i in pontos_de_corte:
        threshold = i
        previsoes_personalizadas = (probs_positivas >= threshold).astype(int)

        # Calculando os valores de tn, fp, fn, tp
        tn = len(np.where((previsoes_personalizadas == 0) & (y_teste == 0))[0])
        fp = len(np.where((previsoes_personalizadas == 1) & (y_teste == 0))[0])
        fn = len(np.where((previsoes_personalizadas == 0) & (y_teste == 1))[0])
        tp = len(np.where((previsoes_personalizadas == 1) & (y_teste == 1))[0])

        # Adicionando os resultados ao DataFrame 'fd'
        fd = pd.concat([fd, pd.DataFrame([[i, tn, fp, fn, tp]], columns=fd.columns)])

    # Substituir valores NaN por 0
    fd.fillna(0, inplace=True)
    fd.reset_index(drop=True, inplace=True)
    ######################################################################################

    # Inicializar as colunas das métricas com NaN
    metricas_de_aval = ['acuracia', 'precisao', 'sensibilidade', 'especificidade', 'f1',
                        'valor_pre_posi', 'valor_pre_neg', 'taxa_falsos_positivos', 'taxa_falsos_negativos',
                        'fdr', 'fo_r', 'indice_youden', 'coef_matthews', 'fb_score']
    fd[metricas_de_aval] = np.nan

    # Loop para calcular as métricas em cada linha
    for l in range(len(fd)):
        # Calcular Acurácia
        if (fd['tn'][l] + fd['fp'][l] + fd['fn'][l] + fd['tp'][l]) != 0:
            fd['acuracia'][l] = (fd['tp'][l] + fd['tn'][l]) / (fd['tn'][l] + fd['fp'][l] + fd['fn'][l] + fd['tp'][l])
        else:
            fd['acuracia'][l] = 0

        # Calcular Precisão
        if (fd['tp'][l] + fd['fp'][l]) != 0:
            fd['precisao'][l] = fd['tp'][l] / (fd['tp'][l] + fd['fp'][l])
        else:
            fd['precisao'][l] = 0

        # Calcular Sensibilidade (Recall)
        if (fd['tp'][l] + fd['fn'][l]) != 0:
            fd['sensibilidade'][l] = fd['tp'][l] / (fd['tp'][l] + fd['fn'][l])
        else:
            fd['sensibilidade'][l] = 0

        # Calcular Especificidade
        if (fd['tn'][l] + fd['fp'][l]) != 0:
            fd['especificidade'][l] = fd['tn'][l] / (fd['tn'][l] + fd['fp'][l])
        else:
            fd['especificidade'][l] = 0

        # Calcular F1-Score
        if (fd['precisao'][l] + fd['sensibilidade'][l]) != 0:
            fd['f1'][l] = 2 * (fd['precisao'][l] * fd['sensibilidade'][l]) / (fd['precisao'][l] + fd['sensibilidade'][l])
        else:
            fd['f1'][l] = 0

        # Calcular Valor Preditivo Positivo
        if (fd['tp'][l] + fd['fp'][l]) != 0:
            fd['valor_pre_posi'][l] = fd['tp'][l] / (fd['tp'][l] + fd['fp'][l])
        else:
            fd['valor_pre_posi'][l] = 0

        # Calcular Valor Preditivo Negativo
        if (fd['tn'][l] + fd['fn'][l]) != 0:
            fd['valor_pre_neg'][l] = fd['tn'][l] / (fd['tn'][l] + fd['fn'][l])
        else:
            fd['valor_pre_neg'][l] = 0

        # Calcular Taxa de Falsos Positivos
        fd['taxa_falsos_positivos'][l] = 1 - fd['especificidade'][l]

        # Calcular Taxa de Falsos Negativos
        fd['taxa_falsos_negativos'][l] = 1 - fd['sensibilidade'][l]

        # Calcular False Discovery Rate (FDR)
        if (fd['tp'][l] + fd['fp'][l]) != 0:
            fd['fdr'][l] = fd['fp'][l] / (fd['tp'][l] + fd['fp'][l])
        else:
            fd['fdr'][l] = 0

        # Calcular False Omission Rate (FOR)
        if (fd['tn'][l] + fd['fn'][l]) != 0:
            fd['fo_r'][l] = fd['fn'][l] / (fd['tn'][l] + fd['fn'][l])
        else:
            fd['fo_r'][l] = 0

        # Calcular Índice de Youden
        fd['indice_youden'][l] = fd['sensibilidade'][l] + fd['especificidade'][l] - 1

        # Calcular Coeficiente de Matthews (MCC)
        denom_matthews = (fd['tp'][l]+fd['fp'][l])*(fd['tp'][l]+fd['fn'][l])*(fd['tn'][l]+fd['fp'][l])*(fd['tn'][l]+fd['fn'][l])
        if denom_matthews != 0:
            fd['coef_matthews'][l] = (fd['tp'][l]*fd['tn'][l] - fd['fp'][l]*fd['fn'][l]) / np.sqrt(denom_matthews)
        else:
            fd['coef_matthews'][l] = 0

        # Calcular F-beta Score
        denom_fb_score = (beta**2 * fd['valor_pre_posi'][l]) + fd['sensibilidade'][l]
        if denom_fb_score != 0:
            fd['fb_score'][l] = (1 + beta**2) * (fd['valor_pre_posi'][l] * fd['sensibilidade'][l]) / denom_fb_score
        else:
            fd['fb_score'][l] = 0

    fd[metricas_de_aval] = fd[metricas_de_aval].round(3)
    return fd
#######################################################

import matplotlib.pyplot as plt
from matplotlib import style
plt.style.use('ggplot')

# Função que plota um grafico da tabela acima
def aval_modelo_corte_grafico(fd):
    plt.figure(figsize=(12, 6))
    
    ####################################################################################
    # Primeiro modo de plotar os graficos
    #plt.plot(fd['threshold'], fd['acuracia'], label='Acurácia', marker='o')
    #plt.plot(fd['threshold'], fd['precisao'], label='Precisão', marker='o')
    #plt.plot(fd['threshold'], fd['sensibilidade'], label='Sensibilidade (Recall)', marker='o')
    #plt.plot(fd['threshold'], fd['especificidade'], label='Especificidade', marker='o')
    #plt.plot(fd['threshold'], fd['f1'], label='F1-Score', marker='o')
    #plt.plot(fd['threshold'], fd['valor_pre_posi'], label='Valor Pre Positivo', marker='o')
    #plt.plot(fd['threshold'], fd['valor_pre_neg'], label='Valor Pre Negativo', marker='o')
    #plt.plot(fd['threshold'], fd['taxa_falsos_positivos'], label='Taxa Falsos Positivos', marker='o')
    #plt.plot(fd['threshold'], fd['taxa_falsos_negativos'], label='Taxa Falsos Negativos', marker='o')
    #plt.plot(fd['threshold'], fd['fdr'], label='FDR', marker='o')
    #plt.plot(fd['threshold'], fd['fo_r'], label='FOR', marker='o')
    #plt.plot(fd['threshold'], fd['indice_youden'], label="Índice de Youden", marker='o')
    #plt.plot(fd['threshold'], fd['coef_matthews'], label='Coeficiente de Matthews', marker='o')
    #plt.plot(fd['threshold'], fd['fb_score'], label='F-Beta Score', marker='o')
    #####################################################

    # Segundo modo de plotar os graficos
    lista_colunas = list(fd.columns) # Definindo colunas para o plot
    for col in ['threshold', 'tn', 'fp', 'fn', 'tp']:
        if col in lista_colunas:
            lista_colunas.remove(col)

    for medida in lista_colunas:
        plt.plot(fd['threshold'], fd[medida], label=medida, marker='o')
    ####################################################################################
    
    # Configurações do gráfico
    plt.title('Métricas de Avaliação do Modelo por threshold')
    plt.xlabel('threshold (Pontos de Corte)')
    plt.ylabel('Valor da Métrica')
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
    plt.grid(True)
    plt.tight_layout()

    # Exibindo o gráfico
    plt.show()

# Exemplo:
# tabela = aval_modelo_corte_tabela(x_teste, y_teste)
# aval_modelo_corte_grafico(tabela[['threshold','especificidade','sensibilidade']])

########################################################################################################################################################

# Proposta da função # Exemplo
#fd = pd.DataFrame({'y': y_teste})
#fd['xgb'] = modelo_xgb.predict_proba(x_teste)[:, 1]*100
#fd['xgb_balanceado'] = modelo_xgb_balanceado.predict_proba(x_teste)[:, 1]*100
#fd['lr'] = modelo_lr.predict_proba(x_teste)[:, 1]*100

# Função
def adicionar_previsoes(x_teste, y_teste, **modelos):
    # Inicializa o DataFrame com a coluna `y`
    df = pd.DataFrame({'y': y_teste})

    # Para cada modelo, gera previsões e adiciona ao DataFrame
    for nome, modelo in modelos.items():
        df[nome] = modelo.predict_proba(x_teste)[:, 1] * 100

    return df

# Exemplo de uso:
# Assumindo que `y_teste` e `x_teste` são seus dados, e `modelo_xgb`, `modelo_xgb_balanceado`, `modelo_lr` são seus modelos.
#df_previsoes = adicionar_previsoes(y_teste=y_teste, x_teste=x_teste,
#    xgb=modelo_xgb,
#    xgb_balanceado=modelo_xgb_balanceado,
#    lr=modelo_lr
#)
#df_previsoes.sample(5)


########################################################################################################################################################


import numpy as np
import itertools

def gridsearch_mult_models_threshold(fd, y='y', beta = 1, linspace_thresholds = list(range(10, 100, 10))):

    y_teste = fd[y]
    colunas_proba_modelos = fd.drop(y, axis=1).columns

    combinacoes = list(itertools.product(linspace_thresholds, repeat = len(colunas_proba_modelos) ))

    #######################################################################################
    tabela = pd.DataFrame(columns=['threshold', 'tn', 'fp', 'fn', 'tp'])

    for i in range(len(combinacoes)):
        #print(combinacoes[i])
        
        soma_classificacoes = np.zeros(len(fd))
        count = 0
        for proba_modelos in colunas_proba_modelos:
            soma_classificacoes += np.where(fd[proba_modelos] >= combinacoes[i][count], 1, 0)
            count =+ 1

        maioria = int(count/2+1)
        # classificacao_final
        previsoes_personalizadas = np.where(soma_classificacoes >= maioria, 1, 0)
        #######################################################################################

        threshold = str(combinacoes[i])

        # Calculando os valores de tn, fp, fn, tp
        tn = len(np.where((previsoes_personalizadas == 0) & (y_teste == 0))[0])
        fp = len(np.where((previsoes_personalizadas == 1) & (y_teste == 0))[0])
        fn = len(np.where((previsoes_personalizadas == 0) & (y_teste == 1))[0])
        tp = len(np.where((previsoes_personalizadas == 1) & (y_teste == 1))[0])

        # Adicionando os resultados ao DataFrame 'fd'
        tabela = pd.concat([tabela, pd.DataFrame([[threshold, tn, fp, fn, tp]], columns=tabela.columns)])
    tabela = tabela.reset_index(drop=True)

    ######################################################################################
    # Inicializar as colunas das métricas com NaN
    metricas_de_aval = ['acuracia', 'precisao', 'sensibilidade', 'especificidade', 'f1',
                        'valor_pre_posi', 'valor_pre_neg', 'taxa_falsos_positivos', 'taxa_falsos_negativos',
                        'fdr', 'fo_r', 'indice_youden', 'coef_matthews', 'fb_score']
    tabela[metricas_de_aval] = np.nan

    # Loop para calcular as métricas em cada linha
    for l in range(len(tabela)):
        # Calcular Acurácia
        if (tabela['tn'][l] + tabela['fp'][l] + tabela['fn'][l] + tabela['tp'][l]) != 0:
            tabela['acuracia'][l] = (tabela['tp'][l] + tabela['tn'][l]) / (tabela['tn'][l] + tabela['fp'][l] + tabela['fn'][l] + tabela['tp'][l])
        else:
            tabela['acuracia'][l] = 0

        # Calcular Precisão
        if (tabela['tp'][l] + tabela['fp'][l]) != 0:
            tabela['precisao'][l] = tabela['tp'][l] / (tabela['tp'][l] + tabela['fp'][l])
        else:
            tabela['precisao'][l] = 0

        # Calcular Sensibilidade (Recall)
        if (tabela['tp'][l] + tabela['fn'][l]) != 0:
            tabela['sensibilidade'][l] = tabela['tp'][l] / (tabela['tp'][l] + tabela['fn'][l])
        else:
            tabela['sensibilidade'][l] = 0

        # Calcular Especificidade
        if (tabela['tn'][l] + tabela['fp'][l]) != 0:
            tabela['especificidade'][l] = tabela['tn'][l] / (tabela['tn'][l] + tabela['fp'][l])
        else:
            tabela['especificidade'][l] = 0

        # Calcular F1-Score
        if (tabela['precisao'][l] + tabela['sensibilidade'][l]) != 0:
            tabela['f1'][l] = 2 * (tabela['precisao'][l] * tabela['sensibilidade'][l]) / (tabela['precisao'][l] + tabela['sensibilidade'][l])
        else:
            tabela['f1'][l] = 0

        # Calcular Valor Preditivo Positivo
        if (tabela['tp'][l] + tabela['fp'][l]) != 0:
            tabela['valor_pre_posi'][l] = tabela['tp'][l] / (tabela['tp'][l] + tabela['fp'][l])
        else:
            tabela['valor_pre_posi'][l] = 0

        # Calcular Valor Preditivo Negativo
        if (tabela['tn'][l] + tabela['fn'][l]) != 0:
            tabela['valor_pre_neg'][l] = tabela['tn'][l] / (tabela['tn'][l] + tabela['fn'][l])
        else:
            tabela['valor_pre_neg'][l] = 0

        # Calcular Taxa de Falsos Positivos
        tabela['taxa_falsos_positivos'][l] = 1 - tabela['especificidade'][l]

        # Calcular Taxa de Falsos Negativos
        tabela['taxa_falsos_negativos'][l] = 1 - tabela['sensibilidade'][l]

        # Calcular False Discovery Rate (FDR)
        if (tabela['tp'][l] + tabela['fp'][l]) != 0:
            tabela['fdr'][l] = tabela['fp'][l] / (tabela['tp'][l] + tabela['fp'][l])
        else:
            tabela['fdr'][l] = 0

        # Calcular False Omission Rate (FOR)
        if (tabela['tn'][l] + tabela['fn'][l]) != 0:
            tabela['fo_r'][l] = tabela['fn'][l] / (tabela['tn'][l] + tabela['fn'][l])
        else:
            tabela['fo_r'][l] = 0

        # Calcular Índice de Youden
        tabela['indice_youden'][l] = tabela['sensibilidade'][l] + tabela['especificidade'][l] - 1

        # Calcular Coeficiente de Matthews (MCC)
        denom_matthews = (tabela['tp'][l]+tabela['fp'][l])*(tabela['tp'][l]+tabela['fn'][l])*(tabela['tn'][l]+tabela['fp'][l])*(tabela['tn'][l]+tabela['fn'][l])
        if denom_matthews != 0:
            tabela['coef_matthews'][l] = (tabela['tp'][l]*tabela['tn'][l] - tabela['fp'][l]*tabela['fn'][l]) / np.sqrt(denom_matthews)
        else:
            tabela['coef_matthews'][l] = 0

        # Calcular F-beta Score
        denom_fb_score = (beta**2 * tabela['valor_pre_posi'][l]) + tabela['sensibilidade'][l]
        if denom_fb_score != 0:
            tabela['fb_score'][l] = (1 + beta**2) * (tabela['valor_pre_posi'][l] * tabela['sensibilidade'][l]) / denom_fb_score
        else:
            tabela['fb_score'][l] = 0
        ######################################################################################
    # Coluna final com os thresholds
    tabela = pd.concat([tabela, pd.DataFrame(combinacoes)], axis=1)

    return tabela


#df_previsoes = adicionar_previsoes(y_teste=y_teste, x_teste=x_teste,
#    xgb=modelo_xgb, xgb_balanceado=modelo_xgb_balanceado, lr=modelo_lr)
#grade_thresholds = gridsearch_mult_models_threshold(df_previsoes, linspace_thresholds = list(range(5, 100, 5)))
#grade_thresholds

########################################################################################################################################################