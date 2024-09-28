import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, RocCurveDisplay

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
    print(confusion_matrix(y_verdadeiro, y_teste))
    
    print('*' * 50)
    
    print("Relatório de Classificação:", conjunto_nome,":\n")
    print(classification_report(y_verdadeiro, y_teste))
    print('*' * 70)

    if plotar_grafico == True:
        display(RocCurveDisplay.from_predictions(y_verdadeiro, y_teste, name = conjunto_nome))
########################################################################################################################################################

# Função que cria tabela com todas metricas de avaliação em cada ponto de threshold
def aval_modelo_corte_tabela(x_teste, y_teste, classificador, beta = 1, pontos_de_corte = np.arange(0, 1.1, 0.10)):
        
    previsoes_proba = classificador.predict_proba(x_teste)
    probs_positivas = previsoes_proba[:, 1]

    # Inicializando um DataFrame para armazenar as métricas
    fd = pd.DataFrame(columns=['threshold', 'tn', 'fp', 'fn', 'tp'])

    # Loop pelos pontos de corte
    for threshold in pontos_de_corte:
        previsoes_personalizadas = (probs_positivas >= threshold).astype(int)

        # Calculando os valores de tn, fp, fn, tp
        tn = len(np.where((previsoes_personalizadas == 0) & (y_teste == 0))[0])
        fp = len(np.where((previsoes_personalizadas == 1) & (y_teste == 0))[0])
        
        fn = len(np.where((previsoes_personalizadas == 0) & (y_teste == 1))[0])
        tp = len(np.where((previsoes_personalizadas == 1) & (y_teste == 1))[0])

        # Adicionando os resultados ao DataFrame 'fd'
        fd = pd.concat([fd, pd.DataFrame([[threshold, tn, fp, fn, tp]], columns=fd.columns)])

    # Substituir valores NaN por 0
    fd.fillna(0, inplace=True)
    fd.reset_index(drop=True, inplace=True)
    ######################################################################################

    # Inicializar as colunas das métricas com NaN
    metricas_de_aval = ['acuracia', 'precisao', 'sensibilidade', 'especificidade', 'f1',
                        'valor_pre_posi', 'valor_pre_neg', 'taxa_falsos_positivos', 'taxa_falsos_negativos',
                        'fdr', 'fo_r', 'indice_youden', 'coef_matthews', 'fb_score']
    fd[metricas_de_aval] = 0

    # Loop para calcular as métricas em cada linha
    for i in range(len(fd)):
        # Acurácia
        if (fd['tn'][i] + fd['fp'][i] + fd['fn'][i] + fd['tp'][i]) != 0:
            fd['acuracia'][i] = (fd['tp'][i] + fd['tn'][i]) / (fd['tn'][i] + fd['fp'][i] + fd['fn'][i] + fd['tp'][i])

        # Precisão
        if (fd['tp'][i] + fd['fp'][i]) != 0:
            fd['precisao'][i] = fd['tp'][i] / (fd['tp'][i] + fd['fp'][i])

        # Sensibilidade (Recall)
        if (fd['tp'][i] + fd['fn'][i]) != 0:
            fd['sensibilidade'][i] = fd['tp'][i] / (fd['tp'][i] + fd['fn'][i])

        # Especificidade
        if (fd['tn'][i] + fd['fp'][i]) != 0:
            fd['especificidade'][i] = fd['tn'][i] / (fd['tn'][i] + fd['fp'][i])

        # F1-Score
        if (fd['precisao'][i] + fd['sensibilidade'][i]) != 0:
            fd['f1'][i] = 2 * (fd['precisao'][i] * fd['sensibilidade'][i]) / (fd['precisao'][i] + fd['sensibilidade'][i])

        # Valor Preditivo Positivo
        if (fd['tp'][i] + fd['fp'][i]) != 0:
            fd['valor_pre_posi'][i] = fd['tp'][i] / (fd['tp'][i] + fd['fp'][i])

        # Valor Preditivo Negativo
        if (fd['tn'][i] + fd['fn'][i]) != 0:
            fd['valor_pre_neg'][i] = fd['tn'][i] / (fd['tn'][i] + fd['fn'][i])

        # Taxa de Falsos Positivos
        fd['taxa_falsos_positivos'][i] = 1 - fd['especificidade'][i]

        # Taxa de Falsos Negativos
        fd['taxa_falsos_negativos'][i] = 1 - fd['sensibilidade'][i]

        # False Discovery Rate (FDR)
        if (fd['tp'][i] + fd['fp'][i]) != 0:
            fd['fdr'][i] = fd['fp'][i] / (fd['tp'][i] + fd['fp'][i])

        # False Omission Rate (FOR)
        if (fd['tn'][i] + fd['fn'][i]) != 0:
            fd['fo_r'][i] = fd['fn'][i] / (fd['tn'][i] + fd['fn'][i])

        # Índice de Youden
        fd['indice_youden'][i] = fd['sensibilidade'][i] + fd['especificidade'][i] - 1

        # Coeficiente de Matthews (MCC)
        denom_matthews = (fd['tp'][i]+fd['fp'][i])*(fd['tp'][i]+fd['fn'][i])*(fd['tn'][i]+fd['fp'][i])*(fd['tn'][i]+fd['fn'][i])
        if denom_matthews != 0:
            fd['coef_matthews'][i] = (fd['tp'][i]*fd['tn'][i] - fd['fp'][i]*fd['fn'][i]) / np.sqrt(denom_matthews)

        # F-beta Score
        denom_fb_score = (beta**2 * fd['valor_pre_posi'][i]) + fd['sensibilidade'][i]
        if denom_fb_score != 0:
            fd['fb_score'][i] = (1 + beta**2) * (fd['valor_pre_posi'][i] * fd['sensibilidade'][i]) / denom_fb_score

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
import itertools

def gridsearch_mult_models_threshold(fd, y='y', beta = 1, linspace_thresholds = np.arange(10, 100, 10)):

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

        maioria = int((count/2) + 1)
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
    tabela[metricas_de_aval] = 0

    # Loop para calcular as métricas em cada linha
    for i in range(len(tabela)):
        # Acurácia
        if (tabela['tn'][i] + tabela['fp'][i] + tabela['fn'][i] + tabela['tp'][i]) != 0:
            tabela['acuracia'][i] = (tabela['tp'][i] + tabela['tn'][i]) / (tabela['tn'][i] + tabela['fp'][i] + tabela['fn'][i] + tabela['tp'][i])

        # Precisão
        if (tabela['tp'][i] + tabela['fp'][i]) != 0:
            tabela['precisao'][i] = tabela['tp'][i] / (tabela['tp'][i] + tabela['fp'][i])

        # Sensibilidade (Recall)
        if (tabela['tp'][i] + tabela['fn'][i]) != 0:
            tabela['sensibilidade'][i] = tabela['tp'][i] / (tabela['tp'][i] + tabela['fn'][i])

        # Especificidade
        if (tabela['tn'][i] + tabela['fp'][i]) != 0:
            tabela['especificidade'][i] = tabela['tn'][i] / (tabela['tn'][i] + tabela['fp'][i])

        # F1-Score
        if (tabela['precisao'][i] + tabela['sensibilidade'][i]) != 0:
            tabela['f1'][i] = 2 * (tabela['precisao'][i] * tabela['sensibilidade'][i]) / (tabela['precisao'][i] + tabela['sensibilidade'][i])

        # Valor Preditivo Positivo
        if (tabela['tp'][i] + tabela['fp'][i]) != 0:
            tabela['valor_pre_posi'][i] = tabela['tp'][i] / (tabela['tp'][i] + tabela['fp'][i])

        # Valor Preditivo Negativo
        if (tabela['tn'][i] + tabela['fn'][i]) != 0:
            tabela['valor_pre_neg'][i] = tabela['tn'][i] / (tabela['tn'][i] + tabela['fn'][i])

        # Taxa de Falsos Positivos
        tabela['taxa_falsos_positivos'][i] = 1 - tabela['especificidade'][i]

        # Taxa de Falsos Negativos
        tabela['taxa_falsos_negativos'][i] = 1 - tabela['sensibilidade'][i]

        # False Discovery Rate (FDR)
        if (tabela['tp'][i] + tabela['fp'][i]) != 0:
            tabela['fdr'][i] = tabela['fp'][i] / (tabela['tp'][i] + tabela['fp'][i])

        # False Omission Rate (FOR)
        if (tabela['tn'][i] + tabela['fn'][i]) != 0:
            tabela['fo_r'][i] = tabela['fn'][i] / (tabela['tn'][i] + tabela['fn'][i])

        # Índice de Youden
        tabela['indice_youden'][i] = tabela['sensibilidade'][i] + tabela['especificidade'][i] - 1

        # Coeficiente de Matthews (MCC)
        denom_matthews = (tabela['tp'][i]+tabela['fp'][i])*(tabela['tp'][i]+tabela['fn'][i])*(tabela['tn'][i]+tabela['fp'][i])*(tabela['tn'][i]+tabela['fn'][i])
        if denom_matthews != 0:
            tabela['coef_matthews'][i] = (tabela['tp'][i]*tabela['tn'][i] - tabela['fp'][i]*tabela['fn'][i]) / np.sqrt(denom_matthews)

        # F-beta Score
        denom_fb_score = (beta**2 * tabela['valor_pre_posi'][i]) + tabela['sensibilidade'][i]
        if denom_fb_score != 0:
            tabela['fb_score'][i] = (1 + beta**2) * (tabela['valor_pre_posi'][i] * tabela['sensibilidade'][i]) / denom_fb_score
    
    ######################################################################################
    # Coluna final com os thresholds
    tabela = pd.concat([tabela, pd.DataFrame(combinacoes)], axis=1)

    return tabela


#df_previsoes = adicionar_previsoes(y_teste=y_teste, x_teste=x_teste,
#    xgb=modelo_xgb, xgb_balanceado=modelo_xgb_balanceado, lr=modelo_lr)
#grade_thresholds = gridsearch_mult_models_threshold(df_previsoes, linspace_thresholds = list(range(5, 100, 5)))
#grade_thresholds

########################################################################################################################################################