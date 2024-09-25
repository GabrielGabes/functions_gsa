# Função para avaliação de modelos exibindo metricas de avaliação
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, RocCurveDisplay

def avaliar_modelo(y_verdadeiro, y_predito, conjunto_nome="", plotar_grafico=False):
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

import pandas as pd
import numpy as np

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
########################################################################################################################################################

import matplotlib.pyplot as plt
from matplotlib import style
plt.style.use('ggplot')

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