# Função para avaliação de modelos exibindo metricas de avaliação
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, RocCurveDisplay

def avaliar_modelo(y_verdadeiro, y_predito, conjunto_nome):
    """
    Função para avaliar e imprimir o relatório de classificação e a matriz de confusão.
    """
    #if conjunto_nome == "Treino":
     #   pass
    #else:
    # Metricas
    print('*' * 70)
    print("Relatório de Classificação para o Conjunto de", conjunto_nome,":\n")
    print(classification_report(y_verdadeiro, y_predito))
    
    print('*' * 50)
    
    # Matriz
    print("Matriz de Confusão para o Conjunto de", conjunto_nome,":\n")
    print(confusion_matrix(y_verdadeiro, y_predito))
    print('*' * 70)

    display(RocCurveDisplay.from_predictions(y_verdadeiro, y_predito, name = conjunto_nome))

#############################################################################################################
# Funções para avaliar validação cruzada

# Intervalo de confiança
import numpy as np
def int_conf(vetor):
    media = vetor.mean()
    desvio_padrao = vetor.std()
    ic_0 = round(media - 2*desvio_padrao, 2)
    ic_1 = round(min(media + 2*desvio_padrao, 1), 2)
    ic = '[' + str(ic_0) + ' - ' + str(ic_1) + ']'
    return ic

############################################################

import matplotlib.pyplot as plt
plt.style.use('dark_background')

# Histograma + Medidas Estatisticas da validação cruzada
def avaliar_validacao(vetor):
    plt.figure(figsize=(15,5))
    plt.hist(vetor, edgecolor='black', density=True) #bins=q_bins, 

    plt.title('hist', fontsize=15)
    plt.grid(True, color='gray')

    # Adicionar linhas verticais para média e mediana
    plt.axvline(x = vetor.mean(), color='red', linestyle='--', label='Média')
    plt.axvline(x = vetor.median(), color='blue', linestyle='--', label='Mediana')

    # Adicionar legenda personalizada
    texto_count = 'Count = ' + str(round(len(vetor), 0))
    texto_media = 'Média = '+ str(round(vetor.mean(), 2))
    texto_dp = 'DP = '+ str(round(vetor.std(), 2))
    texto_min = 'Min = '+ str(round(vetor.min(), 2))
    texto_Q1 = 'Q1 = ' + str(round(vetor.quantile(0.25), 2))
    texto_mediana = 'Q2 = '+ str(round(vetor.median(), 2))
    texto_Q3 = 'Q3 = ' + str(round(vetor.quantile(0.75), 2))
    texto_max = 'Max = '+ str(round(vetor.max(), 2))
    ic = 'IC ' + intervalo_conf(vetor)
    texto_legenda = '\n'.join([texto_count, 
                               texto_min,
                               texto_media, texto_dp, 
                               texto_Q1, texto_mediana, texto_Q3,
                               texto_max, 
                               ic])

    plt.text(0.99, 0.96, texto_legenda, ha='right', va='top', transform=plt.gca().transAxes,
             bbox=dict(facecolor='black', edgecolor='gray', boxstyle='round'),
             fontsize=12)

    plt.show()