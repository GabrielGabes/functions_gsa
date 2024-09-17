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