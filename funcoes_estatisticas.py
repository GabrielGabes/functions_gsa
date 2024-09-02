#############################################################################################################

def freq_table0(df, coluna):
    freq = df[coluna].value_counts()
    perc = df[coluna].value_counts(normalize = True)*100
    tabela = pd.DataFrame({"Freq":freq, "Perc":perc})
    return display(tabela)

def count_table0(df, linha, coluna):
    a = df.groupby([linha, coluna]).size().unstack()
    for i in a.columns:
        a[i].fillna(0, inplace = True)
        a[i] = a[i].apply(lambda x: int(x))
    return a
    print('='*100)

def count_table_percent(df, linha, coluna, opcao):
    result = tabela_cont(df, linha, coluna)
    if opcao == 'Linha':
        return round(result.div(result.sum(axis=1), axis=0) * 100, 2)
    elif opcao == 'Coluna':
        return round(result.div(result.sum(), axis=1) * 100, 2)
    else:
        return 'Opção invalida'

##########################################################################################################################################################################################################################

def pval_string(valor):
    # Formatar o valor com 6 dígitos decimais
    valor_str = f"{valor:.6f}"
    
    if valor < 0.05:
        if valor < 0.001 or valor == 0:
            return "< 0.001"
        elif valor < 0.01:
            return valor_str[:5]  # Retorna até 4 dígitos decimais
        else:
            return valor_str[:4]  # Retorna até 3 dígitos decimais
    else:
        if valor < 0.06:
            return valor_str[:5]  # Retorna até 4 dígitos decimais
        else:
            return valor_str[:4]  # Retorna até 3 dígitos decimais

####################################

def num_string(valor, digitos):
    if abs(valor) < 0.01 and abs(valor) > 0.0009:
        return "{:.3f}".format(valor)
    elif valor == 0:
        return "0.00"
    elif abs(valor) <= 0.0009 and valor != 0 or abs(valor) > 1000:
        return "{:.{prec}e}".format(valor, prec=digitos)
    else:
        return "{:.{prec}f}".format(valor, prec=digitos)


def freq_table(df, variavel):
    # Realiza a contagem dos valores únicos, excluindo NaNs por padrão.
    contagem = df[variavel].value_counts(dropna=False, normalize=True)
    
    # Converte a contagem para porcentagem e formata os números.
    contagem_porcentagem = contagem * 100
    
    # Cria um DataFrame para conter os resultados.
    resultado = pd.DataFrame({
        'Variable': contagem.index,
        'Contagem': contagem.values,
        'Porcentagem': contagem_porcentagem.values
    })
    
    # Formata a coluna 'Porcentagem' para exibir duas casas decimais.
    resultado['Porcentagem'] = resultado['Porcentagem'].apply(lambda x: f"{x:.2f}%")
    
    return resultado

###################################################################################################

from scipy.stats import chi2_contingency, fisher_exact
from statsmodels.stats.proportion import proportions_chisquare

def fisher_criterio(df, var1, var2):
    # Converter para categorias e obter a quantidade de níveis únicos
    length1 = df[var1].nunique()
    length2 = df[var2].nunique()
    
    # Verificar o critério inicial baseado na quantidade de níveis
    if length1 >= 3 or length2 >= 3:
        return False
    else:
        # Criar tabela de contingência
        tabela = pd.crosstab(df[var1], df[var2])
        
        # Calcular as expectativas de frequência usando o teste qui-quadrado
        # (mas não estamos interessados nos valores do teste em si, apenas nas frequências esperadas)
        _, _, _, expected_frequencies = chi2_contingency(tabela, correction=False)
        
        # Verificar se alguma expectativa de frequência é < 5
        return np.any(expected_frequencies < 5)

###########################################################

def count_table(df, var_y, var_x, sentido_percent='col', apenas_fisher=False):
    # Criação da tabela de contingência
    tabela_contingencia = pd.crosstab(df[var_x], df[var_y], margins=True, margins_name="Total")
    
    # Calcula as porcentagens
    if sentido_percent == 'col':
        percentuais = tabela_contingencia.div(tabela_contingencia.loc["Total"], axis=1)
    else:
        percentuais = tabela_contingencia.div(tabela_contingencia["Total"], axis=0)
    
    percentuais = percentuais[:-1] * 100  # Exclui a linha de totais para percentuais
    
    # Teste estatístico
    if apenas_fisher or tabela_contingencia.shape[0] <= 2 or tabela_contingencia.shape[1] <= 2:
        # Teste exato de Fisher para 2x2
        p_value = fisher_exact(tabela_contingencia.iloc[:-1, :-1])[1]
        test_used = "Fisher Exact"
    else:
        # Teste Qui-quadrado
        p_value = chi2_contingency(tabela_contingencia.iloc[:-1, :-1])[1]
        test_used = "Chi-squared"
    
    # Inserção dos resultados dos testes e formatação final
    percentuais['P-value'] = np.nan
    percentuais['P-value'].iloc[0] = pval_string(p_value)
    percentuais['Test_Used'] = np.nan
    percentuais['Test_Used'].iloc[0] = test_used
    
    # Ajuste final das colunas se necessário
    # Removendo símbolos e ajustando espaços
    percentuais = percentuais.applymap(lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x)
    percentuais = percentuais.applymap(lambda x: x.replace("%", "").strip())
    
    return percentuais

##########################################################################################################################################################################################################################

def summary_num_parametric(df, col_num):
    # Filtrar NA e calcular média e desvio padrão
    valid_data = df[col_num].dropna()
    mean = round(valid_data.mean(), 2)
    std = round(valid_data.std(), 2)
    
    # Criar DataFrame de resultado
    tabela = pd.DataFrame({
        "Variable": [col_num],
        "Mean": [mean],
        "Std": [f"± {std}"]
    })
    return tabela
    
###################################################################################################

from scipy.stats import ttest_ind, f_oneway
from statsmodels.stats.power import TTestIndPower

def summary_num_parametric_groups(df, col_num, col_cat):
    # Filtrar NA
    df_filter = df.dropna(subset=[col_num, col_cat])
    
    # Sumário por grupo e geral
    group_summary = df_filter.groupby(col_cat)[col_num].agg(['mean', 'std']).reset_index()
    group_summary['resumo'] = group_summary.apply(lambda x: f"{x['mean']:.2f} ± {x['std']:.2f}", axis=1)
    
    overall_mean = round(df_filter[col_num].mean(), 2)
    overall_std = round(df_filter[col_num].std(), 2)
    overall_summary = pd.DataFrame({
        col_cat: ['Total'],
        'resumo': [f"{overall_mean:.2f} ± {overall_std:.2f}"]
    })
    
    # Combinar os sumários
    final_summary = pd.concat([overall_summary, group_summary[[col_cat, 'resumo']]], ignore_index=True)
    
    # Teste estatístico
    unique_categories = df_filter[col_cat].unique()
    if len(unique_categories) <= 2:
        # T-test para 2 grupos
        group1 = df_filter[df_filter[col_cat] == unique_categories[0]][col_num]
        group2 = df_filter[df_filter[col_cat] == unique_categories[1]][col_num]
        t_stat, p_value = ttest_ind(group1, group2)
        test_used = "T Test"
    else:
        # ANOVA para mais de 2 grupos
        groups = [df_filter[df_filter[col_cat] == category][col_num] for category in unique_categories]
        f_stat, p_value = f_oneway(*groups)
        test_used = "ANOVA"
    
    final_summary['P-value'] = np.nan
    final_summary['P-value'].iloc[0] = pval_string(p_value)
    final_summary['Test_Used'] = np.nan
    final_summary['Test_Used'].iloc[0] = test_used
    
    return final_summary

####################################################################################################################################

def summary_num_nonparametric(df, col_num):
    # Filtrar NA e calcular mediana, Q1 e Q3
    valid_data = df[col_num].dropna()
    median = round(valid_data.median(), 2)
    q1 = round(valid_data.quantile(0.25), 2)
    q3 = round(valid_data.quantile(0.75), 2)
    
    # Criar DataFrame de resultado
    tabela = pd.DataFrame({
        "Variable": [col_num],
        "Median": [median],
        "Q1 - Q3": [f"[{q1} - {q3}]"]
    })
    return tabela

###################################################################################################

from scipy.stats import kruskal, mannwhitneyu

def summary_num_nonparametric_groups(df, col_num, col_cat):
    # Filtrar NA
    df_filter = df.dropna(subset=[col_num, col_cat])
    groups = df_filter[col_cat].unique()
    
    # Preparar sumário por grupo
    summaries = []
    for group in groups:
        group_data = df_filter[df_filter[col_cat] == group][col_num]
        median = round(group_data.median(), 2)
        q1 = round(group_data.quantile(0.25), 2)
        q3 = round(group_data.quantile(0.75), 2)
        summaries.append((group, f"{median} [{q1} - {q3}]"))
    
    # Sumário geral
    overall_data = df_filter[col_num]
    overall_median = round(overall_data.median(), 2)
    overall_q1 = round(overall_data.quantile(0.25), 2)
    overall_q3 = round(overall_data.quantile(0.75), 2)
    summaries.insert(0, ('Total', f"{overall_median} [{overall_q1} - {overall_q3}]"))
    
    # Conversão para DataFrame
    summary_df = pd.DataFrame(summaries, columns=[col_cat, 'Resumo'])
    
    # Teste estatístico
    if len(groups) > 2:
        # Kruskal-Wallis Test
        group_data = [df_filter[df_filter[col_cat] == group][col_num] for group in groups]
        stat, p_value = kruskal(*group_data)
        test_used = "Kruskal-Wallis"
    else:
        # Mann-Whitney U Test
        group1, group2 = groups
        stat, p_value = mannwhitneyu(df_filter[df_filter[col_cat] == group1][col_num],
                                     df_filter[df_filter[col_cat] == group2][col_num])
        test_used = "Mann-Whitney"
    
    # Incluir resultados dos testes
    summary_df['P-value'] = np.nan
    summary_df['P-value'].iloc[0] = pval_string(p_value)
    summary_df['Test_Used'] = np.nan
    summary_df['Test_Used'].iloc[0] = test_used
    
    return summary_df

##########################################################################################################################################################################################################################
# NORMALIDADE

from scipy.stats import shapiro
def group_normality_test(df, col_num, col_cat):
    if len(df) > 5000:
        return False
    for _, group in df.groupby(col_cat):
        # Aplicar o teste de Shapiro-Wilk à coluna numérica do grupo
        stat, p_value = shapiro(group[col_num])
        # Verificar o p-valor para avaliar a normalidade
        if p_value < 0.05:
            # Se algum grupo não for normal, retorna False
            return False
    # Se todos os grupos passarem no teste, retorna True
    return True

####################################################################################################################################

import plotly.express as px # grafico
import scipy.stats as stats # para os testes de hipotese

def num_cat_hip(df, col_cat, col_num):
    grupos = df.groupby(col_cat)[col_num].apply(list)
    
    if len(df[col_cat].unique()) == 2:
        grupo1, grupo2 = grupos
        if group_normality_test(df, col_num, col_cat):
            # teste T de Student
            stat, p_value = stats.ttest_ind(grupo1, grupo2); print('P-value (Teste T):', pval_string(p_value))
        else:
            # teste de Mann-Whitney
            stat, p_value = stats.mannwhitneyu(grupo1, grupo2); print('P-value (Mann-Whitney):', pval_string(p_value))

    elif len(df[col_cat].unique()) > 2:
        if group_normality_test(df, col_num, col_cat):
            # teste ANOVA
            f_value, p_value = stats.f_oneway(*grupos); print('P-value (Anova):', pval_string(p_value))

        else:
            # Kruskal-Wallis
            kruskal_stat, p_value = stats.kruskal(*grupos); print('P-value (Kruskal-Wallis):', pval_string(p_value))

            if p_value < 0.05:
                # Pós Teste de Dunn
                dunn_test = sp.posthoc_dunn(df, val_col=col_num, group_col=col_cat, p_adjust='bonferroni')
                for i in dunn_test.columns:dunn_test[i] = dunn_test[i].apply(lambda x: pval_string(x))
                print('P-value (Pós-Teste de Dunn):')
                display(dunn_test)

    # GRAFICO ####################################################################
    fig = px.violin(df, x=col_cat, y=col_num, color=col_cat, box=True, points="all")
    fig.show()

#############################################################################################################

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

# Histograma + Medidas Estatisticas
import matplotlib.pyplot as plt
def gghist(vetor):
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

#############################################################################################################


#############################################################################################################


#############################################################################################################


#############################################################################################################


#############################################################################################################


#############################################################################################################


#############################################################################################################
































#############################################################################################################


#############################################################################################################



#############################################################################################################


#############################################################################################################


#############################################################################################################