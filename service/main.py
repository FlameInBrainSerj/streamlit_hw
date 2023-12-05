import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

PATH_TO_DATA = Path().cwd().parent / "datasets" / "final_dataset.csv"


# Графики распределений признаков
@st.cache_data
def plot_dists(data):
    st.subheader("Графики распределений признаков и таргета")
    fig, axs = plt.subplots(10, 1, figsize=(13, 50))

    for i, column in enumerate(data.columns):
        if column in ["TARGET", "SOCSTATUS_WORK_FL", "SOCSTATUS_PENS_FL", "GENDER"]:
            axs[i].set_xticks([0, 1])
            sns.histplot(data[column], ax=axs[i])

        if column in ["CHILD_TOTAL", "DEPENDANTS", "LOAN_NUM_TOTAL", "LOAN_NUM_CLOSED"]:
            axs[i].set_xticks(np.arange(data[column].min(), data[column].max() + 1, 1))
            sns.histplot(data[column], ax=axs[i])

        if column in ["AGE"]:
            axs[i].set_xticks(np.arange(data[column].min(), data[column].max() + 1, 5))
            sns.histplot(data[column], ax=axs[i])

        if column in ["PERSONAL_INCOME"]:
            axs[i].set_xticks(np.arange(0, data[column].max() + 1, 50000))
            sns.histplot(data[column], ax=axs[i])

    plt.tight_layout()
    st.pyplot(fig)

    st.write(
        """
        **Наблюдения:**    
        1. У нас несбалансированная выборка по таргету 
        (по большинству наблюдений отклика не было)  
        2. У нас более-менее хорошо представлены все возрастные категории  
        3. Наибольшая часть клиентов работает  
        4. Наибольшая часть клиентов не является пенсионерами  
        5. Наибольшая часть клиентов - мужчины  
        6. Наибольшая часть клиентов имеет не больше 3-х детей   
        7. Наибольшая часть клииентов имеет на иждивении не больше 2-х людей  
        8. Доход имеет лог-нормальное распределение   
        9. Наибольшая часть клиенттов имеет не больше 3-х кредитов  
        10. Большое количество клиентов имеет незакрытые долги  
        """
    )
    st.divider()


# Extra (pairplot + что-нибудь еще)
@st.cache_data
def plot_pairplot(data):
    st.subheader("Попарные распределения")
    fig = sns.pairplot(data)
    st.pyplot(fig)
    st.divider()


# Матрицы корреляций признаков
@st.cache_data
def plot_corr_matrix(data):
    st.subheader("Матрица корреляций")
    corr_matrix = data.corr()
    fig = plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
    st.pyplot(fig)
    st.write(
        """
        **Замечание:** Стоит отметить, что не совсем корректно смотреть 
        на коэффициент корреляции Пирсона между категориальными признаками.
        Если есть необходимость сравнить категориальные признаки друг с другом, то
        лучше использовать тест хи-квадрат или его аналоги, а если есть необходимость
        сравнить категориальные признаки с численными, то тест ANOVA.  
        
        **Наблюдения:** У нас же здесь почти все признаки, кроме AGE и PERSONAL_INCOME, являются
        категориальными (некоторые, такие как DEPENDANTS или CHILD_TOTAL, 
        можно отнести к численным, но я сторонник считать их категориальными), поэтому
        большого смысла в этом графике нет
        """
    )
    st.divider()


# Графики зависимостей целевой переменной и признаков
@st.cache_data
def plot_dependencies(data):
    st.subheader("Графики зависимостей таргета и признаков")
    fig, axs = plt.subplots(9, 1, figsize=(13, 50))

    for i, column in enumerate(data.drop(["TARGET"], axis=1).columns):
        if column in ["AGE", "PERSONAL_INCOME"]:
            sns.boxplot(y="TARGET", x=column, data=data, orient="h", ax=axs[i])

        if column in ["SOCSTATUS_WORK_FL", "SOCSTATUS_PENS_FL", "GENDER"]:
            sns.barplot(x="TARGET", data=data, hue=column, ax=axs[i])
            axs[i].set(ylabel=column, xlabel="Mean value of target")

        if column in ["CHILD_TOTAL", "DEPENDANTS", "LOAN_NUM_TOTAL", "LOAN_NUM_CLOSED"]:
            sns.barplot(y="TARGET", x=column, data=data, ax=axs[i])
            axs[i].set(xlabel=column, ylabel="Mean value of target")

    st.pyplot(fig)
    st.write(
        """
        **Замечание:** Все утверждения ниже не имеют никакой 
        статистической значимости
        
        **Наблюдения:**    
        1. Судя по графику, в нашей выборке люди помоложе с большей 
        вероятностью откликаются на маркетинговую кампанию  
        2. Судя по графику, в нашей выборке работающие люди с большей
        вероятностью откликаются на маркетинговую кампанию  
        3. Судя по графику, в нашей выборке пенсионеры с меньшей
        вероятностью откликаются на маркетинговую кампанию  
        4. Судя по графику, в нашей выборке мужчины с меньшей
        вероятностью откликаются на маркетинговую кампанию
        5. Судя по графику, в нашей выборке чем больше детей у клиента,
        тем больше вероятность отклика на маркетинговую кампанию (примерно)
        6. Судя по графику, в нашей выборке чем больше иждивенцев у клиента, 
        тем больше вероятность отклика на маркетинговую кампанию (примерно)
        7. Судя по графику, в нашей выборке чем больше доход у клиента, тем
        больше вероятность отклика на маркетинговую кампанию
        8. Судя по графику, в нашей выборке чем больше у клиента кредитов, тем
        меньше вероятность его отклика на маркетинговую кампанию (очень слабая зависимость)
        9. Судя по графику, в нашей выборке чем больше у клиента закрытых кредитов,
        тем меньше его вероятность отклика на маркетинговую кампанию (очень слабая зависимость)
        """
    )
    st.divider()


# Числовые характеристики числовых столбцов
@st.cache_data
def plot_descr_stat(data):
    st.subheader("Числовые характеристики числовых столбцов")
    st.table(data.describe())
    st.divider()


def load_page():
    """loads main page"""
    st.set_page_config(layout="centered", page_title="EDA-банк", page_icon=":bank:")

    df = pd.read_csv(PATH_TO_DATA).drop(["AGREEMENT_RK"], axis=1)
    st.title("EDA данных о клиентах банка")
    st.button("Donate, if you liked this <3", on_click=lambda: st.success('Спасибо, теперь все ваши деньги - мои :)'))
    plot_dists(df)
    plot_pairplot(df)
    plot_corr_matrix(df)
    plot_dependencies(df)
    plot_descr_stat(df)

if __name__ == "__main__":
    load_page()
