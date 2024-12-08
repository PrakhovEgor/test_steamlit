import pandas as pd
import streamlit as st
from PIL import Image
from model import model_prediction
import json
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats


def process_main_page():
    st.set_page_config(
        layout="wide",
        initial_sidebar_state="auto",
        page_title="Car Price Prediction",
        page_icon=Image.open('data/car.jpg'),
    )
    page = st.sidebar.selectbox("Выберите страницу", ["Анализ данных", "Предсказание стоимости"])

    if page == "Анализ данных":
        show_main_page()
    elif page == "Предсказание стоимости":
        show_prediction_page()


def show_main_page():
    st.write(
        """
        # Это страница анализа данных
        Используйте боковую панель, чтобы перейти к вводу данных и предсказаниям.
        """
    )

    st.write("### Исходный датасет")
    df = pd.read_csv('data/data.csv')
    st.write(df.head(5))

    with st.expander("Описание данных", expanded=False):
        st.markdown(
            """
            **Целевая переменная**
            - `selling_price`: цена продажи, числовая

            **Признаки**
            - **`name`** *(string)*: модель автомобиля
            - **`year`** *(numeric, int)*: год выпуска с завода-изготовителя
            - **`km_driven`** *(numeric, int)*: пробег на дату продажи
            - **`fuel`** *(categorical: _Diesel_, _Petrol_, _CNG_, _LPG_, _electric_)*: тип топлива
            - **`seller_type`** *(categorical: _Individual_, _Dealer_, _Trustmark Dealer_)*: продавец
            - **`transmission`** *(categorical: _Manual_, _Automatic_)*: тип трансмиссии
            - **`owner`** *(categorical: _First Owner_, _Second Owner_, _Third Owner_, _Fourth & Above Owner_)*: какой по счёту хозяин?
            - **`mileage`** *(string, по смыслу числовой)*: пробег, требует предобработки
            - **`engine`** *(string, по смыслу числовой)*: рабочий объем двигателя, требует предобработки
            - **`max_power`** *(string, по смыслу числовой)*: пиковая мощность двигателя, требует предобработки
            - **`torque`** *(string, по смыслу числовой, а то и 2)*: крутящий момент, требует предобработки
            - **`seats`** *(numeric, float; по смыслу categorical, int)*: количество мест
            """
        )

    with st.expander("Проверка на NaN", expanded=True):
        nan_summary = df.isnull().sum()
        if nan_summary.sum() == 0:
            st.success("В данных нет пропущенных значений!")
        else:
            st.warning("В данных есть пропущенные значения!")
            st.write("Количество пропущенных значений по столбцам:")
            st.write(nan_summary[nan_summary > 0])

    st.write("### Датасет, после некоторых преобразований")
    st.write("name -> Company, Model; Mileage transform; Engine transform; Max_power transform;")
    df_v1 = pd.read_csv('data/data_preproc_v1.csv')
    st.write(df_v1.head(5))

    with st.expander("EDA", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            st.write("### Корреляционная матрица числовых признаков")
            corr_matrix = df_v1.select_dtypes(include=['int64', 'float64']).corr()
            f, ax = plt.subplots(figsize=(6, 6))
            sns.heatmap(
                corr_matrix,
                annot=True,
                linewidths=0.5,
                linecolor="red",
                fmt=".4f",
                cmap="coolwarm",
                ax=ax
            )
            st.pyplot(f)
            st.write(
                "Из диаграммы видно, что больше всего коррелируют переменные `engine` и `max_power`, "
                "что логично, так как пиковая мощность двигателя напрямую зависит от его объема."
            )

            st.write("### Зависимость цены продажи от пробега")
            plt.figure(figsize=(8, 6))
            sns.scatterplot(x='km_driven', y='selling_price', data=df_v1)
            plt.title('Зависимость цены продажи от пробега автомобиля')
            plt.xlabel('Пробег (km)')
            plt.ylabel('Цена продажи (selling_price)')
            st.pyplot(plt)
            st.write("С увеличением пробега цена продажи автомобиля обычно снижается")



            st.write("### Отношение видов топлива")
            types_fuel = df_v1["fuel"].unique()
            x_fuel = df_v1["fuel"].value_counts()
            f, ax = plt.subplots(figsize=(5, 5))
            ax.pie(x_fuel, labels=types_fuel, autopct="%1.1f%%", startangle=90)
            ax.set_title("Отношение видов топлива")
            st.pyplot(f)

        with col2:
            st.write("### Отношение видов продавцов")
            types = df_v1["seller_type"].unique()
            x = df_v1["seller_type"].value_counts()
            f, ax = plt.subplots(figsize=(5, 5))
            ax.pie(x, labels=types, autopct="%1.1f%%", startangle=90)
            ax.set_title("Отношение видов продавцов")
            st.pyplot(f)
            st.write(
                "Видно, что подержанные авто чаще продают индивидуальные лица"
            )

            st.write("### Количество записей по компаниям")
            companies = df_v1["Company"].unique()
            count = df_v1["Company"].value_counts()
            f, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(x=companies, y=count.values, ax=ax)
            ax.set_xlabel("Название компании")
            ax.set_ylabel("Количество")
            ax.set_xticklabels(companies, rotation=75)
            st.pyplot(f)

            st.write("### Отношение видов трансмиссии")
            types_transmission = df_v1["transmission"].unique()
            x_transmission = df_v1["transmission"].value_counts()
            f, ax = plt.subplots(figsize=(5, 5))
            ax.pie(x_transmission, labels=types_transmission, autopct="%1.1f%%", startangle=90)
            ax.set_title("Отношение видов трансмиссии")
            st.pyplot(f)

            st.write("Большинство автомобилей имеют механическую трансмиссию, что составляет 87.1% от общего числа. Это может указывать на более широкое распространение механических коробок передач, особенно на вторичном рынке.")

        st.write("### Сравнение распределений цен для разных типов топлива")
        plt.figure(figsize=(10, 5))
        sns.boxplot(x='fuel', y='selling_price', data=df_v1)
        plt.title("Распределение цены продажи для разных типов топлива")
        plt.xlabel("Тип топлива")
        plt.ylabel("Цена продажи")
        st.pyplot(plt)
        st.write("Из графика видно, что автомобили с дизельным топливом имеют широкий диапазон цен. Большинство автомобилей находится в диапазоне цен около 0.2")

        # 2. Сравнение распределений цен для различных типов трансмиссий
        st.write("### Сравнение распределений цен для разных типов трансмиссий")
        plt.figure(figsize=(10, 5))
        sns.boxplot(x='transmission', y='selling_price', data=df_v1)
        plt.title("Распределение цены продажи для разных типов трансмиссий")
        plt.xlabel("Тип трансмиссии")
        plt.ylabel("Цена продажи")
        st.pyplot(plt)
        st.write("По графику видно, что машины с автоматической коробкой передач продаются значительно дороже, чем машины с механикой")

    with st.expander("Гипотеза", expanded=True):
        st.write("### Гипотеза: Бензиновые автомобили с автоматической коробкой передач, продаваемые дилером, имеют более высокую цену, чем те, которые продаются частными лицами")

        group_dealer = df_v1[(df_v1['fuel'] == 'Petrol') & (df_v1['transmission'] == 'Automatic') & (df_v1['seller_type'] == 'Dealer')][
            'selling_price']
        group_individual = df_v1[(df_v1['fuel'] == 'Petrol') & (df_v1['transmission'] == 'Automatic') & (df_v1['seller_type'] == 'Individual')][
            'selling_price']

        # 2. Сравнение средних значений цены
        st.write("#### Сравнение средней цены продажи для двух групп: дилер vs частное лицо")
        plt.figure(figsize=(10, 5))
        sns.boxplot(x='seller_type', y='selling_price',
                    data=df_v1[(df_v1['fuel'] == 'Petrol') & (df_v1['transmission'] == 'Automatic')])
        plt.title("Распределение цен продажи для бензиновых автомобилей с автоматической коробкой передач")
        plt.xlabel("Тип продавца")
        plt.ylabel("Цена продажи")
        st.pyplot(plt)

        t_stat, p_value = stats.ttest_ind(group_dealer, group_individual, equal_var=False)

        st.write(f"t-тест: t-статистика = {t_stat:.2f}, p-значение = {p_value:.4f}")

        # 4. Интерпретация результатов
        if p_value < 0.05:
            st.success(
                "Гипотеза подтверждается: бензиновые автомобили с автоматической коробкой передач, продаваемые дилером, имеют более высокую цену, чем те, которые продаются частными лицами.")
        else:
            st.warning("Гипотеза опровергается: статистически значимых различий в цене продажи между группами нет.")


    st.write("### Датасет для обучения модели")
    st.write("[owner, fuel, seller_type, transmission, name] -> to digits\n")
    df_v2 = pd.read_csv('data/data_preproc_v2.csv')
    st.write(df_v2.head(5))

def show_prediction_page():
    st.header("Введите параметры автомобиля")
    user_input_df = main_page_input_features()

    if st.button("Сделать предсказание"):
        prediction = model_prediction(user_input_df)
        write_user_data(user_input_df)
        write_prediction(prediction)


def main_page_input_features():
    col1, col2 = st.columns(2)

    with col1:
        name = st.text_input("Name")
        year = st.slider("Year of manufacture", min_value=1950, max_value=2023, value=2015, step=1)
        km_driven = st.slider("km driven", min_value=0, max_value=1000000, value=50000, step=1)
        fuel = st.selectbox("Fuel type", ("Diesel", "Petrol", "LPG", "CNG"))
        seller_type = st.selectbox("Seller type", ("Individual", "Dealer", "Trustmark Dealer"))

    with col2:
        transmission = st.selectbox("Transmission", ("Automatic", "Manual"))
        owner = st.selectbox("Owner", ("First", "Second", "Third", "Fourth & Above", "Test Drive car"))
        mileage = st.slider("Mileage, kmpl", min_value=0, max_value=50, value=18, step=1)
        engine = st.slider("Engine, CC", min_value=500, max_value=3600, value=1000, step=1)
        max_power = st.slider("Max power, bhp", min_value=0, max_value=14, value=7, step=1)
        seats = st.slider("Seats", min_value=2, max_value=14, value=5, step=1)

    translatetion = {
        "Diesel": 1,
        "Petrol": 2,
        "LPG": 3,
        "CNG": 4,
        "Individual": 1,
        "Dealer": 2,
        "Trustmark Dealer": 3,
        "Manual": 1,
        "Automatic": 2,
        "First": 1,
        "Second": 2,
        "Third": 3,
        "Fourth & Above": 4,
        "Test Drive car": 5,
    }

    data = {
        "name": prep_name(name),
        "year": year,
        "km_driven": km_driven,
        "fuel": translatetion[fuel],
        "seller_type": translatetion[seller_type],
        "transmission": translatetion[transmission],
        "owner": translatetion[owner],
        "mileage": mileage,
        "engine": engine,
        "max_power": max_power,
        "seats": seats,
    }

    df = pd.DataFrame(data, index=[0])
    return df


def write_user_data(df):
    st.write("## Ваши данные")
    st.write(df)


def write_prediction(prediction):
    st.write("## Предсказание")
    st.write(prediction)


def prep_name(name):
    with open('models.json', 'r') as json_file:
        d = json.load(json_file)
    for i in d.keys():
        if name.lower().strip() in i.lower():
            return d[i]
    else:
        return 0


if __name__ == "__main__":
    process_main_page()
