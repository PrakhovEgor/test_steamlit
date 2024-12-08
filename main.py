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
    page = st.sidebar.selectbox("Select a page", ["Data Analysis", "Price Prediction"])

    if page == "Data Analysis":
        show_main_page()
    elif page == "Price Prediction":
        show_prediction_page()


def show_main_page():
    st.write(
        """
        # This is the data analysis page
        Use the sidebar to navigate to data input and predictions.
        """
    )

    st.write("### Original Dataset")
    df = pd.read_csv('data/data.csv')
    st.write(df.head(5))

    with st.expander("Data Description", expanded=False):
        st.markdown(
            """
            **Target Variable**
            - `selling_price`: selling price, numeric

            **Features**
            - **`name`** *(string)*: car model
            - **`year`** *(numeric, int)*: year of manufacture
            - **`km_driven`** *(numeric, int)*: mileage at the time of sale
            - **`fuel`** *(categorical: _Diesel_, _Petrol_, _CNG_, _LPG_, _electric_)*: fuel type
            - **`seller_type`** *(categorical: _Individual_, _Dealer_, _Trustmark Dealer_)*: seller type
            - **`transmission`** *(categorical: _Manual_, _Automatic_)*: transmission type
            - **`owner`** *(categorical: _First Owner_, _Second Owner_, _Third Owner_, _Fourth & Above Owner_)*: which owner is it?
            - **`mileage`** *(string, numerically interpreted)*: mileage, requires preprocessing
            - **`engine`** *(string, numerically interpreted)*: engine volume, requires preprocessing
            - **`max_power`** *(string, numerically interpreted)*: peak engine power, requires preprocessing
            - **`torque`** *(string, numerically interpreted, possibly two values)*: torque, requires preprocessing
            - **`seats`** *(numeric, float; logically categorical, int)*: number of seats
            """
        )

    with st.expander("Check for NaN", expanded=True):
        nan_summary = df.isnull().sum()
        if nan_summary.sum() == 0:
            st.success("No missing values in the data!")
        else:
            st.warning("There are missing values in the data!")
            st.write("Number of missing values per column:")
            st.write(nan_summary[nan_summary > 0])

    st.write("### Dataset after some transformations")
    st.write("name -> Company, Model; Mileage transform; Engine transform; Max_power transform;")
    df_v1 = pd.read_csv('data/data_preproc_v1.csv')
    st.write(df_v1.head(5))

    with st.expander("EDA", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            st.write("### Correlation Matrix of Numerical Features")
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
                "From the chart, it is clear that `engine` and `max_power` are most correlated, "
                "which makes sense since peak engine power is directly dependent on engine volume."
            )

            st.write("### Dependence of selling price on mileage")
            plt.figure(figsize=(8, 6))
            sns.scatterplot(x='km_driven', y='selling_price', data=df_v1)
            plt.title('Dependence of Selling Price on Mileage')
            plt.xlabel('Mileage (km)')
            plt.ylabel('Selling Price (selling_price)')
            st.pyplot(plt)
            st.write("As mileage increases, the selling price typically decreases")

            st.write("### Fuel Type Distribution")
            types_fuel = df_v1["fuel"].unique()
            x_fuel = df_v1["fuel"].value_counts()
            f, ax = plt.subplots(figsize=(5, 5))
            ax.pie(x_fuel, labels=types_fuel, autopct="%1.1f%%", startangle=90)
            ax.set_title("Fuel Type Distribution")
            st.pyplot(f)

        with col2:
            st.write("### Seller Type Distribution")
            types = df_v1["seller_type"].unique()
            x = df_v1["seller_type"].value_counts()
            f, ax = plt.subplots(figsize=(5, 5))
            ax.pie(x, labels=types, autopct="%1.1f%%", startangle=90)
            ax.set_title("Seller Type Distribution")
            st.pyplot(f)
            st.write(
                "It is clear that used cars are most often sold by individual sellers"
            )

            st.write("### Record Count by Company")
            companies = df_v1["Company"].unique()
            count = df_v1["Company"].value_counts()
            f, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(x=companies, y=count.values, ax=ax)
            ax.set_xlabel("Company Name")
            ax.set_ylabel("Count")
            ax.set_xticklabels(companies, rotation=75)
            st.pyplot(f)

            st.write("### Transmission Type Distribution")
            types_transmission = df_v1["transmission"].unique()
            x_transmission = df_v1["transmission"].value_counts()
            f, ax = plt.subplots(figsize=(5, 5))
            ax.pie(x_transmission, labels=types_transmission, autopct="%1.1f%%", startangle=90)
            ax.set_title("Transmission Type Distribution")
            st.pyplot(f)

            st.write("Most cars have a manual transmission, which makes up 87.1% of the total. This may indicate a wider distribution of manual gearboxes, especially in the used car market.")

        st.write("### Comparison of Price Distributions for Different Fuel Types")
        plt.figure(figsize=(10, 5))
        sns.boxplot(x='fuel', y='selling_price', data=df_v1)
        plt.title("Selling Price Distribution for Different Fuel Types")
        plt.xlabel("Fuel Type")
        plt.ylabel("Selling Price")
        st.pyplot(plt)
        st.write("From the chart, it is evident that diesel cars have a wide range of prices. Most cars fall in the price range around 0.2")

        # 2. Comparison of Price Distributions for Different Transmission Types
        st.write("### Comparison of Price Distributions for Different Transmission Types")
        plt.figure(figsize=(10, 5))
        sns.boxplot(x='transmission', y='selling_price', data=df_v1)
        plt.title("Selling Price Distribution for Different Transmission Types")
        plt.xlabel("Transmission Type")
        plt.ylabel("Selling Price")
        st.pyplot(plt)
        st.write("From the graph, it is evident that cars with automatic transmissions are sold significantly more expensive than manual ones")

    with st.expander("Hypothesis", expanded=True):
        st.write("### Hypothesis: Petrol cars with automatic transmission, sold by dealers, have a higher price than those sold by individuals")

        group_dealer = df_v1[(df_v1['fuel'] == 'Petrol') & (df_v1['transmission'] == 'Automatic') & (df_v1['seller_type'] == 'Dealer')][
            'selling_price']
        group_individual = df_v1[(df_v1['fuel'] == 'Petrol') & (df_v1['transmission'] == 'Automatic') & (df_v1['seller_type'] == 'Individual')][
            'selling_price']

        # 2. Comparison of Mean Prices
        st.write("#### Comparison of Average Selling Prices for Two Groups: Dealer vs Individual")
        plt.figure(figsize=(10, 5))
        sns.boxplot(x='seller_type', y='selling_price',
                    data=df_v1[(df_v1['fuel'] == 'Petrol') & (df_v1['transmission'] == 'Automatic')])
        plt.title("Selling Price Distribution for Petrol Cars with Automatic Transmission")
        plt.xlabel("Seller Type")
        plt.ylabel("Selling Price")
        st.pyplot(plt)

        t_stat, p_value = stats.ttest_ind(group_dealer, group_individual, equal_var=False)

        st.write(f"t-test: t-statistic = {t_stat:.2f}, p-value = {p_value:.4f}")

        # 4. Interpretation of Results
        if p_value < 0.05:
            st.success(
                "The hypothesis is confirmed: petrol cars with automatic transmission, sold by dealers, have a higher price than those sold by individuals.")
        else:
            st.warning("The hypothesis is refuted: there is no statistically significant difference in selling prices between the groups.")

    st.write("### Dataset for Model Training")
    st.write("[owner, fuel, seller_type, transmission, name] -> to digits\n")
    df_v2 = pd.read_csv('data/data_preproc_v2.csv')
    st.write(df_v2.head(5))


def show_prediction_page():
    st.header("Enter Car Parameters")
    user_input_df = main_page_input_features()

    if st.button("Make Prediction"):
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
    st.write("## Your Data")
    st.write(df)


def write_prediction(prediction):
    st.write("## Prediction")
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
