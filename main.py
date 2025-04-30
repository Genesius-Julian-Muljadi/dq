import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import time
from PIL import Image

st.set_page_config(page_title="Heart Disease Data Analyzer", layout="wide")
add_selectitem = st.sidebar.selectbox("Pages", ("Home", "Dataset statistics", "Predictions"))

st.title("Heart Disease Data Analysis App")

def home():
    st.subheader("Description")
    st.write("""
    This is an app that predicts heart disease with sklearn machine learning, using the data model from this [Heart Disease dataset](https://archive.ics.uci.edu/dataset/45/heart+disease).

    When not using a custom dataset, the app will use the dataset above. Any custom datasets must use the same model (same columns) or some functions will break.

    This project was made to complete DQLab's portfolio assignment.
    """)

def statistics():
    df = None

    if uploaded_file:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)

    if df is None:
        df = pd.read_csv("heart_disease.csv")

    st.write("This page simply describes and displays the data of the current dataset, before any analysis.")

    st.subheader("Data Preview")
    st.dataframe(df.head())

    st.subheader("Summary Statistics")
    st.write(df.describe())

    st.subheader("Visualizations")
    numeric_columns = df.select_dtypes(include=["float64", "int64"]).columns.tolist()

    st.markdown("### Histogram")
    col = st.selectbox("Choose a numeric column for histogram", numeric_columns)
    fig, ax = plt.subplots()
    sns.histplot(df[col], kde=True, ax=ax)
    st.pyplot(fig)

    st.markdown("### Scatter Plot")
    col_x = st.selectbox("X-axis", numeric_columns, key="scatter_x")
    col_y = st.selectbox("Y-axis", numeric_columns, key="scatter_y")
    fig2, ax2 = plt.subplots()
    sns.scatterplot(data=df, x=col_x, y=col_y, ax=ax2)
    st.pyplot(fig2)

    st.markdown("### Correlation Heatmap")
    fig3, ax3 = plt.subplots()
    corr = df[numeric_columns].corr()
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        ax=ax3,
        annot_kws={"size": 8},
        cbar_kws={"shrink": 0.75}
    )
    ax3.tick_params(labelsize=8)
    st.pyplot(fig3)

def heart():
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
    else:
        img = Image.open("heart-disease.jpg")
        st.image(img, width=500)
        def user_input_features():
            st.sidebar.header("Inputs")
            cp = st.sidebar.slider("Chest pain type", 1,4,2)
            if cp == 1.0:
                wcp = "Stable angina"
            elif cp == 2.0:
                wcp = "Unstable angina"
            elif cp == 3.0:
                wcp = "Severe unstable angina"
            else:
                wcp = "Non-heart-related"
            st.sidebar.write(wcp)
            thalach = st.sidebar.slider("Maximum heart rate", 71, 202, 80)
            slope = st.sidebar.slider("ST segment slope on EKG", 0, 2, 1)
            oldpeak = st.sidebar.slider("Decreasing ST segment count", 0.0, 6.2, 1.0)
            exang = st.sidebar.slider("Exercise-induced angina?", 0, 1, 1)
            ca = st.sidebar.slider("Major vessel count", 0, 3, 1)
            thal = st.sidebar.slider("Thalium test result", 1, 3, 1)
            sex = st.sidebar.selectbox("Sex", ("Female", "Male"))
            if sex == "Female":
                sex = 0
            else:
                sex = 1 
            age = st.sidebar.slider("Age", 29, 77, 30)
            data = {"cp": cp,
                    "thalach": thalach,
                    "slope": slope,
                    "oldpeak": oldpeak,
                    "exang": exang,
                    "ca":ca,
                    "thal":thal,
                    "sex": sex,
                    "age":age}
            features = pd.DataFrame(data, index=[0])
            return features
        input_df = user_input_features()

    if st.sidebar.button("Predict!"):
        df = input_df
        st.write(df)
        with open("output_decision_tree.pkl", "rb") as file:  
            loaded_model = pickle.load(file)
        prediction = loaded_model.predict(df)        
        result = ["No Heart Disease" if prediction == 0 else "Likely Heart Disease. Please check with a doctor"]
        st.subheader("Prediction: ")
        output = str(result[0])
        with st.spinner("Please wait..."):
            time.sleep(4)
            st.success(f"{output}")

if add_selectitem == "Home":
    home()
elif add_selectitem == "Dataset statistics":
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    st.info("""
    You may upload your own CSV file containing a DataFrame.

    Default data is from the [Heart Disease dataset](https://archive.ics.uci.edu/dataset/45/heart+disease) by UCIML.
    """)

    statistics()
elif add_selectitem == "Predictions":
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    st.info("""
    You may upload your own CSV file containing a DataFrame.

    Default data is from the [Heart Disease dataset](https://archive.ics.uci.edu/dataset/45/heart+disease) by UCIML.
    """)

    heart()
