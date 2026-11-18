import streamlit as st
import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.metrics import confusion_matrix
from pathlib import Path

st.set_page_config(
    page_title="Fake Job Posting Detection",
    page_icon="🕵️",
    layout="wide"
)

@st.cache_resource
def load_model():
    model_path = Path(__file__).resolve().parent / "fake_job_model.pkl"
    with open(model_path, "rb") as f:
        return pickle.load(f)
try:
    model_data = load_model()
    nb_classifier = model_data["nb_classifier"]
    clf_log = model_data["clf_log"]
    clf_num = model_data["clf_num"]
    count_vectorizer = model_data["count_vectorizer"]
except Exception as e:
    st.error("Unable to load the trained model.")
    st.error(f"Error: {e}")
    st.stop()

location_stats = {
    "new york": (30, 120),
    "san francisco": (10, 90),
    "remote": (20, 80),
    "los angeles": (25, 100)
}

def calculate_ratio(location):
    location = str(location).strip().lower()
    if location in location_stats:
        fake, real = location_stats[location]
        total = fake + real
        if total > 0:
            return fake / total
    return 0.5

def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        ax=ax,
        xticklabels=["Real", "Fake"],
        yticklabels=["Real", "Fake"]
    )
    ax.set_title(f"Confusion Matrix: {model_name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)
    plt.close(fig)

st.title("🕵️ Fake Job Posting Detection App")
st.write(
    "This application uses machine learning to analyze job postings "
    "and predict whether a job posting is likely to be genuine or fraudulent."
)
st.markdown("---")

tab1, tab2 = st.tabs(["🔍 Predict Fake Job", "📊 Exploratory Data Analysis"])

with tab1:
    st.header("🔍 Enter Job Posting Details")
    st.write(
        "Enter the details of a job posting below and click **Predict**."
    )

    location = st.text_input(
        "📍 Job Location",
        placeholder="Example: New York, Remote, San Francisco"
    )

    description = st.text_area(
        "📄 Job Description",
        placeholder="Enter the job description..."
    )

    requirements = st.text_area(
        "📋 Job Requirements",
        placeholder="Enter the job requirements..."
    )

    telecommuting = st.selectbox(
        "🏠 Telecommuting (Remote Job)",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    if st.button("🚀 Predict Job", use_container_width=True):
        if not description.strip() and not requirements.strip():
            st.warning(
                "Please enter a job description or job requirements."
            )
        else:
            combined_text = f"{description} {requirements}"
            character_count = len(combined_text)
            ratio = calculate_ratio(location)

            text_vector = count_vectorizer.transform([combined_text])

            numerical_input = np.array(
                [[telecommuting, ratio, character_count]]
            )

            pred_nb = nb_classifier.predict(text_vector)
            pred_log = clf_log.predict(text_vector)
            pred_num = clf_num.predict(numerical_input)

            final_pred = (
                0
                if pred_log[0] == 0 and pred_num[0] == 0
                else 1
            )

            st.markdown("---")
            st.subheader("Prediction Result")
            if final_pred == 0:
                st.success("✅ This is likely a **genuine job posting**.")
            else:
                st.error("⚠️ This is likely a **fraudulent job posting**.")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.info(
                    f"Naive Bayes: **{'Real' if pred_nb[0] == 0 else 'Fake'}**"
                )
            with col2:
                st.info(
                    f"Logistic Regression: **{'Real' if pred_log[0] == 0 else 'Fake'}**"
                )
            with col3:
                st.info(
                    f"Numerical Model: **{'Real' if pred_num[0] == 0 else 'Fake'}**"
                )

            st.markdown("---")
            st.subheader("📊 Input Features")

            feature_col1, feature_col2, feature_col3 = st.columns(3)

            with feature_col1:
                st.metric("Character Count", character_count)
            with feature_col2:
                st.metric("Location Ratio", f"{ratio:.2f}")
            with feature_col3:
                st.metric(
                    "Telecommuting",
                    "Yes" if telecommuting == 1 else "No"
                )

with tab2:
    st.header("📊 Exploratory Data Analysis")
    st.write(
        "Upload a cleaned job-posting CSV file to explore its statistics, "
        "distributions, correlations and word patterns."
    )

    uploaded_file = st.file_uploader(
        "📂 Upload a cleaned CSV dataset",
        type=["csv"]
    )

    if uploaded_file is None:
        st.info("Upload a CSV file to start the exploratory analysis.")
    else:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Unable to read the CSV file: {e}")
            st.stop()

        st.markdown("---")
        st.subheader("📄 Dataset Information")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows", df.shape[0])
        with col2:
            st.metric("Columns", df.shape[1])
        with col3:
            st.metric("Missing Values", int(df.isnull().sum().sum()))

        st.subheader("📋 Dataset Columns")
        st.write(list(df.columns))

        st.subheader("👀 Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)

        numeric_df = df.select_dtypes(include=[np.number])

        st.subheader("📊 Basic Statistics")
        if not numeric_df.empty:
            st.dataframe(numeric_df.describe(), use_container_width=True)
        else:
            st.info("No numerical columns are available.")

        if "fraudulent" in df.columns:
            st.markdown("---")
            st.subheader("🚨 Fraudulent Job Posting Distribution")
            fig1, ax1 = plt.subplots(figsize=(7, 5))
            sns.countplot(data=df, x="fraudulent", ax=ax1)
            ax1.set_xlabel("Fraudulent")
            ax1.set_ylabel("Number of Job Postings")
            ax1.set_xticklabels(["Real", "Fake"])
            st.pyplot(fig1)
            plt.close(fig1)

        if "character_count" in df.columns:
            st.subheader("📝 Character Count Distribution")
            fig2, ax2 = plt.subplots(figsize=(8, 5))
            sns.histplot(df["character_count"].dropna(), kde=True, ax=ax2)
            ax2.set_xlabel("Character Count")
            ax2.set_ylabel("Frequency")
            st.pyplot(fig2)
            plt.close(fig2)

        if "ratio" in df.columns:
            st.subheader("📍 Location Ratio Distribution")
            fig3, ax3 = plt.subplots(figsize=(8, 5))
            sns.histplot(df["ratio"].dropna(), kde=True, ax=ax3)
            ax3.set_xlabel("Location Ratio")
            ax3.set_ylabel("Frequency")
            st.pyplot(fig3)
            plt.close(fig3)

        if "location" in df.columns:
            st.markdown("---")
            st.subheader("📍 Top 10 Job Locations")
            location_counts = df["location"].dropna().value_counts().head(10)
            if not location_counts.empty:
                fig4, ax4 = plt.subplots(figsize=(10, 6))
                sns.barplot(
                    x=location_counts.index,
                    y=location_counts.values,
                    ax=ax4
                )
                ax4.set_xlabel("Location")
                ax4.set_ylabel("Number of Job Postings")
                ax4.tick_params(axis="x", rotation=45)
                st.pyplot(fig4)
                plt.close(fig4)

        if "industry" in df.columns:
            st.subheader("🏢 Top 10 Industries")
            industry_counts = df["industry"].dropna().value_counts().head(10)
            if not industry_counts.empty:
                fig5, ax5 = plt.subplots(figsize=(10, 6))
                sns.barplot(
                    x=industry_counts.index,
                    y=industry_counts.values,
                    ax=ax5
                )
                ax5.set_xlabel("Industry")
                ax5.set_ylabel("Number of Job Postings")
                ax5.tick_params(axis="x", rotation=45)
                st.pyplot(fig5)
                plt.close(fig5)

        if numeric_df.shape[1] > 1:
            st.markdown("---")
            st.subheader("🔥 Correlation Heatmap")
            corr = numeric_df.corr()
            fig6, ax6 = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax6)
            st.pyplot(fig6)
            plt.close(fig6)

        if "description" in df.columns and "fraudulent" in df.columns:
            st.markdown("---")
            st.subheader("☁️ Job Description Word Clouds")

            real_desc = " ".join(
                df[df["fraudulent"] == 0]["description"]
                .dropna()
                .astype(str)
            )
            fake_desc = " ".join(
                df[df["fraudulent"] == 1]["description"]
                .dropna()
                .astype(str)
            )

            if real_desc.strip():
                wordcloud_real = WordCloud(
                    width=1200,
                    height=600,
                    max_words=100,
                    background_color="white"
                ).generate(real_desc)

                st.markdown("### ✅ Real Job Descriptions")
                st.image(
                    wordcloud_real.to_array(),
                    use_container_width=True
                )

            if fake_desc.strip():
                wordcloud_fake = WordCloud(
                    width=1200,
                    height=600,
                    max_words=100,
                    background_color="black",
                    colormap="Reds"
                ).generate(fake_desc)

                st.markdown("### ⚠️ Fake Job Descriptions")
                st.image(
                    wordcloud_fake.to_array(),
                    use_container_width=True
                )

        required_columns = ["fraudulent", "description", "requirements"]

        if all(column in df.columns for column in required_columns):
            st.markdown("---")
            st.subheader("🤖 Model Evaluation")

            desc = df["description"].fillna("").astype(str)
            reqs = df["requirements"].fillna("").astype(str)
            combined_texts = desc + " " + reqs

            text_vectors = count_vectorizer.transform(combined_texts)

            numerical_inputs = []

            for _, row in df.iterrows():
                try:
                    telecommuting_value = int(
                        row.get("telecommuting", 0)
                    )
                except Exception:
                    telecommuting_value = 0

                location_value = str(row.get("location", ""))
                description_value = str(row.get("description", ""))
                requirements_value = str(row.get("requirements", ""))

                character_count_value = len(
                    description_value + requirements_value
                )

                numerical_inputs.append([
                    telecommuting_value,
                    calculate_ratio(location_value),
                    character_count_value
                ])

            numerical_inputs = np.array(numerical_inputs)
            y_true = df["fraudulent"].astype(int)

            pred_nb = nb_classifier.predict(text_vectors)
            pred_log = clf_log.predict(text_vectors)
            pred_num = clf_num.predict(numerical_inputs)

            final_pred = (
                ((pred_log == 1) | (pred_num == 1))
                .astype(int)
            )

            st.markdown("### Confusion Matrices")

            col1, col2 = st.columns(2)
            with col1:
                plot_confusion_matrix(
                    y_true, pred_nb, "Naive Bayes"
                )
            with col2:
                plot_confusion_matrix(
                    y_true, pred_log, "Logistic Regression"
                )

            col3, col4 = st.columns(2)
            with col3:
                plot_confusion_matrix(
                    y_true, pred_num, "Numerical Model"
                )
            with col4:
                plot_confusion_matrix(
                    y_true, final_pred, "Combined Model"
                )
        else:
            st.info(
                "For model evaluation, the uploaded dataset must contain: "
                "fraudulent, description and requirements."
            )

st.markdown("---")
st.caption(
    "Fake Job Posting Detection | Machine Learning & Data Analysis"
)
