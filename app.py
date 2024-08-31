
import streamlit
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA

def detect_fraud(transactions_df, trained_model):
    # Ensure the input is in numeric format
    transactions_df = transactions_df.apply(pd.to_numeric, errors='coerce')

    # Drop rows with missing values
    transactions_df_clean = transactions_df.dropna()

    # Predict anomalies (fraudulent transactions)
    predictions = trained_model.predict(transactions_df_clean)

    # Get indices of fraudulent transactions
    fraudulent_indices = transactions_df_clean[predictions == -1].index

    # Return the fraudulent transactions
    fraudulent_transactions = transactions_df.loc[fraudulent_indices]
    
    return fraudulent_transactions

# Load your trained model here (for example purposes, I'll assume you have an IsolationForest model)
isolation_forest = IsolationForest(contamination=0.001, random_state=42)
y_pred_if = isolation_forest.fit_predict(X)

def detect_fraud(transactions_df, trained_model):
    transactions_df = transactions_df.apply(pd.to_numeric, errors='coerce')
    transactions_df_clean = transactions_df.dropna()
    predictions = trained_model.predict(transactions_df_clean)
    fraudulent_indices = transactions_df_clean[predictions == -1].index
    fraudulent_transactions = transactions_df.loc[fraudulent_indices]
    return fraudulent_transactions

def main():
    st.title("Credit Card Fraud Detection")
    
    uploaded_file = st.file_uploader("Upload your credit card transactions dataset (CSV or Excel)", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                transactions_df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                transactions_df = pd.read_excel(uploaded_file, sheet_name='creditcard_test')  # Adjust the sheet name if needed

            st.write("### Uploaded Dataset:")
            st.write(transactions_df.head())

            # Predict fraud
            fraudulent_transactions = detect_fraud(transactions_df, trained_model)

            if not fraudulent_transactions.empty:
                st.write("### Fraudulent Transactions Detected:")
                st.write(fraudulent_transactions)

                # Visualize anomalies
                pca = PCA(n_components=2)
                pca_transformed = pca.fit_transform(transactions_df.dropna())

                plt.figure(figsize=(10, 6))
                sns.scatterplot(x=pca_transformed[:, 0], y=pca_transformed[:, 1], hue=(transactions_df.index.isin(fraudulent_transactions.index)).astype(int), palette='coolwarm')
                plt.title('PCA: Fraudulent vs Non-Fraudulent Transactions')
                st.pyplot(plt)
            else:
                st.write("No fraudulent transactions detected.")

        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == '__main__':
    main()