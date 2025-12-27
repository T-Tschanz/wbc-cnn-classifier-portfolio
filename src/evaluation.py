import os
import pandas as pd
from sklearn.metrics import classification_report
from src.visualization import plot_confusion_matrix

def evaluate_predictions(df, labels_df, class_labels, st):
    # Normalize filenames before merging
    df['Filename'] = df['Filename'].apply(lambda x: os.path.basename(str(x).strip().lower()))
    labels_df['Filename'] = labels_df['Filename'].apply(lambda x: os.path.basename(str(x).strip().lower()))
    merged_df = pd.merge(df, labels_df, on='Filename', how='inner')
    
    # Select only Arrow-compatible columns for display
    display_df = merged_df[['Filename', 'Prediction', 'Class', 'Confidence']].copy()
    
    # Display merged data
    st.dataframe(display_df)
    
    # Use 'Class' to match the CSV column name
    y_true = merged_df['Class']
    y_pred = merged_df['Prediction']
    
    # Generate and display classification report
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose()
    st.subheader("Classification Report")
    st.dataframe(report_df)
    
    # Generate and display confusion matrix
    st.subheader("Confusion Matrix")
    plot_confusion_matrix(y_true, y_pred, labels=class_labels, st=st)
