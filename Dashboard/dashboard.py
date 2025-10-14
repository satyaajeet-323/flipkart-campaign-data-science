import os
import json
import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

import shap
from scipy.stats import ks_2samp

# ----------------------------
# Paths & lazy loading
# ----------------------------
ART_DIR = "artifacts"
# DATA_DIR = "../data"  # Data folder outside dashboard
CSV_PATH = os.path.join("..\data\flipkart_campaign.csv")
PIPE_PATH = os.path.join(ART_DIR, "flipkart_pipeline.joblib")
COLS_PATH = os.path.join(ART_DIR, "expected_columns.json")
REF_PATH = os.path.join(ART_DIR, "reference_sample.csv")

# For Flipkart dataset - predicting high performing campaigns
LABEL_MAP = {0: "Low Performance", 1: "High Performance"}


@st.cache_resource(show_spinner=False)
def load_artifacts():
    load_error = None
    pipeline, expected_cols, ref = None, None, None
    
    # Create artifacts directory if it doesn't exist
    os.makedirs(ART_DIR, exist_ok=True)
    
    try:
        import joblib
        if os.path.exists(PIPE_PATH):
            pipeline = joblib.load(PIPE_PATH)
        else:
            load_error = "Model pipeline not found. Please train the model first."
    except Exception as e:
        load_error = f"Artifact load failed: {e}"

    try:
        if os.path.exists(COLS_PATH):
            expected_cols = json.load(open(COLS_PATH))["expected_input_cols"]
        else:
            # Define expected columns based on your dataset
            expected_cols = [
                'Total_amt_of_sale', 'avg_discount', 'no_of_customers_visited', 
                'no_of_products_sold', 'duration_days', 'campaign_budget', 
                'conversion_rate', 'returning_customers_percent', 'click_through_rate',
                'impressions', 'avg_session_time', 'customer_rating_avg'
            ]
    except Exception as e:
        load_error = f"Columns config load failed: {e}"

    if os.path.exists(REF_PATH):
        try:
            ref = pd.read_csv(REF_PATH)
        except Exception as e:
            load_error = f"Reference sample load failed: {e}"

    return pipeline, expected_cols, ref, load_error


@st.cache_data(show_spinner=False)
def load_data():
    """Load data from the data folder"""
    try:
        if os.path.exists(CSV_PATH):
            df = pd.read_csv(CSV_PATH)
            st.success(f"✅ Successfully loaded data")
            return df, None
        else:
            error_msg = f"CSV file not found at: {CSV_PATH}"
            st.error(f"❌ {error_msg}")
            return None, error_msg
    except Exception as e:
        error_msg = f"Error loading CSV: {e}"
        st.error(f"❌ {error_msg}")
        return None, error_msg


inference_pipeline, EXPECTED_COLS, REF, LOAD_ERR = load_artifacts()

st.set_page_config(page_title="Flipkart Campaign Performance Dashboard", layout="wide")
st.title("📊 Flipkart Campaign Performance — Predictions & Insights")
st.caption("Streamlit dashboard for campaign performance predictions, SHAP explanations, and analysis.")

# ----------------------------
# Sidebar controls
# ----------------------------
with st.sidebar:
    st.header("Controls")
    
    # Load data button instead of file uploader
    if st.button("🔄 Load Campaign Data", type="primary"):
        st.session_state.data_loaded = True
    else:
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False

    # Auto-detect candidate sensitive & target columns
    detected_sensitive = "platform"
    detected_target = "performance"
    
    if st.session_state.data_loaded:
        df_data, data_error = load_data()
        if df_data is not None:
            try:
                # Candidate sensitive cols: categorical columns
                cand_sens = [
                    c for c in df_data.columns 
                    if df_data[c].dtype == 'object' and df_data[c].nunique() <= 20
                ]
                if cand_sens:
                    for pref in ["platform", "Type", "maximum_sale_category", "payment_mode_used"]:
                        if pref in cand_sens:
                            detected_sensitive = pref
                            break
                    else:
                        detected_sensitive = cand_sens[0]

                # Candidate target columns
                for pref_t in ["performance", "target", "label", "conversion_rate", "Total_amt_of_sale"]:
                    if pref_t in df_data.columns:
                        detected_target = pref_t
                        break
            except Exception:
                pass

    sensitive_attr = st.text_input("Sensitive attribute (grouping column)", value=detected_sensitive)
    target_attr = st.text_input("Ground-truth column (optional)", value=detected_target)

    threshold = st.slider("Probability threshold for 'High Performance'", 0.0, 1.0, 0.5, 0.01)
    
    # Performance threshold for creating binary target
    perf_threshold = st.slider("Performance threshold (for binary classification)", 
                             0.0, 1.0, 0.7, 0.01)
    
    st.divider()
    st.write("Artifacts status:", "`OK`" if inference_pipeline else f"`Degraded: {LOAD_ERR}`")
    st.write("Data status:", "`Loaded`" if st.session_state.data_loaded else "`Click to load`")

# ----------------------------
# Helper functions
# ----------------------------
def align_columns(df: pd.DataFrame, expected_cols):
    for c in expected_cols:
        if c not in df.columns:
            df[c] = 0  # Fill missing with 0 instead of None for numeric columns
    return df[expected_cols]


def predict_df(df: pd.DataFrame):
    if inference_pipeline is None:
        # Fallback: simple rule-based prediction if no model
        conversion_rates = df.get('conversion_rate', pd.Series([0] * len(df)))
        total_sales = df.get('Total_amt_of_sale', pd.Series([0] * len(df)))
        
        # Simple scoring based on conversion rate and total sales
        scores = (conversion_rates / 10) + (total_sales / df['Total_amt_of_sale'].max() if df['Total_amt_of_sale'].max() > 0 else 0)
        proba = scores / 2  # Normalize to 0-1
        preds = (proba >= 0.5).astype(int)
        return preds, proba
    
    # Use actual model if available
    model = getattr(inference_pipeline, "named_steps", {}).get("model", None)
    proba = None
    if hasattr(inference_pipeline, "predict_proba"):
        proba = inference_pipeline.predict_proba(df)[:, 1]
    elif model is not None and hasattr(model, "predict_proba"):
        proba = model.predict_proba(
            inference_pipeline.named_steps["preprocess"].transform(df)
        )[:, 1]
    preds = inference_pipeline.predict(df)
    return preds, proba


def create_performance_target(df, threshold_col='conversion_rate', threshold_val=0.7):
    """Create binary performance target based on conversion rate"""
    if threshold_col in df.columns:
        return (df[threshold_col] >= threshold_val).astype(int)
    else:
        # Fallback: use total sales if conversion rate not available
        sales_threshold = df['Total_amt_of_sale'].quantile(0.7)
        return (df['Total_amt_of_sale'] >= sales_threshold).astype(int)


def psi(reference: pd.Series, current: pd.Series, bins: int = 10):
    ref = reference.dropna().astype(float)
    cur = current.dropna().astype(float)
    if len(ref) < 10 or len(cur) < 10:
        return np.nan
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.quantile(ref, quantiles)
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    ref_counts = np.histogram(ref, bins=edges)[0]
    cur_counts = np.histogram(cur, bins=edges)[0]
    ref_perc = np.where(ref_counts == 0, 1e-6, ref_counts) / max(1, ref_counts.sum())
    cur_perc = np.where(cur_counts == 0, 1e-6, cur_counts) / max(1, cur_counts.sum())
    return float(np.sum((cur_perc - ref_perc) * np.log(cur_perc / ref_perc)))


# ----------------------------
# Tabs
# ----------------------------
tab_pred, tab_shap, tab_fair, tab_drift, tab_eda = st.tabs(
    ["🔮 Predict", "🔎 SHAP", "⚖️ Fairness", "🌊 Drift", "📈 EDA"]
)

# ----------------------------
# Predict tab
# ----------------------------
with tab_pred:
    st.subheader("Campaign Performance Predictions")
    
    if inference_pipeline is None:
        # st.warning("No trained model found. Using rule-based predictions based on conversion rate and total sales.")
        pass
    
    df_in = None
    if st.session_state.data_loaded:
        df_data, data_error = load_data()
        if df_data is not None:
            df_in = df_data
            st.write(f"Loaded {len(df_in)} campaigns from data folder")
            st.dataframe(df_in.head(), use_container_width=True)
        else:
            st.error(f"Could not load data: {data_error}")
            df_in = None
    else:
        st.info("Click 'Load Campaign Data' button in sidebar to load data from data folder.")
        # Create sample data based on your dataset structure
        demo = {
            'Total_amt_of_sale': 5000000,
            'avg_discount': 35.5,
            'no_of_customers_visited': 40000,
            'no_of_products_sold': 8000,
            'duration_days': 5,
            'campaign_budget': 2000000,
            'conversion_rate': 3.5,
            'returning_customers_percent': 45.0,
            'click_through_rate': 2.8,
            'impressions': 500000,
            'avg_session_time': 10.5,
            'customer_rating_avg': 4.2
        }
        df_in = pd.DataFrame([demo])

    if df_in is not None:
        aligned_df = align_columns(df_in.copy(), EXPECTED_COLS)
        preds, proba = predict_df(aligned_df)
        
        out = pd.DataFrame({
            "prediction": preds.astype(int),
            "label": [LABEL_MAP.get(int(p), str(p)) for p in preds],
            "performance_score": proba if proba is not None else np.nan,
        })
        
        st.success(f"Predicted {len(out)} campaigns")
        st.dataframe(pd.concat([df_in.reset_index(drop=True), out], axis=1).head(50), 
                   use_container_width=True)

        # Performance summary
        high_perf_count = (out['prediction'] == 1).sum()
        st.metric("High Performance Campaigns", f"{high_perf_count}/{len(out)}", 
                 f"{(high_perf_count/len(out)*100):.1f}%")

        # Optional ground truth evaluation
        if target_attr and target_attr in df_in.columns:
            y_true = create_performance_target(df_in, target_attr, perf_threshold)
            y_pred = (out['performance_score'] >= threshold).astype(int) if 'performance_score' in out.columns else out['prediction']
            
            st.write("**Performance Metrics**")
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            
            col1, col2 = st.columns(2)
            col1.metric("Accuracy", f"{acc:.3f}")
            col2.metric("F1 Score", f"{f1:.3f}")
            
            # Classification report
            report_dict = classification_report(y_true, y_pred, output_dict=True, digits=3)
            report_df = pd.DataFrame(report_dict).T.reset_index()
            st.write("**Detailed Classification Report**")
            st.dataframe(report_df, use_container_width=True)
            
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            cm_df = pd.DataFrame(
                cm,
                index=["True Low Perf", "True High Perf"],
                columns=["Pred Low Perf", "Pred High Perf"],
            )
            st.write("**Confusion Matrix**")
            st.dataframe(cm_df, use_container_width=True)

# ----------------------------
# SHAP tab
# ----------------------------
with tab_shap:
    st.subheader("Feature Importance Analysis")
    
    if not st.session_state.data_loaded:
        st.info("Click 'Load Campaign Data' button to analyze feature importance.")
    else:
        try:
            df_data, data_error = load_data()
            if df_data is None:
                st.error(f"Data not loaded: {data_error}")
            else:
                df = df_data
                aligned_df = align_columns(df.copy(), EXPECTED_COLS)
                
                # Calculate correlation with performance if target exists
                if target_attr in df.columns:
                    y_true = create_performance_target(df, target_attr, perf_threshold)
                    correlations = {}
                    for col in EXPECTED_COLS:
                        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                            correlations[col] = np.corrcoef(df[col], y_true)[0, 1]
                    
                    corr_df = pd.DataFrame({
                        'feature': list(correlations.keys()),
                        'correlation_with_target': list(correlations.values())
                    }).sort_values('correlation_with_target', key=abs, ascending=False)
                    
                    st.write("**Feature Correlations with Performance**")
                    st.dataframe(corr_df, use_container_width=True)
                    
                    # Plot correlations
                    fig, ax = plt.subplots(figsize=(10, 6))
                    top_corr = corr_df.head(15)
                    colors = ['red' if x < 0 else 'blue' for x in top_corr['correlation_with_target']]
                    ax.barh(range(len(top_corr)), top_corr['correlation_with_target'], color=colors)
                    ax.set_yticks(range(len(top_corr)))
                    ax.set_yticklabels(top_corr['feature'])
                    ax.set_xlabel('Correlation with Performance')
                    ax.set_title('Top Feature Correlations with Campaign Performance')
                    plt.tight_layout()
                    st.pyplot(fig)
                
                # Basic feature statistics
                st.write("**Feature Statistics**")
                stats_df = aligned_df[EXPECTED_COLS].describe()
                st.dataframe(stats_df, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error in feature analysis: {e}")

# ----------------------------
# Fairness tab
# ----------------------------
with tab_fair:
    st.subheader("Campaign Performance by Groups")
    
    if not st.session_state.data_loaded:
        st.info("Click 'Load Campaign Data' button to analyze performance across groups.")
    else:
        try:
            df_data, data_error = load_data()
            if df_data is None:
                st.error(f"Data not loaded: {data_error}")
            else:
                df = df_data
                aligned_df = align_columns(df.copy(), EXPECTED_COLS)
                preds, proba = predict_df(aligned_df)
                pred_hat = (proba >= threshold).astype(int) if proba is not None else preds.astype(int)
                
                if sensitive_attr in df.columns:
                    # Group-level performance analysis
                    grp = df.groupby(df[sensitive_attr].astype(str), dropna=False)
                    summary = grp.apply(
                        lambda g: pd.Series({
                            'n': int(len(g)),
                            'high_perf_rate': float((pred_hat[g.index] == 1).mean()),
                            'avg_conversion': float(g.get('conversion_rate', pd.Series([0]*len(g))).mean()),
                            'avg_sales': float(g.get('Total_amt_of_sale', pd.Series([0]*len(g))).mean()),
                        })
                    ).reset_index()
                    
                    st.write(f"**Performance by {sensitive_attr}**")
                    st.dataframe(
                        summary.sort_values('high_perf_rate', ascending=False), 
                        use_container_width=True
                    )
                    
                    # Visualization
                    fig, ax = plt.subplots(figsize=(10, 6))
                    summary_sorted = summary.sort_values('high_perf_rate', ascending=True).tail(10)
                    ax.barh(summary_sorted[sensitive_attr], summary_sorted['high_perf_rate'])
                    ax.set_xlabel('High Performance Rate')
                    ax.set_title(f'High Performance Rate by {sensitive_attr}')
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                else:
                    st.warning(f"Sensitive attribute '{sensitive_attr}' not found in data.")
                    st.write("Available categorical columns:", 
                            [c for c in df.columns if df[c].dtype == 'object'][:10])
                
        except Exception as e:
            st.error(f"Error in fairness analysis: {e}")

# ----------------------------
# Drift tab
# ----------------------------
with tab_drift:
    st.subheader("Data Drift Analysis")
    
    if not st.session_state.data_loaded:
        st.info("Click 'Load Campaign Data' button to check for data drift.")
    else:
        try:
            df_data, data_error = load_data()
            if df_data is None:
                st.error(f"Data not loaded: {data_error}")
            else:
                current_df = df_data
                
                # If we have a reference, compare with it
                if REF is not None:
                    common_cols = list(set(REF.columns).intersection(current_df.columns))
                    numeric_cols = [
                        c for c in common_cols 
                        if pd.api.types.is_numeric_dtype(REF.get(c, pd.Series([0]))) 
                        and pd.api.types.is_numeric_dtype(current_df.get(c, pd.Series([0])))
                    ]
                    
                    if numeric_cols:
                        drift_results = []
                        for col in numeric_cols:
                            if col in REF.columns and col in current_df.columns:
                                psi_val = psi(REF[col], current_df[col])
                                ks_stat, ks_pval = ks_2samp(
                                    REF[col].dropna(), 
                                    current_df[col].dropna()
                                )
                                drift_results.append({
                                    'feature': col,
                                    'PSI': psi_val,
                                    'KS_pvalue': ks_pval,
                                    'drift_alert': 'HIGH' if psi_val > 0.2 else 'MEDIUM' if psi_val > 0.1 else 'LOW'
                                })
                        
                        drift_df = pd.DataFrame(drift_results).sort_values('PSI', ascending=False)
                        st.write("**Drift Analysis Results**")
                        st.dataframe(drift_df, use_container_width=True)
                    else:
                        st.warning("No common numeric columns for drift analysis.")
                else:
                    # Basic statistics of current data
                    st.write("**Current Data Statistics**")
                    numeric_cols = current_df.select_dtypes(include=[np.number]).columns.tolist()
                    if numeric_cols:
                        st.dataframe(current_df[numeric_cols].describe(), use_container_width=True)
                
        except Exception as e:
            st.error(f"Error in drift analysis: {e}")

# ----------------------------
# EDA tab
# ----------------------------
with tab_eda:
    st.subheader("Exploratory Data Analysis")
    
    if not st.session_state.data_loaded:
        st.info("Click 'Load Campaign Data' button to explore the data.")
    else:
        try:
            df_data, data_error = load_data()
            if df_data is None:
                st.error(f"Data not loaded: {data_error}")
            else:
                df = df_data
                
                # Basic info
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Campaigns", len(df))
                col2.metric("Columns", len(df.columns))
                col3.metric("Data Types", f"{len(df.select_dtypes(include=[np.number]).columns)} numeric, {len(df.select_dtypes(include=['object']).columns)} categorical")
                
                # Key metrics summary
                st.write("**Key Campaign Metrics Summary**")
                key_metrics = ['Total_amt_of_sale', 'conversion_rate', 'click_through_rate', 'customer_rating_avg']
                available_metrics = [m for m in key_metrics if m in df.columns]
                
                if available_metrics:
                    summary = df[available_metrics].describe()
                    st.dataframe(summary, use_container_width=True)
                
                # Distribution plots
                st.write("**Distribution of Key Metrics**")
                plot_cols = st.columns(2)
                
                for i, metric in enumerate(available_metrics[:4]):
                    with plot_cols[i % 2]:
                        fig, ax = plt.subplots(figsize=(8, 4))
                        ax.hist(df[metric].dropna(), bins=20, alpha=0.7, edgecolor='black')
                        ax.set_title(f'Distribution of {metric}')
                        ax.set_xlabel(metric)
                        ax.set_ylabel('Frequency')
                        plt.tight_layout()
                        st.pyplot(fig)
                
                # Correlation heatmap
                st.write("**Correlation Heatmap**")
                numeric_df = df.select_dtypes(include=[np.number])
                if len(numeric_df.columns) > 1:
                    fig, ax = plt.subplots(figsize=(12, 8))
                    corr_matrix = numeric_df.corr()
                    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
                    ax.set_title('Feature Correlations')
                    plt.tight_layout()
                    st.pyplot(fig)
                    
        except Exception as e:
            st.error(f"Error in EDA: {e}")

# ----------------------------
# Footer
# ----------------------------
st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **Flipkart Campaign Analytics**
    - Predict campaign performance
    - Analyze feature importance  
    - Check fairness across platforms
    - Monitor data drift
    - Explore campaign data
    """
)