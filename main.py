"""
main.py — Black Friday UI Predictions pipeline
Branch: testing-all-models

Runs all three clustering models — KMeans, Agglomerative (Ward), and GMM —
fully and independently on each dataset. Saves clustered CSVs for every
model and produces a single model_comparison_summary.csv that can be
cited directly in the paper.

Output files saved to src/:
  Clustered data (one per model per dataset):
    demographic_kmeans_clustered.csv
    demographic_agglom_clustered.csv
    demographic_gmm_clustered.csv      ← also includes soft probability columns
    clickstream_kmeans_clustered.csv
    clickstream_agglom_clustered.csv
    clickstream_gmm_clustered.csv
    purchase_kmeans_clustered.csv
    purchase_agglom_clustered.csv
    purchase_gmm_clustered.csv

  Summary for the paper:
    model_comparison_summary.csv       ← best k and silhouette per model per dataset
"""

import pandas as pd
import numpy as np
from src.features import FeatureEngineer
from src.modelmanager import ModelManager
from src.evaluator import Evaluator

# ── Setup ──────────────────────────────────────────────────────────────────────
fe = FeatureEngineer()
mm = ModelManager()
ev = Evaluator()

# ── 1. Load ────────────────────────────────────────────────────────────────────
demo_raw     = pd.read_csv("data/BlackFriday-Demographic.csv")
click_raw    = pd.read_csv("data/Click_Button.csv")
purchase_raw = pd.read_csv("data/Purchases.csv")

# ── 2. Clean ───────────────────────────────────────────────────────────────────
demo_clean     = fe.clean_demographic(demo_raw)
click_clean    = fe.clean_clickstream(click_raw)
purchase_clean = fe.clean_purchases(purchase_raw)
purchase_agg   = fe.aggregate_purchases(purchase_clean)

# ── 3. Encode ──────────────────────────────────────────────────────────────────
demo_enc     = fe.encode(demo_clean)
click_enc    = fe.encode(click_clean)
purchase_enc = fe.encode(purchase_agg)

# ── 4. Scale + PCA ────────────────────────────────────────────────────────────
demo_scaled, _     = mm.scale_data(demo_enc)
demo_pca, _        = mm.apply_pca(demo_scaled)

click_scaled, _    = mm.scale_data(click_enc)
click_pca, _       = mm.apply_pca(click_scaled)

purchase_scaled, _ = mm.scale_data(purchase_enc)
purchase_pca, _    = mm.apply_pca(purchase_scaled)

# ── 5. Three-way silhouette comparison (printed summary) ──────────────────────
print("\n" + "="*60)
print("  THREE-WAY COMPARISON: KMeans vs Agglomerative vs GMM")
print("="*60)

print("\nDEBUG CHECK — DEMO PCA")
print("NaN:", demo_pca.isna().any().any())
print("Inf:", np.isinf(demo_pca.values).any())
print("Dtypes:", demo_pca.dtypes.unique())

demo_pca_sample = demo_pca.sample(n=min(5000, len(demo_pca)), random_state=42)
click_pca_sample = click_pca.sample(n=min(5000, len(click_pca)), random_state=42)


mm.compare_all_models(demo_pca_sample,     "Demographic")
mm.compare_all_models(click_pca_sample,    "Clickstream")
mm.compare_all_models(purchase_pca, "Purchases")

# ── 6. Fit all three models on each dataset ───────────────────────────────────
print("\n" + "="*60)
print("  FITTING BEST MODELS")
print("="*60)

demo_fit = demo_pca.sample(n=min(5000, len(demo_pca)), random_state=42)
click_fit = click_pca.sample(n=min(5000, len(click_pca)), random_state=42)
purchase_fit = purchase_pca.sample(n=min(5000, len(purchase_pca)), random_state=42)

km_demo,  km_demo_labels,  km_demo_results  = mm.choose_best_kmeans(demo_fit)
ag_demo,  ag_demo_labels,  ag_demo_results  = mm.choose_best_agglomerative(demo_fit)
gm_demo,  gm_demo_labels,  gm_demo_proba, gm_demo_results = mm.choose_best_gmm(demo_fit)

# ── 7. Profile each model on each dataset ─────────────────────────────────────
print("\n" + "="*60)
print("  CLUSTER PROFILES")
print("="*60)

# KMeans profiles
km_demo_profile,  _ = ev.cluster_profile(demo_clean,    km_demo_labels,  "Demographic  [KMeans]")
km_click_profile, _ = ev.cluster_profile(click_clean,   km_click_labels, "Clickstream  [KMeans]")
km_pur_profile,   _ = ev.cluster_profile(purchase_agg,  km_pur_labels,   "Purchases    [KMeans]")

# Agglomerative profiles
ag_demo_profile,  _ = ev.agglomerative_profile(demo_clean,   ag_demo_labels,  "Demographic  [Agglom]", ag_demo_results)
ag_click_profile, _ = ev.agglomerative_profile(click_clean,  ag_click_labels, "Clickstream  [Agglom]", ag_click_results)
ag_pur_profile,   _ = ev.agglomerative_profile(purchase_agg, ag_pur_labels,   "Purchases    [Agglom]", ag_pur_results)

# GMM profiles (includes responsibility summary)
gm_demo_profile,  _, _ = ev.gmm_profile(demo_clean,   gm_demo_labels,  gm_demo_proba,  "Demographic  [GMM]", gm_demo_results)
gm_click_profile, _, _ = ev.gmm_profile(click_clean,  gm_click_labels, gm_click_proba, "Clickstream  [GMM]", gm_click_results)
gm_pur_profile,   _, _ = ev.gmm_profile(purchase_agg, gm_pur_labels,   gm_pur_proba,   "Purchases    [GMM]", gm_pur_results)

# ── 8. Build paper-ready comparison summary ───────────────────────────────────
def best_row(results_df, model_name, dataset_name):
    """Extract the best-k row from a results DataFrame for the summary table."""
    best = results_df.sort_values("silhouette", ascending=False).iloc[0]
    return {
        "dataset":   dataset_name,
        "model":     model_name,
        "best_k":    int(best["k"]),
        "silhouette": round(best["silhouette"], 4),
    }

summary_rows = [
    best_row(km_demo_results,  "KMeans",       "Demographic"),
    best_row(ag_demo_results,  "Agglomerative","Demographic"),
    best_row(gm_demo_results,  "GMM",          "Demographic"),
    best_row(km_click_results, "KMeans",       "Clickstream"),
    best_row(ag_click_results, "Agglomerative","Clickstream"),
    best_row(gm_click_results, "GMM",          "Clickstream"),
    best_row(km_pur_results,   "KMeans",       "Purchases"),
    best_row(ag_pur_results,   "Agglomerative","Purchases"),
    best_row(gm_pur_results,   "GMM",          "Purchases"),
]

summary_df = pd.DataFrame(summary_rows)

# Mark the winning model per dataset
summary_df["best_model"] = False
for dataset in summary_df["dataset"].unique():
    mask = summary_df["dataset"] == dataset
    best_idx = summary_df.loc[mask, "silhouette"].idxmax()
    summary_df.loc[best_idx, "best_model"] = True

print("\n" + "="*60)
print("  PAPER SUMMARY — Best k and Silhouette per Model per Dataset")
print("="*60)
print(summary_df.to_string(index=False))

summary_df.to_csv("src/model_comparison_summary.csv", index=False)
print("\n  Saved: src/model_comparison_summary.csv")

# ── 9. Save all clustered CSVs ────────────────────────────────────────────────
print("\nSaving all clustered datasets...")

def save_clustered(clean_df, labels, path, proba_df=None):
    out = clean_df.copy().reset_index(drop=True)
    out["cluster"] = labels
    if proba_df is not None:
        out = pd.concat([out, proba_df.reset_index(drop=True)], axis=1)
    out.to_csv(path, index=False)

# Demographic
save_clustered(demo_clean,   km_demo_labels,  "src/demographic_kmeans_clustered.csv")
save_clustered(demo_clean,   ag_demo_labels,  "src/demographic_agglom_clustered.csv")
save_clustered(demo_clean,   gm_demo_labels,  "src/demographic_gmm_clustered.csv",   gm_demo_proba)

# Clickstream
save_clustered(click_clean,  km_click_labels, "src/clickstream_kmeans_clustered.csv")
save_clustered(click_clean,  ag_click_labels, "src/clickstream_agglom_clustered.csv")
save_clustered(click_clean,  gm_click_labels, "src/clickstream_gmm_clustered.csv",   gm_click_proba)

# Purchases
save_clustered(purchase_agg, km_pur_labels,   "src/purchase_kmeans_clustered.csv")
save_clustered(purchase_agg, ag_pur_labels,   "src/purchase_agglom_clustered.csv")
save_clustered(purchase_agg, gm_pur_labels,   "src/purchase_gmm_clustered.csv",      gm_pur_proba)

print("  Done. 9 clustered CSVs + model_comparison_summary.csv saved to src/")

