import pandas as pd


class Evaluator:


    def cluster_profile(self, original_df, labels, dataset_name):
        df = original_df.copy()
        df["cluster"] = labels

        numeric_profile = df.groupby("cluster").mean(numeric_only=True)
        cluster_sizes = df["cluster"].value_counts().sort_index()

        print(f"\n===== {dataset_name.upper()} CLUSTER SIZES =====")
        print(cluster_sizes)

        print(f"\n===== {dataset_name.upper()} CLUSTER PROFILE =====")
        print(numeric_profile)

        return numeric_profile, cluster_sizes

    def top_cluster_features(self, profile, top_n=5):
        overall_mean = profile.mean()
        differences = profile.subtract(overall_mean).abs()

        top_features = {}

        for cluster in profile.index:
            top_features[cluster] = differences.loc[cluster].sort_values(ascending=False).head(top_n)

        return top_features

    def compare_datasets(self, demo_profile, click_profile, purchase_profile):
        comparison = pd.DataFrame({
            "demographic_avg": demo_profile.mean(axis=1),
            "clickstream_avg": click_profile.mean(axis=1),
            "purchase_avg": purchase_profile.mean(axis=1)
        })

        correlation_matrix = comparison.corr()

        print("\n===== SEGMENT LEVEL COMPARISON =====")
        print(comparison)

        print("\n===== CORRELATION MATRIX =====")
        print(correlation_matrix)

        return comparison, correlation_matrix

    def agglomerative_profile(self, original_df, labels, dataset_name,
                              results_df=None):
        """
        Profiles agglomerative clusters and optionally prints the per-k metric
        table so the choice of k is transparent and reproducible.

        Parameters
        ----------
        original_df : DataFrame
            Un-encoded original features (for interpretable means).
        labels : ndarray
            Cluster assignments from choose_best_agglomerative.
        dataset_name : str
            Display label used in printed headers.
        results_df : DataFrame or None
            Output of choose_best_agglomerative; if provided, the full metric
            table (silhouette, Davies-Bouldin, Calinski-Harabasz) is printed
            so reviewers can see why this k was chosen.

        Returns
        -------
        numeric_profile : DataFrame
        cluster_sizes : Series
        """
        if results_df is not None:
            print(f"\n===== {dataset_name.upper()} — AGGLOMERATIVE K SELECTION =====")
            print(results_df.to_string(index=False))

        profile, sizes = self.cluster_profile(original_df, labels, dataset_name)
        return profile, sizes

    def gmm_profile(self, original_df, labels, proba_df, dataset_name,
                    results_df=None):
        """
        Profiles GMM clusters and prints the soft assignment summary — the key
        output that separates GMM from KMeans and Agglomerative.

        The soft assignments (responsibilities from your lecture) show how
        confidently each point belongs to its assigned cluster. A user with
        prob_cluster_0 = 0.95 is a clear member; one with 0.51 sits on the
        boundary and might deserve a blended UI experience.

        Parameters
        ----------
        original_df : DataFrame
            Un-encoded original features (for interpretable cluster means).
        labels : ndarray
            Hard cluster assignments (argmax of responsibilities).
        proba_df : DataFrame
            Soft probabilities from choose_best_gmm — one column per cluster.
        dataset_name : str
            Display label used in printed headers.
        results_df : DataFrame or None
            Output of choose_best_gmm; if provided, prints the per-k silhouette
            and BIC table so the choice of k is documented.

        Returns
        -------
        numeric_profile : DataFrame
        cluster_sizes : Series
        confidence_summary : DataFrame
            Mean and min responsibility score per cluster — a low mean means
            the cluster boundary is fuzzy and users are genuinely ambiguous.
        """
        if results_df is not None:
            print(f"\n===== {dataset_name.upper()} — GMM K SELECTION =====")
            print(results_df.to_string(index=False))

        profile, sizes = self.cluster_profile(original_df, labels, dataset_name)

        # Confidence summary: for each cluster, what's the average probability
        # that its assigned members actually belong there?
        proba_with_label = proba_df.copy()
        proba_with_label["cluster"] = labels

        confidence_rows = []
        for cluster_id in sorted(set(labels)):
            col = f"prob_cluster_{cluster_id}"
            if col not in proba_with_label.columns:
                continue
            member_probs = proba_with_label.loc[
                proba_with_label["cluster"] == cluster_id, col
            ]
            confidence_rows.append({
                "cluster": cluster_id,
                "mean_responsibility": member_probs.mean(),
                "min_responsibility": member_probs.min(),
                "pct_high_confidence": (member_probs >= 0.80).mean()
            })

        confidence_summary = pd.DataFrame(confidence_rows).set_index("cluster")

        print(f"\n===== {dataset_name.upper()} — GMM RESPONSIBILITY SUMMARY =====")
        print("(mean_responsibility: avg P(cluster | point) for assigned members)")
        print("(pct_high_confidence: share of members with P >= 0.80)")
        print(confidence_summary.round(3).to_string())

        return profile, sizes, confidence_summary