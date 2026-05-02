import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt


class ModelManager:

    def scale_data(self, df):
        # Ensure numeric only
        df_numeric = df.select_dtypes(include=[np.number])

        # Remove any bad values
        df_numeric = df_numeric.replace([np.inf, -np.inf], np.nan)
        df_numeric = df_numeric.fillna(df_numeric.mean())

        scaler = StandardScaler()
        scaled = scaler.fit_transform(df_numeric)

        scaled_df = pd.DataFrame(
            scaled,
            columns=df_numeric.columns,
            index=df.index
        )
        return scaled_df, scaler

    def apply_pca(self, scaled_df, variance=0.90):
        pca = PCA(n_components=variance, random_state=42)
        pca_data = pca.fit_transform(scaled_df)

        pca_df = pd.DataFrame(
            pca_data,
            columns=[f"PC{i+1}" for i in range(pca_data.shape[1])],
            index=scaled_df.index
        )

        return pca_df, pca

    def choose_best_kmeans(self, X, min_k=2, max_k=8):
        results = []

        for k in range(min_k, max_k + 1):
            model = KMeans(
                n_clusters=k,
                init="k-means++",
                random_state=42,
                n_init=10
            )

            labels = model.fit_predict(X)

            silhouette = silhouette_score(X, labels)

            results.append({
                "k": k,
                "wcss": model.inertia_,
                "silhouette": silhouette
            })

        results_df = pd.DataFrame(results)

        # Pick highest silhouette score
        best_k = int(results_df.sort_values("silhouette", ascending=False).iloc[0]["k"])

        best_model = KMeans(
            n_clusters=best_k,
            init="k-means++",
            random_state=42,
            n_init=10
        )

        labels = best_model.fit_predict(X)

        return best_model, labels, results_df

    def choose_best_agglomerative(self, X, min_k=2, max_k=8, linkage_method="ward"):
        """
        Fits AgglomerativeClustering for each k in [min_k, max_k] and returns
        the model with the highest silhouette score, along with a full results
        DataFrame for comparison.

        Parameters
        ----------
        X : array-like or DataFrame
            PCA-reduced, scaled feature matrix.
        min_k : int
            Minimum number of clusters to evaluate.
        max_k : int
            Maximum number of clusters to evaluate.
        linkage_method : str
            Linkage criterion — 'ward', 'complete', 'average', or 'single'.
            'ward' minimizes within-cluster variance and is recommended for
            compact, well-separated segments.

        Returns
        -------
        best_model : AgglomerativeClustering
            Fitted model with the best silhouette score.
        labels : ndarray
            Cluster assignments for each sample.
        results_df : DataFrame
            Silhouette, Davies-Bouldin, and Calinski-Harabasz scores for
            every k evaluated.
        """
        results = []

        X_arr = X.values if hasattr(X, "values") else np.array(X)

        # fix crash??
        X_arr = np.array(X_arr, dtype=float)
        X_arr = np.nan_to_num(X_arr)

        for k in range(min_k, max_k + 1):
            model = AgglomerativeClustering(
                n_clusters=k,
                linkage=linkage_method
            )

            labels_k = model.fit_predict(X_arr)

            silhouette = silhouette_score(X_arr, labels_k)
            db = davies_bouldin_score(X_arr, labels_k)
            ch = calinski_harabasz_score(X_arr, labels_k)

            results.append({
                "k": k,
                "silhouette": silhouette,
                "davies_bouldin": db,
                "calinski_harabasz": ch
            })

        results_df = pd.DataFrame(results)

        # Pick highest silhouette score
        best_k = int(results_df.sort_values("silhouette", ascending=False).iloc[0]["k"])

        best_model = AgglomerativeClustering(
            n_clusters=best_k,
            linkage=linkage_method
        )

        labels = best_model.fit_predict(X_arr)

        return best_model, labels, results_df

    def plot_dendrogram(self, X, dataset_name, linkage_method="ward",
                        max_display_levels=10, save_path=None):
        """
        Plots a truncated dendrogram from a scipy linkage matrix, which helps
        visually justify the chosen number of clusters by showing where the
        largest merge distances (long vertical lines) occur.

        Parameters
        ----------
        X : array-like or DataFrame
            PCA-reduced, scaled feature matrix. For large datasets a random
            sample of up to 2000 rows is used automatically.
        dataset_name : str
            Label shown in the plot title.
        linkage_method : str
            Must match the linkage used in choose_best_agglomerative.
        max_display_levels : int
            Number of merge levels to display (passed as truncate_mode='level').
        save_path : str or None
            If provided, saves the figure to this path instead of showing it.
        """
        X_arr = X.values if hasattr(X, "values") else np.array(X)

        # Sample large datasets so scipy linkage stays tractable
        if len(X_arr) > 2000:
            np.random.seed(42)
            idx = np.random.choice(len(X_arr), size=2000, replace=False)
            X_arr = X_arr[idx]

        Z = linkage(X_arr, method=linkage_method)

        fig, ax = plt.subplots(figsize=(12, 5))
        dendrogram(
            Z,
            truncate_mode="level",
            p=max_display_levels,
            ax=ax,
            color_threshold=0.7 * max(Z[:, 2])
        )

        ax.set_title(f"Dendrogram — {dataset_name} ({linkage_method} linkage)",
                     fontsize=14)
        ax.set_xlabel("Sample index (or cluster size)")
        ax.set_ylabel("Merge distance")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
            plt.close()
        else:
            plt.show()

    def compare_models(self, X, dataset_name, min_k=2, max_k=8):
        """
        Runs both KMeans and Agglomerative clustering across the same k range
        and prints a side-by-side silhouette comparison.

        Returns
        -------
        km_results : DataFrame
        agg_results : DataFrame
        """
        print(f"\n===== MODEL COMPARISON: {dataset_name.upper()} =====")

        _, _, km_results = self.choose_best_kmeans(X, min_k=min_k, max_k=max_k)
        _, _, agg_results = self.choose_best_agglomerative(X, min_k=min_k, max_k=max_k)

        comparison = km_results[["k", "silhouette"]].rename(
            columns={"silhouette": "kmeans_silhouette"}
        ).merge(
            agg_results[["k", "silhouette", "davies_bouldin", "calinski_harabasz"]].rename(
                columns={"silhouette": "agglom_silhouette"}
            ),
            on="k"
        )

        comparison["better"] = comparison.apply(
            lambda r: "KMeans" if r["kmeans_silhouette"] > r["agglom_silhouette"] else "Agglom",
            axis=1
        )

        print(comparison.to_string(index=False))
        return km_results, agg_results

    def choose_best_gmm(self, X, min_k=2, max_k=8, covariance_type="diag"):
        """
        Fits a Gaussian Mixture Model for each k in [min_k, max_k] and returns
        the model with the highest silhouette score.

        How GMM works (from your lecture):
          - Each cluster k is modelled as a Gaussian with its own mean vector μ
            and covariance matrix Σ, plus a mixing weight π that represents
            how much of the data that Gaussian is assumed to have generated.
          - For every point, the model computes a responsibility — P(cluster | point)
            — using Bayes' Rule applied to the Gaussian density values. This is
            the soft assignment your slides described.
          - Hard labels (for silhouette scoring) come from argmax of those
            responsibilities, i.e. whichever cluster has the highest probability
            for that point.

        Why covariance_type='diag' and not 'full':
          After PCA you have 14 components. A full covariance matrix would have
          14×14 = 196 free parameters *per cluster*. With k=6 that's 1176
          parameters — too many to estimate reliably even at 91K rows, and it
          tends to overfit. 'diag' assumes the PCA components are independent
          (reasonable, since PCA decorrelates them by design) and only fits 14
          variance parameters per cluster, keeping the model stable.

        Parameters
        ----------
        X : array-like or DataFrame
            PCA-reduced, scaled feature matrix.
        min_k : int
            Minimum number of components (clusters) to evaluate.
        max_k : int
            Maximum number of components to evaluate.
        covariance_type : str
            Shape of each cluster's covariance matrix. 'diag' is recommended
            after PCA. Options: 'full', 'tied', 'diag', 'spherical'.

        Returns
        -------
        best_model : GaussianMixture
            Fitted GMM with the best silhouette score.
        labels : ndarray
            Hard cluster assignments (argmax of responsibilities).
        proba_df : DataFrame
            Soft probabilities — one column per cluster, one row per sample.
            This is the key advantage of GMM: each user gets a probability
            score for every segment, not just a single label.
        results_df : DataFrame
            Silhouette score for every k evaluated.
        """
        results = []

        X_arr = X.values if hasattr(X, "values") else np.array(X)
        X_arr = np.array(X_arr, dtype=float)
        X_arr = np.nan_to_num(X_arr)

        for k in range(min_k, max_k + 1):
            model = GaussianMixture(
                n_components=k,
                covariance_type=covariance_type,
                random_state=42,
                n_init=5,           # re-run EM from 5 random starts, keep best
                max_iter=200
            )

            model.fit(X_arr)
            labels_k = model.predict(X_arr)

            # Silhouette needs at least 2 distinct labels
            if len(set(labels_k)) < 2:
                continue

            silhouette = silhouette_score(X_arr, labels_k)

            results.append({
                "k": k,
                "silhouette": silhouette,
                "bic": model.bic(X_arr),   # lower BIC = better fit vs complexity
                "aic": model.aic(X_arr)    # lower AIC = better fit vs complexity
            })

        results_df = pd.DataFrame(results)

        # Pick highest silhouette (keeps evaluation consistent with KMeans/Agglom)
        best_k = int(results_df.sort_values("silhouette", ascending=False).iloc[0]["k"])

        best_model = GaussianMixture(
            n_components=best_k,
            covariance_type=covariance_type,
            random_state=42,
            n_init=5,
            max_iter=200
        )

        best_model.fit(X_arr)
        labels = best_model.predict(X_arr)

        # predict_proba gives the responsibility matrix — P(cluster | point)
        # This is the soft assignment from your lecture's responsibility step
        proba = best_model.predict_proba(X_arr)
        proba_df = pd.DataFrame(
            proba,
            columns=[f"prob_cluster_{i}" for i in range(best_k)],
            index=X.index if hasattr(X, "index") else None
        )

        return best_model, labels, proba_df, results_df

    def compare_all_models(self, X, dataset_name, min_k=2, max_k=8):
        """
        Runs KMeans, Agglomerative, and GMM across the same k range and prints
        a three-way silhouette comparison.

        Returns
        -------
        km_results, agg_results, gmm_results : DataFrames
        """
        print(f"\n===== THREE-WAY MODEL COMPARISON: {dataset_name.upper()} =====")

        _, _, km_results = self.choose_best_kmeans(X, min_k=min_k, max_k=max_k)
        _, _, agg_results = self.choose_best_agglomerative(X, min_k=min_k, max_k=max_k)
        _, _, _, gmm_results = self.choose_best_gmm(X, min_k=min_k, max_k=max_k)

        comparison = (
            km_results[["k", "silhouette"]]
            .rename(columns={"silhouette": "kmeans"})
            .merge(
                agg_results[["k", "silhouette"]].rename(columns={"silhouette": "agglom"}),
                on="k"
            )
            .merge(
                gmm_results[["k", "silhouette", "bic"]].rename(columns={"silhouette": "gmm"}),
                on="k"
            )
        )

        comparison["best_silhouette"] = comparison[["kmeans", "agglom", "gmm"]].idxmax(axis=1)

        print(comparison.to_string(index=False))
        return km_results, agg_results, gmm_results