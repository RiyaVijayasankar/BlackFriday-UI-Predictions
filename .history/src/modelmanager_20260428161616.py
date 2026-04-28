import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


class ModelManager:

    def scale_data(self, df):
        scaler = StandardScaler()
        scaled = scaler.fit_transform(df)

        scaled_df = pd.DataFrame(
            scaled,
            columns=df.columns,
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