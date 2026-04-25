import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
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

    def choose_best_gmm(self, X, min_k=2, max_k=8):
        results = []

        for k in range(min_k, max_k + 1):
            gmm = GaussianMixture(
                n_components=k,
                covariance_type="full",
                random_state=42
            )

            labels = gmm.fit_predict(X)

            if len(set(labels)) > 1:
                silhouette = silhouette_score(X, labels)
            else:
                silhouette = None

            results.append({
                "k": k,
                "bic": gmm.bic(X),
                "aic": gmm.aic(X),
                "silhouette": silhouette
            })

        results_df = pd.DataFrame(results)

        best_k = int(results_df.sort_values("bic").iloc[0]["k"])

        best_gmm = GaussianMixture(
            n_components=best_k,
            covariance_type="full",
            random_state=42
        )

        labels = best_gmm.fit_predict(X)
        probabilities = best_gmm.predict_proba(X)

        return best_gmm, labels, probabilities, results_df