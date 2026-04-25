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