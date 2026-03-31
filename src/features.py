class FeatureEngineer:

    # input: cleaned + merged dataframe
    # output: X (features), y (target)

    def create_features(self, df):
        # create behavior-based features
        pass

    def encode_categorical(self, df):
        # convert categorical → numeric
        pass

    def select_target(self, df, target_column):
        # split into X and y
        pass

    def get_features_and_target(self, df, target_column):
        # full pipeline for features
        pass