class Evaluator:

    # input: y_test, y_pred
    # output: metrics

    def evaluate(self, y_test, y_pred):
        # accuracy, precision, recall, F1
        pass

    def print_report(self, metrics):
        # display results
        pass

    def evaluate_by_group(self, X_test, y_test, y_pred, group_column):
        # compare across demographics
        pass