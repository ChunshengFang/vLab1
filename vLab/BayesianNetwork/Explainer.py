
class Explainer:
    def __init__(self, values, base_values, data, feature_names):
        self.base_values = base_values
        self.data = data
        self.feature_names = feature_names
        self.values = values
