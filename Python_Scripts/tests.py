import unittest
import pandas as pd
import shap
from sklearn.ensemble import IsolationForest
from ..Helpers import load_data, save_results, fit_isolation_forest

class TestHelpers(unittest.TestCase):
    def test_load_data(self):
        # This test ensures that the load_data function correctly loads data and returns a pandas DataFrame.
        df = load_data('path_to_output_csv.csv')
        self.assertIsInstance(df, pd.DataFrame)

    def test_save_results(self):
        # This test checks that the save_results function can save a DataFrame without issues.
        df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        save_results(df, 'path_to_output_csv.csv')

    def test_fit_isolation_forest(self):
        # This test verifies that the fit_isolation_forest function successfully fits an Isolation Forest model to the data.
        df = pd.DataFrame({'feature1': [0, 1, 0, 1], 'feature2': [1, 0, 1, 0]})
        model = fit_isolation_forest(df)
        self.assertIsNotNone(model)

    def test_calculate_shap_values(self):
        # Create a small DataFrame to serve as test input.
        # Small, controlled data helps in predictable testing and easier debugging.
        data = pd.DataFrame({'feature1': [0, 1], 'feature2': [1, 0]})

        # Fit an Isolation Forest model to the created data.
        # We use a simple model fit because we need a trained model to generate SHAP values.
        model = IsolationForest()
        model.fit(data)

        # Invoke the calculate_shap_values function using the test data and the fitted model.
        # This is the function being tested for correctness.
        shap_values_df = calculate_shap_values(data, model)

        # Assert that the result is a DataFrame.
        # Ensuring the output is a DataFrame confirms the return type is as expected.
        self.assertIsInstance(shap_values_df, pd.DataFrame)

        # Assert that the output DataFrame has the same shape as the input data.
        # This confirms that SHAP values are calculated for each feature in each instance.
        self.assertEqual(shap_values_df.shape, data.shape)

        # Optionally, confirm that the column names in the output match those in the input.
        # This ensures that SHAP values are aligned correctly with their corresponding features.
        self.assertListEqual(list(shap_values_df.columns), list(data.columns))

if __name__ == '__main__':
    unittest.main()
