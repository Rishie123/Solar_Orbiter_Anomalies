import pandas as pd
import plotly.express as px
import Helpers as h
import cProfile
from memory_profiler import profile, memory_usage


# @profile  "Uncomment this line to do memory profiling of the code"
" @profile can be used to get the memory usage per line of the code, this has already been performed and stored in the Scalability folder"
"""Reference: https://pypi.org/project/memory-profiler/ 
    You can further run "mprof run --python3 Run_Ml_Models.py" and "mprof plot" on terminal to get the memory usage over time of the code
 These results have been stored in the Scalability folder"""

def main():
    """Main function to calculate SHAP values and detect outliers using an Isolation Forest model."""
    
    # Define file paths for data input and output
    data_path = "../Data/Solar_Orbiter.csv"  # Input file path.
    output_path = "../Data/Solar_Orbiter_With_Anomalies.csv"  # Output file path.
    shap_plot_path = "Explainability/Shap_Values_Plot.html"  # Path to save SHAP values plot, now as HTML for interactivity

    # Loading the dataset
    solar_data = h.load_data(data_path)  # Load data from the CSV file using a helper function.

    # Convert 'Date' column to datetime format and drop it for easir analysis 
    #I am doing this because, it is difficult for Isolation Forest and SHAP to handle datetime data)
    solar_data['Date'] = pd.to_datetime(solar_data['Date'])
    data_for_analysis = solar_data.drop(columns=['Date'])

    """
    The Isolation Forest algorithm is used to detect anomalies in the dataset.
  
    The Isolation Forest algorithm is an unsupervised learning algorithm for anomaly detection.
    It is based on the principle that within decision trees, anomalies are easier to isolate than normal data points.
    The algorithm works by randomly selecting a feature and then randomly selecting a split value between the maximum and minimum values of the selected feature.
    The process is repeated recursively to create a tree-like structure.
    Anomalies are isolated in the tree with a shorter path length, i.e., fewer splits.

     Reference 1: https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf - Original Paper"
     Reference 2: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html - Scikit-learn Documentation"
     Reference 3: https://towardsdatascience.com/outlier-detection-with-isolation-forest-3d190448d45e - Towards Data Science"
     Reference 4: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html
     Reference 5 : https://medium.com/@corymaklin/isolation-forest-799fceacdda4
    
     """
    # Fit Isolation Forest model to detect outliers
    iso_forest_model = h.fit_isolation_forest(data_for_analysis)  # Train the Isolation Forest model on the data.
    # Compute anomaly scores for each instance using the Isolation Forest model
    solar_data['anomaly_score'] = iso_forest_model.decision_function(data_for_analysis)
    
    """  decision_function(X) : Average anomaly score of X of the base classifiers.
         Parameters:
        X{array-like, sparse matrix} of shape (n_samples, n_features)
        The input samples. Internally, it will be converted to dtype=np.float32 and if a sparse matrix is provided to a sparse csr_matrix.

        Returns:
        scoresndarray of shape (n_samples,)
        The anomaly score of the input samples. The lower, the more abnormal. Negative scores represent outliers, positive scores represent inliers.
        Reference: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html"""


    " SHAP values are used to explain the decisions of the Isolation Forest model.They help to understand the contribution of each feature to the anomaly score."

    # Calculate SHAP values to explain the decisions of the Isolation Forest model
    shap_values = h.calculate_shap_values(data_for_analysis, iso_forest_model)


    " This visualisation is inspired by the SHAP summary plot from the SHAP library."
    " It shows the mean absolute SHAP values for each feature in the dataset."
    " The mean absolute SHAP value represents the average impact of a feature on the model output."
    " A higher mean absolute SHAP value indicates a higher impact on the anomaly score."
    " The plot helps to identify the most important features for detecting anomalies."
    " The plot is saved as an interactive HTML file for easy sharing and exploration."

    #The Statistical inferenece for the same can be found here.

    """
    "Features with large absolute Shapley values are important. 
    Since we want the global importance, we average the absolute Shapley 
    values per feature across the data."

    Reference 1: https://christophm.github.io/interpretable-ml-book/shap.html
    Reference 2: https://www.youtube.com/watch?v=5p8B2Ikcw-k - PyData Conference, Tel Aviv
    Reference 3: https://towardsdatascience.com/explain-your-model-with-the-shap-values-bc36aac4de3d"
    Reference 4: https://shap.readthedocs.io/en/latest/ - SHAP Documentation
    """
    # Visualization of SHAP values using Plotly for interactive visualization

    shap_summary = shap_values.abs().mean().sort_values(ascending=False)
    fig = px.bar(shap_summary, x=shap_summary.values, y=shap_summary.index, labels={'x': 'Mean Absolute SHAP Value', 'index': 'Features'}, orientation='h')
    fig.update_layout(title_text='Feature Importance based on SHAP Values', title_x=0.5)
    fig.write_html(shap_plot_path)  # Save the plot as an interactive HTML file

    # Save the updated DataFrame with anomaly scores (excluding SHAP values) to a new CSV file
    solar_data.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")
    print(f"SHAP values plot saved to {shap_plot_path}")

if __name__ == "__main__":
    main()
    """Uncomment this line to do time profiling of the code"

    #cProfile.run('main()', 'Scalability/Time_Profiling/Run_Ml_Models.prof') 

    This has already been performed and stored in the Scalability folder"
     to access the time profiling results,
    run the following command in the terminal:
    
    $ snakeviz Scalability/Time_Profiling/Run_Ml_Models.prof

    (Ensure you have snakeviz installed by running `pip install snakeviz`)

    Reference: https://docs.python.org/3/library/profile.html
    """
 
