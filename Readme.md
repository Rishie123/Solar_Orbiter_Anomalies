## ESA's Solar Orbiter

Deployed project link:[Dashboard](https://my-render-jh3k.onrender.com/)
 

This is a project to analyze data from ESA's Solar Orbiter for a period of 2 years, from *1st of January 2022* to *1st of January 2024*.

All the data needed has already been provided for the convenience of the user in the Data folder as `Solar_Orbiter.csv`.

### Primary Tasks

There are two primary tasks which I want to perform:
1. Build a Dashboard of all the instruments' behavior over time.
2. Detect anomalies in the data to understand on which dates the spacecraft was doing something interesting.

### Setup Instructions

To get started, ensure you have Python3 installed on your system. You can download the latest stable version from here: 
[Python Download](https://www.python.org/downloads/)

**To quickly run everything and see main results, run:**
- `python3 Python_Scripts/run_ml_models.py`
- `python3 Python_Scripts/dashboard.py`

### Environment Setup

1. **Build a virtual environment (optional)** - This will help keep your project isolated from other python modules on your device.
   - Make a Virtual Environment by typing this in your console:
     ```
     python3 -m venv my_project
     ```
   - Activate it whenever you want access to the project by typing:
     ```
     source my_project/bin/activate
     ```
   - *Change `my_project` with whatever name you want to give it.*

2. **Install all the dependencies of the project** by typing this in your console:
**pip install -r Setup_Files/Requirements.txt**


### Understanding the Data

Open the Data Folder. It has `Solar_Orbiter.csv`. This file contains, per day, mean values of:
- Radial Distance from Sun (AU)
- Electronic Box Temperature (DegC)
- Out Board Sensor Temperature (DegC)
- In Board Sensor Temperature (DegC)
- Search Coil Magnetometers Temperature (DegC)
- Solar Array Angle (Deg)
- High Gain Antenna azimuth (Deg)

### Detecting Anomalies

**TO KEEP THIS BRIEF(ER), I HAVE PROVIDED MORE DETAILED EXPLANATIONS IN CODE COMMENTS**

1. **Open the Python_Scripts folder.**
- Here, you will find a file called `run_ml_models.py`.
- Running this in the terminal will detect all the anomalies within the dataset using the Isolation Forest model.

2. **The Isolation Forest algorithm** is an unsupervised learning algorithm for anomaly detection that works by:
- Randomly selecting a feature and a split value between the maximum and minimum values of that feature.
- Repeating this process recursively to create a tree-like structure.
- Anomalies are isolated in the tree with a shorter path length, i.e., fewer splits.

References:
- [Original Paper](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)

### Understanding Anomalies with SHAP

1. **SHAP values** are used to explain the decisions of the Isolation Forest model.
- SHAP (SHapley Additive exPlanations) values derive from game theory and provide insights into the contribution of each feature to a specific prediction made by the model.
- These will be calculated upon running the `run_ml_models.py` file.

2. **Visualization**: 
- A visualization for the mean absolute value of SHAP values to get feature importance is created and stored in the Explainability folder, based on section 9.6.5 of textbook of Interpretable ML Book

References:
- [Interpretable ML Book - SHAP](https://christophm.github.io/interpretable-ml-book/shap.html) (Section 9.6.5 SHAP Feature Importance, Section 9.6 SHAP)
- [PyData Conference on SHAP](https://www.youtube.com/watch?v=5p8B2Ikcw-k) - Tel Aviv
- [SHAP Documentation](https://shap.readthedocs.io/en/latest/)

3. **Output**:  
- We will have Solar_Orbiter_with_Anomalies.csv within the Data folder saved ( This contains the original database with anomaly scores explained in code)
- We will have shap_values_plot.html saved in Python_Scripts/Explainability, containing the visualisation of Feature importance

### Dashboarding the Data

1. **Local run** - The dashboard, can be run on a local server on your own system by running the **dashboard.py file** within the Python_Scripts folder
- The Dashboard consists of 4 key visualisations, 
- Time Series Chart: This shows how each of the features varies over time and gives us insights about how the data looks overall
- Correlation Heatmap: This calculates correlation coefficient between several features and displays it in the form of a heatmap. Interestingly,
Solar Array Angle is highly correlated to Radial Distance from the sun. This is because, the Solar Arrays change their angle to point in the direction of the sun.
- Anomaly Score Chart: This is used to find the anomalous dates within the spacecraft. Lower the score, more anomalous the date. Interestingly,
5 May to 11 May are identified as anomalous dates by the model. This is consistent with the fact that the spacecraft was having high noise period around that time ( https://www.cosmos.esa.int/web/soar/support-data )
- Feature Importance Plot: This is simply embedded from the explainability folder using the Dash HTML component Iframe

2. **Deploying using Render** - I have deployed the dashboard on the web using render, largely by following a tutorial
- Please follow this tutorial for doing the same https://www.youtube.com/watch?v=XWJBJoV5yww&t=0s
- For the same, you will find the entire Dashboard named as app and all the needed things within src folder in the DeployWithRender folder 
- Copy the DeployWithRender directory and open it in a separate project to avoid nested git repositories
- Ensure you have dash-tools installed, it is their in requirements.txt ( so I am assuming it is installed or do pip install dash-tools)
- type "dashtools gui" in terminal
- Go to Deploy section on the newly opened page 
- Open your file there, by putting the path of your folder in the text box
- Follow the instruction further in the tutorial and you will be able to deploy it, just like this: https://my-render-jh3k.onrender.com/

## Scalability

1. Memory Profiling:  

- We use the memoryprofiler library to do memory profiling
- The results are stored in Scalability/memory_profiling
- From the results, it can be seen that within the dashboard, every line involves about 120 Mib of memory while callback requires 120 Mib recurrently
- Also, within the run_ml_models.py calculating shap values and fitting the model are the most memory intensive tasks
- Interestingly, as seen in Scalability/Plot_ML_Model, there is a growth and decline in memory usage for run_ml_models but, no decline for dashboard.

- Reference: https://pypi.org/project/memory-profiler/  #
- Reference: https://github.com/pythonprofilers/memory_profiler


2. Time Profiling:

- We use the cProfile package for doing time profiling
- The results are stored in Scalability/time_profiling
- To reproduce, simply follow the instructions at the bottom of the code for dashboard.py and run_ml_models.py
- you can interpret the results using snakeviz as mentioned there
- It shows the time required to load the dashboard completely is 23.22seconds, along with breakdown of time required by different components
- It shows the time required to run the model and get the shaply values with visualisation is 3.56 seconds with breakdown

- Reference: https://docs.python.org/3/library/profile.html