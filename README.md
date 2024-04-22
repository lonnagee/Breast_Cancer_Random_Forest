This project uses Spark to analyze features of breast cancer masses in order to differentiate healthy from cancerous tissue.  The analysis was set up to allow streaming data (imagine live results fed in from hospitals or surgeons), in order to provide real-time updates to the model and feedback to pathologists and oncologists.

# University of Wisconsin Breast Cancer Random Forest
This analysis seeks to use PySpark to analyze a data set of features of breast masses. The features were documented based on images from the fine needle aspiration of the mass, and desribe the cell nucleus. The outcome label, located in the second column, is B - Benign, or M - malignant. The data was donated by the University of Wisconsin and is located on the UCI Machine Learning Repository (https://archive.ics.uci.edu/datasets) and listed under "Breast Cancer Wisconsin (Diagnostic)."
Below are the fields in the data set:
1.	ID number
2.	Diagnosis (M = malignant, B = benign)
3.	through 32. Ten real-valued features are computed for each cell nucleus (each one list listed 3 times):
 - a) radius (mean of distances from center to points on the perimeter)
 - b) texture (standard deviation of gray-scale values)
 - c) perimeter
 - d) area
 - e) smoothness (local variation in radius lengths)
 - f) compactness (perimeter^2 / area - 1.0)
 - g) concavity (severity of concave portions of the contour)
 - h) concave points (number of concave portions of the contour)
 - i) symmetry 
 - j) fractal dimension ("coastline approximation" - 1)
  
Cancer touches so many, and robs many families of years together. As data scientists and researchers, helping to refine early identification and removal of cancer can be lifesaving for people. Further, accurate identification of benign and malignant tumors can be a life or death situation, and the ability to provide predictive analytics/confirmational support for human pathologists could be beneficial.

## Load and Examine Data
The first step for our analysis is to load in the data set. Before we do this, we need to indicate the schema for our file so that we have each feature properly labeled. As noted above, our data has many dimensions and features and a schema ensures we have data accurately loaded and can best identify the most important features.  We are using Databricks to perform our analysis so the data is loaded into a DataFrame that functions much like a database.
 ![Picture1](https://github.com/lonnagee/Breast_Cancer_Random_Forest/assets/136399598/061f93a4-8d95-43e2-bfc0-57c21a4e53c2)

As seen above, the many features in the data are visible and the display is somewhat problematic for examining the actual features.  The UCI Machine Learning Repository documentation indicated that all the values are floats (numbers with decimal points), which we indicated when we loaded our data, however it also indicated that there were no null values so no cleanup or modification is necessary.  Before we fit to our model, we created a training and test set out of the data.  We used a random split into 70% training and 30% testing in order to ensure that sequential data collected did not influence the model.  The training set had 392 rows of data and the test set had 177.  
## Create Pipeline and Fit to Data
The next thing we need to do is create a pipeline to prepare our data for fitting our model, and then finally pass it to our model. A pipeline allows sequential modification of the data in order to ensure that the same steps are completed each time on each row of data. Creating a pipeline allows us to also ensure that the training and test sets are treated the same. The first step in our pipeline is a StringIndexer, which takes our diagnosis column (containing M for malignant or B for benign and converts it to a numerical representation that our model can use, in this case, it assigns indexes 0 and 1 depending on the value in a given row) and places this value into a new column titled "label." The next step in our pipeline is a VectorAssembler. This function takes the input features of each row and creates a vector version of them, and places that value in a new column titled "features." Both of these steps are necessary to prepare the data for our model. After data preparation, our pipeline ends in our model, and at this point our data is fit to the model. In order for our model to fit the data, we must create the model variable and indicate what columns will be used for the label, features, and number of trees. The model I have chosen is a Random Forest Classifier. I chose this model as it is a good model for high dimensional data. A random forest creates many decision trees and the importance of the feature from each tree is averaged to provide the final feature importance. The parameter, numTrees, tells the model how many trees to assemble. A decision trees starts with a single data feature and uses it to split the data based on whether evaluation of the feature is True or False, and iterates many times through the features. much like a flow chart. An example with our data for a first node split would be whether or not radius1 is greater than or less than a certain value. At the next node, it would evaluate another feature, and on and on. As noted, a random forest is an aggregation of many decision trees, and the average importance of a feature is used in the final evaluation of data.
Below we create our indexer for our label (outcome) column, create our vector of features for input into our model, instantiate our model, and finally pass out training data to our pipeline, fitting it to our model.
 ![Picture2](https://github.com/lonnagee/Breast_Cancer_Random_Forest/assets/136399598/c8fc6f3b-2404-4455-9c9f-c9b741251fdc)

The final step in the code creates a model, which we can use to transform our testing data and make predictions for comparison against our known values.  We can do this with our testing set (177 rows referenced above) as a single set, or we can create a streaming analysis which continuously provides output feedback as the sample information arrives.  Creating and executing with this methodology would allow near real-time feedback to oncologists or pathologists.

## Streaming Test Data Analysis
With our random forest classifier fit to our training data, we can now address our test data. In order to view the training data as historical data which can be used as a predictor of new data as it arrives, we need to incorporate streaming into the analysis (think of all the biopsies occurring at each oncology practice or hospital throughout the world being documented and fed into the model in real time.)
We can simulate streaming data for our test set by separating the data into many partitions (subsets) of data at a time and writing them as separate files.  Then, in order to analyze them, we read in each smaller file as if only that portion of data had arrived.  Below is the code to execute the streaming analysis as well as an output of the first 20 rows.
 
![Picture3](https://github.com/lonnagee/Breast_Cancer_Random_Forest/assets/136399598/e083ec8a-3c90-4004-8e9e-7fae7e0417e0)

![Picture4](https://github.com/lonnagee/Breast_Cancer_Random_Forest/assets/136399598/34609448-4aa3-4714-ad7d-95b2ff9f7c08)

The “prediction” column is the prediction our model made for that row of data, with the “label” column being our binarized known outcome.  Looking at the rows displayed, only one has an incorrect prediction.

## Model Evaluation and Conclusion
We fit our model to our test data, both in a static single fit of all of the test data, and in a streaming query, where the continuous input of data was used to make predictions on each row as it arrived. To evaluate the overall Random Forest Classifier performance on the test data set, I am using a binary classification evaluator. It takes the known label column (Malignant or Benign from the original dataset) and the prediction column from our fitted test data output and compares them. In classification models such as ours, the Receiver Operating Characteristic, which compares true positives and false negatives using the known values and predictions. Next, it computes the area under the curve of these values, which can range from 0 to 1. 1 is an ideal value.
Based on our ROC AUC value of approximately 0.95, our dataset and Random Forest Classifier model were a good predictor for our test data. The model could be further optimized to improve the ROC AUC even more, however our focus was on creating a streaming model that could provide "real-time" feedback to surgeons and pathologists. The main challenge with this dataset was the number of features and attempting to review output given the way they displayed. Further optimization of the method could be done via dimensionality reduction, which could help with this, and the evaluation of the model could be done in a streaming step as well, however depending on the use of the model (machine learning prediction prior to pathologist's final report or machine learning confirmation after final report), labels may not be known.

