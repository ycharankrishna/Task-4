# Task-4

# linear regression

What Is Regression?
Regression searches for relationships among variables.

For example, you can observe several employees of some company and try to understand how their salaries depend on the features, such as experience, level of education, role, city they work in, and so on.

This is a regression problem where data related to each employee represent one observation. The presumption is that the experience, education, role, and city are the independent features, while the salary depends on them.

Similarly, you can try to establish a mathematical dependence of the prices of houses on their areas, numbers of bedrooms, distances to the city center, and so on.

Generally, in regression analysis, you usually consider some phenomenon of interest and have a number of observations. Each observation has two or more features. Following the assumption that (at least) one of the features depends on the others, you try to establish a relation among them.

In other words, you need to find a function that maps some features or variables to others sufficiently well.

The dependent features are called the dependent variables, outputs, or responses.

The independent features are called the independent variables, inputs, or predictors.

Regression problems usually have one continuous and unbounded dependent variable. The inputs, however, can be continuous, discrete, or even categorical data such as gender, nationality, brand, and so on.

It is a common practice to denote the outputs with ๐ฆ and inputs with ๐ฅ. If there are two or more independent variables, they can be represented as the vector ๐ฑ = (๐ฅโ, โฆ, ๐ฅแตฃ), where ๐ is the number of inputs.

When Do You Need Regression?
Typically, you need regression to answer whether and how some phenomenon influences the other or how several variables are related. For example, you can use it to determine if and to what extent the experience or gender impact salaries.

Regression is also useful when you want to forecast a response using a new set of predictors. For example, you could try to predict electricity consumption of a household for the next hour given the outdoor temperature, time of day, and number of residents in that household.

Regression is used in many different fields: economy, computer science, social sciences, and so on. Its importance rises every day with the availability of large amounts of data and increased awareness of the practical value of data.

Linear Regression
Linear regression is probably one of the most important and widely used regression techniques. Itโs among the simplest regression methods. One of its main advantages is the ease of interpreting results.

Problem Formulation
When implementing linear regression of some dependent variable ๐ฆ on the set of independent variables ๐ฑ = (๐ฅโ, โฆ, ๐ฅแตฃ), where ๐ is the number of predictors, you assume a linear relationship between ๐ฆ and ๐ฑ: ๐ฆ = ๐ฝโ + ๐ฝโ๐ฅโ + โฏ + ๐ฝแตฃ๐ฅแตฃ + ๐. This equation is the regression equation. ๐ฝโ, ๐ฝโ, โฆ, ๐ฝแตฃ are the regression coefficients, and ๐ is the random error.

Linear regression calculates the estimators of the regression coefficients or simply the predicted weights, denoted with ๐โ, ๐โ, โฆ, ๐แตฃ. They define the estimated regression function ๐(๐ฑ) = ๐โ + ๐โ๐ฅโ + โฏ + ๐แตฃ๐ฅแตฃ. This function should capture the dependencies between the inputs and output sufficiently well.

The estimated or predicted response, ๐(๐ฑแตข), for each observation ๐ = 1, โฆ, ๐, should be as close as possible to the corresponding actual response ๐ฆแตข. The differences ๐ฆแตข - ๐(๐ฑแตข) for all observations ๐ = 1, โฆ, ๐, are called the residuals. Regression is about determining the best predicted weights, that is the weights corresponding to the smallest residuals.

To get the best weights, you usually minimize the sum of squared residuals (SSR) for all observations ๐ = 1, โฆ, ๐: SSR = ฮฃแตข(๐ฆแตข - ๐(๐ฑแตข))ยฒ. This approach is called the method of ordinary least squares.
Regression Performance
The variation of actual responses ๐ฆแตข, ๐ = 1, โฆ, ๐, occurs partly due to the dependence on the predictors ๐ฑแตข. However, there is also an additional inherent variance of the output.

The coefficient of determination, denoted as ๐ยฒ, tells you which amount of variation in ๐ฆ can be explained by the dependence on ๐ฑ using the particular regression model. Larger ๐ยฒ indicates a better fit and means that the model can better explain the variation of the output with different inputs.

The value ๐ยฒ = 1 corresponds to SSR = 0, that is to the perfect fit since the values of predicted and actual responses fit completely to each other.

Simple Linear Regression
Simple or single-variate linear regression is the simplest case of linear regression with a single independent variable, ๐ฑ = ๐ฅ.

The following figure illustrates simple linear regression:
![image](https://user-images.githubusercontent.com/75231800/142756394-a9bcbb62-c27e-42b3-856a-262e72f1f4b3.png)
When implementing simple linear regression, you typically start with a given set of input-output (๐ฅ-๐ฆ) pairs (green circles). These pairs are your observations. For example, the leftmost observation (green circle) has the input ๐ฅ = 5 and the actual output (response) ๐ฆ = 5. The next one has ๐ฅ = 15 and ๐ฆ = 20, and so on.

The estimated regression function (black line) has the equation ๐(๐ฅ) = ๐โ + ๐โ๐ฅ. Your goal is to calculate the optimal values of the predicted weights ๐โ and ๐โ that minimize SSR and determine the estimated regression function. The value of ๐โ, also called the intercept, shows the point where the estimated regression line crosses the ๐ฆ axis. It is the value of the estimated response ๐(๐ฅ) for ๐ฅ = 0. The value of ๐โ determines the slope of the estimated regression line.

The predicted responses (red squares) are the points on the regression line that correspond to the input values. For example, for the input ๐ฅ = 5, the predicted response is ๐(5) = 8.33 (represented with the leftmost red square).

The residuals (vertical dashed gray lines) can be calculated as ๐ฆแตข - ๐(๐ฑแตข) = ๐ฆแตข - ๐โ - ๐โ๐ฅแตข for ๐ = 1, โฆ, ๐. They are the distances between the green circles and red squares. When you implement linear regression, you are actually trying to minimize these distances and make the red squares as close to the predefined green circles as possible.

Multiple Linear Regression
Multiple or multivariate linear regression is a case of linear regression with two or more independent variables.

If there are just two independent variables, the estimated regression function is ๐(๐ฅโ, ๐ฅโ) = ๐โ + ๐โ๐ฅโ + ๐โ๐ฅโ. It represents a regression plane in a three-dimensional space. The goal of regression is to determine the values of the weights ๐โ, ๐โ, and ๐โ such that this plane is as close as possible to the actual responses and yield the minimal SSR.

The case of more than two independent variables is similar, but more general. The estimated regression function is ๐(๐ฅโ, โฆ, ๐ฅแตฃ) = ๐โ + ๐โ๐ฅโ + โฏ +๐แตฃ๐ฅแตฃ, and there are ๐ + 1 weights to be determined when the number of inputs is ๐.

Polynomial Regression
You can regard polynomial regression as a generalized case of linear regression. You assume the polynomial dependence between the output and inputs and, consequently, the polynomial estimated regression function.

In other words, in addition to linear terms like ๐โ๐ฅโ, your regression function ๐ can include non-linear terms such as ๐โ๐ฅโยฒ, ๐โ๐ฅโยณ, or even ๐โ๐ฅโ๐ฅโ, ๐โ๐ฅโยฒ๐ฅโ, and so on.

The simplest example of polynomial regression has a single independent variable, and the estimated regression function is a polynomial of degree 2: ๐(๐ฅ) = ๐โ + ๐โ๐ฅ + ๐โ๐ฅยฒ.

Now, remember that you want to calculate ๐โ, ๐โ, and ๐โ, which minimize SSR. These are your unknowns!

Keeping this in mind, compare the previous regression function with the function ๐(๐ฅโ, ๐ฅโ) = ๐โ + ๐โ๐ฅโ + ๐โ๐ฅโ used for linear regression. They look very similar and are both linear functions of the unknowns ๐โ, ๐โ, and ๐โ. This is why you can solve the polynomial regression problem as a linear problem with the term ๐ฅยฒ regarded as an input variable.

In the case of two variables and the polynomial of degree 2, the regression function has this form: ๐(๐ฅโ, ๐ฅโ) = ๐โ + ๐โ๐ฅโ + ๐โ๐ฅโ + ๐โ๐ฅโยฒ + ๐โ๐ฅโ๐ฅโ + ๐โ๐ฅโยฒ. The procedure for solving the problem is identical to the previous case. You apply linear regression for five inputs: ๐ฅโ, ๐ฅโ, ๐ฅโยฒ, ๐ฅโ๐ฅโ, and ๐ฅโยฒ. What you get as the result of regression are the values of six weights which minimize SSR: ๐โ, ๐โ, ๐โ, ๐โ, ๐โ, and ๐โ.

Of course, there are more general problems, but this should be enough to illustrate the point.
Underfitting and Overfitting
One very important question that might arise when youโre implementing polynomial regression is related to the choice of the optimal degree of the polynomial regression function.

There is no straightforward rule for doing this. It depends on the case. You should, however, be aware of two problems that might follow the choice of the degree: underfitting and overfitting.

Underfitting occurs when a model canโt accurately capture the dependencies among data, usually as a consequence of its own simplicity. It often yields a low ๐ยฒ with known data and bad generalization capabilities when applied with new data.

Overfitting happens when a model learns both dependencies among data and random fluctuations. In other words, a model learns the existing data too well. Complex models, which have many features or terms, are often prone to overfitting. When applied to known data, such models usually yield high ๐ยฒ. However, they often donโt generalize well and have significantly lower ๐ยฒ when used with new data.

The next figure illustrates the underfitted, well-fitted, and overfitted models:
![image](https://user-images.githubusercontent.com/75231800/142756417-ea0ce123-5746-43f9-95fc-23d89877e734.png)
The top left plot shows a linear regression line that has a low ๐ยฒ. It might also be important that a straight line canโt take into account the fact that the actual response increases as ๐ฅ moves away from 25 towards zero. This is likely an example of underfitting.

The top right plot illustrates polynomial regression with the degree equal to 2. In this instance, this might be the optimal degree for modeling this data. The model has a value of ๐ยฒ that is satisfactory in many cases and shows trends nicely.

The bottom left plot presents polynomial regression with the degree equal to 3. The value of ๐ยฒ is higher than in the preceding cases. This model behaves better with known data than the previous ones. However, it shows some signs of overfitting, especially for the input values close to 60 where the line starts decreasing, although actual data donโt show that.

Finally, on the bottom right plot, you can see the perfect fit: six points and the polynomial line of the degree 5 (or higher) yield ๐ยฒ = 1. Each actual response equals its corresponding prediction.

In some situations, this might be exactly what youโre looking for. In many cases, however, this is an overfitted model. It is likely to have poor behavior with unseen data, especially with the inputs larger than 50.

For example, it assumes, without any evidence, that there is a significant drop in responses for ๐ฅ > 50 and that ๐ฆ reaches zero for ๐ฅ near 60. Such behavior is the consequence of excessive effort to learn and fit the existing data.
Implementing Linear Regression in Python
Python Packages for Linear Regression
Import the packages and classes you need.
Provide data to work with and eventually do appropriate transformations.
Create a regression model and fit it with existing data.
Check the results of model fitting to know whether the model is satisfactory.
Apply the model for predictions.

# logistic regression
What Is Classification?
Supervised machine learning algorithms define models that capture relationships among data. Classification is an area of supervised machine learning that tries to predict which class or category some entity belongs to, based on its features.

For example, you might analyze the employees of some company and try to establish a dependence on the features or variables, such as the level of education, number of years in a current position, age, salary, odds for being promoted, and so on. The set of data related to a single employee is one observation. The features or variables can take one of two forms:

Independent variables, also called inputs or predictors, donโt depend on other features of interest (or at least you assume so for the purpose of the analysis).
Dependent variables, also called outputs or responses, depend on the independent variables.
In the above example where youโre analyzing employees, you might presume the level of education, time in a current position, and age as being mutually independent, and consider them as the inputs. The salary and the odds for promotion could be the outputs that depend on the inputs.

Note: Supervised machine learning algorithms analyze a number of observations and try to mathematically express the dependence between the inputs and outputs. These mathematical representations of dependencies are the models.

The nature of the dependent variables differentiates regression and classification problems. Regression problems have continuous and usually unbounded outputs. An example is when youโre estimating the salary as a function of experience and education level. On the other hand, classification problems have discrete and finite outputs called classes or categories. For example, predicting if an employee is going to be promoted or not (true or false) is a classification problem.

There are two main types of classification problems:

Binary or binomial classification: exactly two classes to choose between (usually 0 and 1, true and false, or positive and negative)
Multiclass or multinomial classification: three or more classes of the outputs to choose from
If thereโs only one input variable, then itโs usually denoted with ๐ฅ. For more than one input, youโll commonly see the vector notation ๐ฑ = (๐ฅโ, โฆ, ๐ฅแตฃ), where ๐ is the number of the predictors (or independent features). The output variable is often denoted with ๐ฆ and takes the values 0 or 1.

When Do You Need Classification?
You can apply classification in many fields of science and technology. For example, text classification algorithms are used to separate legitimate and spam emails, as well as positive and negative comments. You can check out Practical Text Classification With Python and Keras to get some insight into this topic. Other examples involve medical applications, biological classification, credit scoring, and more.

Image recognition tasks are often represented as classification problems. For example, you might ask if an image is depicting a human face or not, or if itโs a mouse or an elephant, or which digit from zero to nine it represents, and so on. To learn more about this, check out Traditional Face Detection With Python and Face Recognition with Python, in Under 25 Lines of Code.

Logistic Regression Overview
Logistic regression is a fundamental classification technique. It belongs to the group of linear classifiers and is somewhat similar to polynomial and linear regression. Logistic regression is fast and relatively uncomplicated, and itโs convenient for you to interpret the results. Although itโs essentially a method for binary classification, it can also be applied to multiclass problems.

Math Prerequisites
Youโll need an understanding of the sigmoid function and the natural logarithm function to understand what logistic regression is and how it works.

This image shows the sigmoid function (or S-shaped curve) of some variable ๐ฅ:
![image](https://user-images.githubusercontent.com/75231800/142756564-c9791a33-524e-46c3-bb60-e2a467fa2b55.png)
The sigmoid function has values very close to either 0 or 1 across most of its domain. This fact makes it suitable for application in classification methods.

This image depicts the natural logarithm log(๐ฅ) of some variable ๐ฅ, for values of ๐ฅ between 0 and 1:
![image](https://user-images.githubusercontent.com/75231800/142756573-0dab077f-de34-4bd9-a12e-8f1cccd918c5.png)
As ๐ฅ approaches zero, the natural logarithm of ๐ฅ drops towards negative infinity. When ๐ฅ = 1, log(๐ฅ) is 0. The opposite is true for log(1 โ ๐ฅ).

Note that youโll often find the natural logarithm denoted with ln instead of log. In Python, math.log(x) and numpy.log(x) represent the natural logarithm of x, so youโll follow this notation in this tutorial.
Problem Formulation
In this tutorial, youโll see an explanation for the common case of logistic regression applied to binary classification. When youโre implementing the logistic regression of some dependent variable ๐ฆ on the set of independent variables ๐ฑ = (๐ฅโ, โฆ, ๐ฅแตฃ), where ๐ is the number of predictors ( or inputs), you start with the known values of the predictors ๐ฑแตข and the corresponding actual response (or output) ๐ฆแตข for each observation ๐ = 1, โฆ, ๐.

Your goal is to find the logistic regression function ๐(๐ฑ) such that the predicted responses ๐(๐ฑแตข) are as close as possible to the actual response ๐ฆแตข for each observation ๐ = 1, โฆ, ๐. Remember that the actual response can be only 0 or 1 in binary classification problems! This means that each ๐(๐ฑแตข) should be close to either 0 or 1. Thatโs why itโs convenient to use the sigmoid function.

Once you have the logistic regression function ๐(๐ฑ), you can use it to predict the outputs for new and unseen inputs, assuming that the underlying mathematical dependence is unchanged.

Methodology
Logistic regression is a linear classifier, so youโll use a linear function ๐(๐ฑ) = ๐โ + ๐โ๐ฅโ + โฏ + ๐แตฃ๐ฅแตฃ, also called the logit. The variables ๐โ, ๐โ, โฆ, ๐แตฃ are the estimators of the regression coefficients, which are also called the predicted weights or just coefficients.

The logistic regression function ๐(๐ฑ) is the sigmoid function of ๐(๐ฑ): ๐(๐ฑ) = 1 / (1 + exp(โ๐(๐ฑ)). As such, itโs often close to either 0 or 1. The function ๐(๐ฑ) is often interpreted as the predicted probability that the output for a given ๐ฑ is equal to 1. Therefore, 1 โ ๐(๐ฅ) is the probability that the output is 0.

Logistic regression determines the best predicted weights ๐โ, ๐โ, โฆ, ๐แตฃ such that the function ๐(๐ฑ) is as close as possible to all actual responses ๐ฆแตข, ๐ = 1, โฆ, ๐, where ๐ is the number of observations. The process of calculating the best weights using available observations is called model training or fitting.

To get the best weights, you usually maximize the log-likelihood function (LLF) for all observations ๐ = 1, โฆ, ๐. This method is called the maximum likelihood estimation and is represented by the equation LLF = ฮฃแตข(๐ฆแตข log(๐(๐ฑแตข)) + (1 โ ๐ฆแตข) log(1 โ ๐(๐ฑแตข))).

When ๐ฆแตข = 0, the LLF for the corresponding observation is equal to log(1 โ ๐(๐ฑแตข)). If ๐(๐ฑแตข) is close to ๐ฆแตข = 0, then log(1 โ ๐(๐ฑแตข)) is close to 0. This is the result you want. If ๐(๐ฑแตข) is far from 0, then log(1 โ ๐(๐ฑแตข)) drops significantly. You donโt want that result because your goal is to obtain the maximum LLF. Similarly, when ๐ฆแตข = 1, the LLF for that observation is ๐ฆแตข log(๐(๐ฑแตข)). If ๐(๐ฑแตข) is close to ๐ฆแตข = 1, then log(๐(๐ฑแตข)) is close to 0. If ๐(๐ฑแตข) is far from 1, then log(๐(๐ฑแตข)) is a large negative number.

There are several mathematical approaches that will calculate the best weights that correspond to the maximum LLF, but thatโs beyond the scope of this tutorial. For now, you can leave these details to the logistic regression Python libraries youโll learn to use here!

Once you determine the best weights that define the function ๐(๐ฑ), you can get the predicted outputs ๐(๐ฑแตข) for any given input ๐ฑแตข. For each observation ๐ = 1, โฆ, ๐, the predicted output is 1 if ๐(๐ฑแตข) > 0.5 and 0 otherwise. The threshold doesnโt have to be 0.5, but it usually is. You might define a lower or higher value if thatโs more convenient for your situation.

Thereโs one more important relationship between ๐(๐ฑ) and ๐(๐ฑ), which is that log(๐(๐ฑ) / (1 โ ๐(๐ฑ))) = ๐(๐ฑ). This equality explains why ๐(๐ฑ) is the logit. It implies that ๐(๐ฑ) = 0.5 when ๐(๐ฑ) = 0 and that the predicted output is 1 if ๐(๐ฑ) > 0 and 0 otherwise.

Classification Performance
Binary classification has four possible types of results:

True negatives: correctly predicted negatives (zeros)
True positives: correctly predicted positives (ones)
False negatives: incorrectly predicted negatives (zeros)
False positives: incorrectly predicted positives (ones)
You usually evaluate the performance of your classifier by comparing the actual and predicted outputsand counting the correct and incorrect predictions.

The most straightforward indicator of classification accuracy is the ratio of the number of correct predictions to the total number of predictions (or observations). Other indicators of binary classifiers include the following:

The positive predictive value is the ratio of the number of true positives to the sum of the numbers of true and false positives.
The negative predictive value is the ratio of the number of true negatives to the sum of the numbers of true and false negatives.
The sensitivity (also known as recall or true positive rate) is the ratio of the number of true positives to the number of actual positives.
The specificity (or true negative rate) is the ratio of the number of true negatives to the number of actual negatives.
The most suitable indicator depends on the problem of interest. In this tutorial, youโll use the most straightforward form of classification accuracy.
Single-Variate Logistic Regression
Single-variate logistic regression is the most straightforward case of logistic regression. There is only one independent variable (or feature), which is ๐ฑ = ๐ฅ. This figure illustrates single-variate logistic regression:
![image](https://user-images.githubusercontent.com/75231800/142756595-32a72450-962f-40b0-bd4c-1fd4b4477f17.png)
Here, you have a given set of input-output (or ๐ฅ-๐ฆ) pairs, represented by green circles. These are your observations. Remember that ๐ฆ can only be 0 or 1. For example, the leftmost green circle has the input ๐ฅ = 0 and the actual output ๐ฆ = 0. The rightmost observation has ๐ฅ = 9 and ๐ฆ = 1.

Logistic regression finds the weights ๐โ and ๐โ that correspond to the maximum LLF. These weights define the logit ๐(๐ฅ) = ๐โ + ๐โ๐ฅ, which is the dashed black line. They also define the predicted probability ๐(๐ฅ) = 1 / (1 + exp(โ๐(๐ฅ))), shown here as the full black line. In this case, the threshold ๐(๐ฅ) = 0.5 and ๐(๐ฅ) = 0 corresponds to the value of ๐ฅ slightly higher than 3. This value is the limit between the inputs with the predicted outputs of 0 and 1.
Multi-Variate Logistic Regression
Multi-variate logistic regression has more than one input variable. This figure shows the classification with two independent variables, ๐ฅโ and ๐ฅโ:
![image](https://user-images.githubusercontent.com/75231800/142756604-223954d6-f691-46cf-8123-74dd370606af.png)
The graph is different from the single-variate graph because both axes represent the inputs. The outputs also differ in color. The white circles show the observations classified as zeros, while the green circles are those classified as ones.

Logistic regression determines the weights ๐โ, ๐โ, and ๐โ that maximize the LLF. Once you have ๐โ, ๐โ, and ๐โ, you can get:

The logit ๐(๐ฅโ, ๐ฅโ) = ๐โ + ๐โ๐ฅโ + ๐โ๐ฅโ
The probabilities ๐(๐ฅโ, ๐ฅโ) = 1 / (1 + exp(โ๐(๐ฅโ, ๐ฅโ)))
The dash-dotted black line linearly separates the two classes. This line corresponds to ๐(๐ฅโ, ๐ฅโ) = 0.5 and ๐(๐ฅโ, ๐ฅโ) = 0.

Regularization
Overfitting is one of the most serious kinds of problems related to machine learning. It occurs when a model learns the training data too well. The model then learns not only the relationships among data but also the noise in the dataset. Overfitted models tend to have good performance with the data used to fit them (the training data), but they behave poorly with unseen data (or test data, which is data not used to fit the model).

Overfitting usually occurs with complex models. Regularization normally tries to reduce or penalize the complexity of the model. Regularization techniques applied with logistic regression mostly tend to penalize large coefficients ๐โ, ๐โ, โฆ, ๐แตฃ:

L1 regularization penalizes the LLF with the scaled sum of the absolute values of the weights: |๐โ|+|๐โ|+โฏ+|๐แตฃ|.
L2 regularization penalizes the LLF with the scaled sum of the squares of the weights: ๐โยฒ+๐โยฒ+โฏ+๐แตฃยฒ.
Elastic-net regularization is a linear combination of L1 and L2 regularization.
Regularization can significantly improve model performance on unseen data.

Logistic Regression in Python
Now that you understand the fundamentals, youโre ready to apply the appropriate packages as well as their functions and classes to perform logistic regression in Python. In this section, youโll see the following:

A summary of Python packages for logistic regression (NumPy, scikit-learn, StatsModels, and Matplotlib)
Two illustrative examples of logistic regression solved with scikit-learn
One conceptual example solved with StatsModels
One real-world example of classifying handwritten digits
