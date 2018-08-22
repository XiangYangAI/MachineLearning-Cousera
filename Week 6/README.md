##Advice for Applying Machine Learning
> - How to tell when a learning algorithm is doing poorly
> - Describe the 'best practices' for how to 'debug' your learning algorithm and  go about improving its performance
> - Discuss how to understand the performance of a machine learning system with multiple  pars
> - how to deal with skewed data

### Evaluating a Learning Algorithm

* Debugging a learning algorithm

* Machine learning diagnostic

  * Diagnostic : A test that you can run to gain insight what is/isn't working with a learning algorithm, and gain guidance as to how best to improve its performance.

  * Diagnostics can give guidance as to what might be more fruitful things to try to improve a learning algorithm.

  * Diagnostics can be time-consuming to implement and try, but they can still be a very good use of your time

  * A diagnostic can sometimes rule out certain courses of action (changes to your learning algorithm) as being unlikely to improve its performance significantly.

### Evaluating a hypothesis

* A hypothesis may have a low error for the training examples but still be inaccurate (because of overfitting). Thus, to evaluate a hypothesis, given a dataset of training examples, we can split up the data into two sets: a **training set** and a **test set**. Typically, the training set consists of 70 % of your data and the test set is the remaining 30 %.
* The new procedure using these two sets is then:
  * Learn  $\Theta $ and minimize ${J_{train}}\left( \Theta  \right)$ using the training set
  * Compute the test set error ${J_{test}}\left( \Theta  \right)$

####  The test set error

* For linear regression : ${J_{test}}\left( \Theta  \right) = {1 \over {2{m_{test}}}}\sum\limits_{i = 1}^{{m_{test}}} {{{({h_\Theta }(x_{test}^{(i)}) - y_{test}^{(i)})}^2}} $

* For classification ~ Misclassification error (aka 0/1 misclassification error) :



  $$err({h_\Theta }(x),y) = \left\{ {\matrix{
     1 & {{\rm{ if }}{h_\Theta }(x) \ge 0.5{\rm{ }}and{\rm{ }}y = 0{\rm{ }}or{\rm{ }}{h_\Theta }(x) < 0.5{\rm{ }}and{\rm{ }}y = 1}  \cr 
     0 & {otherwise}  \cr  } } \right.$$



  This gives us a binary 0 or 1 error result based on a misclassification. The average test error for the test set is:

  $Test{\rm{ }}Error = {1 \over {{m_{test}}}}\sum\limits_{i = 1}^{{m_{test}}} {err({h_\Theta }(x_{test}^{(i)}),y_{test}^{(i)})} $

  This gives us the proportion of the test data that was misclassified.

### Model Selection

*  Just because a learning algorithm fits a training set well, that does not mean it is a good hypothesis. It could over fit and as a result your predictions on the test set would be poor. The error of your hypothesis as measured on the data set with which you trained the parameters will be lower than the error on any other data set. 

* Given many models with different polynomial degrees, we can use a systematic approach to identify the `best` function. In order to choose the model of your hypothesis, you can test each degree of polynomial and look at the error result.

* One way to break down our dataset into the three sets is:
  * Training set: 60%
  * Cross validation set: 20%
  * Test set: 20%

* We can now calculate three separate error values for the three different sets using the following method:

  1. Optimize the parameters in $\Theta $ using the training set for each polynomial degree.
  2. Find the polynomial degree d with the least error using the cross validation set.
  3. Estimate the generalization error using the test set with ${J_{test}}({\Theta ^{(d)}})$ (d = theta from polynomial with lower error);

  This way, the degree of the polynomial d has not been trained using the test set.

### Bias and variance

Knowing  bias and variance will give you a very strong indicator for what the useful and promising ways to try to improve your algorithm.

* In this section we examine the relationship between the degree of the polynomial d and the underfitting or overfitting of our hypothesis.
  * We need to distinguish whether **bias** or **variance** is the problem contributing to bad predictions.
  * High bias is underfitting and high variance is overfitting. Ideally, we need to find a golden mean between these two.

* The training error will tend to **decrease** as we increase the degree d of the polynomial.

  At the same time, the cross validation error will tend to **decrease** as we increase d up to a point, and then it will **increase** as d is increased, forming a convex curve. 

* **High bias (underfitting)**: both ${J_{train}}(\Theta )$ and  ${J_{CV}}(\Theta )$ will be high. Also, ${J_{train}}(\Theta )$ ≈${J_{CV}}(\Theta )$.

  **High variance (overfitting)**:  ${J_{train}}(\Theta )$ will be low and  ${J_{CV}}(\Theta )$ will be much greater than ${J_{train}}(\Theta )$.

  * The is summarized in the figure below:

   ![bias-variance](images/bias-variance.png)

### Regularization and Bias/Variance

  ![reg-b-v](images/reg-b-v.png)

* In the figure above, we see that as $\lambda$ increases, our fit becomes more rigid. On the other hand, as $ \lambda$ approaches 0, we tend to overfit the data. So how do we choose our parameter$  \lambda$  to get it 'just right' ? In order to choose the model and the regularization term $ \lambda$, we need to:
  1. Create a list of lambdas (i.e. λ∈{0,0.01,0.02,0.04,0.08,0.16,0.32,0.64,1.28,2.56,5.12,10.24});
  2. Create a set of models with different degrees or any other variants.
  3. Iterate through the $\lambda$s and for each $lambda$ go through all the models to learn some $\Theta$.
  4. Compute the cross validation error using the learned Θ (computed with λ) on the  ${J_{CV}}(\Theta )$ **without** regularization or λ = 0.
  5. Select the best combo that produces the lowest error on the cross validation set.
  6. Using the best combo Θ and λ, apply it on$ J_{test}(\Theta)$to see if it has a good generalization of the problem.

### Learning curves

* Training an algorithm on a very few number of data points (such as 1, 2 or 3) will easily have 0 errors because we can always find a quadratic curve that touches exactly those number of points. Hence:
  * As the training set gets larger, the error for a quadratic function increases.
  * The error value will plateau out after a certain m, or training set size.

* **Experiencing high bias:** 

  * Low training set size : causes c

  * Large training set size : causes both ${J_{train}}(\Theta )$ and  ${J_{CV}}(\Theta )$ to be high with ${J_{train}}(\Theta )$$ \approx $ ${J_{CV}}(\Theta )$

  * If a learning algorithm is suffering from **high bias**, getting more training data will not **(by itself)** help much.

![high-bias-lc](images/high-bias-lc.png)

* **Experiencing high variance:**
  * Low training set size : ${J_{train}}(\Theta )$ to be low and ${J_{CV}}(\Theta )$ to be high.
  * Large training set size :  ${J_{train}}(\Theta )$ increases with training set size and  ${J_{CV}}(\Theta )$ continues to decrease without leveling off. Also, ${J_{train}}(\Theta )$ < ${J_{CV}}(\Theta )$ but the difference between them remains significant.
  * If a learning algorithm is suffering from **high variance**, getting more training data is likely to help. 

![high-variance-lc](images/high-variance-lc.png)

##  Decide what to do next revisited

* Our decision process can be broken down as follows:
  * Getting more training examples : Fixes high variance
  * Trying smaller sets of features : Fixes high variance
  * Adding feature : Fixes high bias
  * Adding polynomial features : Fixes high bias
  * Decreasing $\lambda$ : Fixes high bias
  * Increasing $\lambda$ : Fixes high variance

* Diagnosing Neural Networks
  * A neural network with fewer parameters is **prone to underfitting**. It is also **computationally cheaper**.
  * A large neural network with more parameters is **prone to overfitting**. It is also **computationally expensive**. In this case you can use regularization (increase λ) to address the overfitting.
  * Using a single hidden layer is a good starting default. You can train your neural network on a number of hidden layers using your cross validation set. You can then select the one that performs best.

* Model complexity effects

  * Lower-order polynomials (low model complexity) have high bias and low variance. In this case, the model fits poorly consistently.

  * Higher-order polynomials (high model complexity) fit the training data extremely well and the test data extremely poorly. These have low bias on the training data, but very high variance.

  * In reality, we would want to choose a model somewhere in between, that can generalize well but also fits the data reasonably well.

### Prioritizing what to work on

**Building a spam classifier**

* Given a data set of emails, we could construct a vector for each email. Each entry in this vector represents a word. The vector normally contains 10,000 to 50,000 entries gathered by finding the most frequently used words in our data set. If a word is to be found in the email, we would assign its respective entry a 1, else if it is not found, that entry would be a 0. Once we have all our x vectors ready, we train our algorithm and finally, we could use it to classify if an email is a spam or not.

* How to spend time to make it have low error:

  * Collect lots of data - E.g.  "honeypot" project

  * Develop sophisticated feature based on email routing information(from email header)

  * Develop sophisticated feature for message body, e.g. should "discount" and "discounts" be treated as the same word? Features about punctuation?

  * Develop sophisticated algorithm to detect misspellings (e.d. m0rtgage, med1cine, w4tches.)

    `It is difficult to tell which of the options will be most helpful.`

### Error Analysis

* Recommend approach

  * Start with a simple algorithm that you can implement quickly. Implement it and test if on your cross-validation data

  * Plot learning curves to decide if more data, more features, etc.  are likely to help. Use that evident instead of gut feeling.

  * Manually examine the errors on examples in the cross validation set and try to spot a trend where most of the errors were made.

* For example, assume that we have 500 emails and our algorithm misclassifies a 100 of them. We could manually analyze the 100 emails and categorize them based on what type of emails they are. We could then try to come up with new cues and features that would help us classify these 100 emails correctly. Hence, if most of our misclassified emails are those which try to steal passwords, then we could find some features that are particular to those emails and add them to our model. We could also see how classifying each word according to its root changes our error rate。

* t is very important to get error results as a single, numerical value. Otherwise it is difficult to assess your algorithm's performance. For example if we use stemming, which is the process of treating the same word with different forms (fail/failing/failed) as one word (fail), and get a 3% error rate instead of 5%, then we should definitely add it to our model. However, if we try to distinguish between upper case and lower case letters and end up getting a 3.2% error rate instead of 3%, then we should avoid using this new feature. Hence, we should try new things, get a numerical value for our error rate, and based on our result decide whether we want to keep the new feature or not.

## Handling skewed data

**skewed classes**

When we're faced with such a skewed classes therefore we would want to come up with a different error metric / evaluation metric.

### Precision/Recall

y = 1 in presence of rare class that we want to detect

![R-A-table](/Users/apple/Documents/GitHub/MachineLearning-Cousera/Week 6/images/R-A-table.png)

**Precision** : Of all patients where we predicted y = 1, what fraction actually has cancer

` true POS/(#predicted POS) = true POS/(true POS + false POS)  `  

> Higer Precision is better

**Recall** : Of all patients that actually have cancer, what fraction did we correctly detect as having cancer

`true POS/(#actual POS) = true POS / (true POS + false NEG)`

> Higer Recall is better

* `Precision` and` Recall` are defined setting y equasl to 1, to be the presence of the **rare class** that we're trying to detect. 

* For the problem of skewed classes `Precision` and`Recall` gives us a much better way to evaluate learning algorithm.

### Trading off Precision and Recall

### Data For Machine Learning

