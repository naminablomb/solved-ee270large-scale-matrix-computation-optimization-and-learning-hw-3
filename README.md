Download Link: https://assignmentchef.com/product/solved-ee270large-scale-matrix-computation-optimization-and-learning-hw-3
<br>






<h1>1.   Logistic Regression</h1>

In this question, we will study logistic regression, which is a popular method in machine learning. We let <em>w </em>∈ R<em><sup>p </sup></em>be our decision variable. <em>w </em>represents weights for the columns of the <em>n </em>× <em>p </em>data matrix <em>X</em>, where the <em>i<sup>th </sup></em>row of <em>X </em>is <em>x<sup>T</sup><sub>i </sub></em>. Let

be the sigmoid/logistic function. This function is non-convex. However −log(<em>σ</em>(<em>a</em>)) is convex, which arises in maximum likelihood estimation as we describe next.

We assume that we collect data <em>x<sub>i</sub></em>, and a binary response variable <em>y<sub>i</sub></em>, which is the label, where

(+1 with probability <em>σ</em>(<em>w</em><em>T x</em><em>i</em>)

<em>y</em><em>i </em>=          −1 with probability 1 − <em>σ</em>(<em>w</em><em>T x</em><em>i</em>)          <em>.                                                         </em>(1)

We can write the above in the compact form <em>p</em>(<em>y<sub>i</sub></em>|<em>w,x<sub>i</sub></em>) = <em>σ</em>(<em>y<sub>i</sub>w<sup>T </sup>x<sub>i</sub></em>) since <em>σ</em>(<em>a</em>) = 1 − <em>σ</em>(−<em>a</em>). If we collect the observations, then the probability of observing this outcome is) and so the negative log-likelihood, which we will use as our

objective function, is

<em>.</em>

Note that minimizing negative log-likelihood is same as maximizing the likelihood. This corresponds to maximum likelihood estimation of the parameter <em>w</em>. Once <em>w </em>is identified, we can use (1) to infer the label of a test data point <em>x</em>.

<ul>

 <li>Derive the gradient ∇<em>`</em>(<em>w</em>).</li>

 <li>Derive the Hessian ∇<sup>2</sup><em>`</em>(<em>w</em>).</li>

 <li>Is the cost function <em>`</em>(<em>w</em>) convex?</li>

</ul>

<h1>2.   Logistic Regression for Spam E-mail Classification</h1>

Download the email spam data set , which is available at <a href="https://github.com/probml/pmtk3/tree/master/data/spamData">https://github.com/probml/ </a><a href="https://github.com/probml/pmtk3/tree/master/data/spamData">pmtk3/tree/master/data/spamData</a> in Matlab and plain text format. This set consists of 4601 email messages, from which 57 features have been extracted. These are as follows

<ul>

 <li>48 features, in [0 100], giving the percentage of words in a given message which match a given word on the list. The list contains words such as “business”, “free”, “george”, etc.</li>

 <li>6 features, in [0 100], giving the percentage of characters in the email that match a given character on the list. The characters are ; ( [ ! $ #</li>

 <li>Feature 55: The average length of an uninterrupted sequence of capital letters (max is 40.3, mean is 4.9) Feature 56: The length of the longest uninterrupted sequence of capital letters ( max is 45, mean is 52.6) • Feature 57: The sum of the lengths of uninterrupted sequence of capital letters (max is 25.6, mean is 282.2)</li>

</ul>

Load the data set using the provided link. In the Matlab version, there is a training set of size 3065 and a test set of size 1536. In the plain text version, there is no predefined testing/training set, you can randomly shuffle the data to pick a training set of size 3065 and use the rest for testing (or you can use the Matlab data along with scipy.io.loadmat).

There are different methods to pre-process the data, e.g., standardize the columns of <em>X</em>. For this problem, transform the features using log(<em>x<sub>ij </sub></em>+ 0<em>.</em>1). One could also add some regularization to the loss function which can help generalization error but this is not necessary. Also note that you need to transform the labels from {0<em>,</em>1} to {1<em>,</em>−1}.

<ul>

 <li>Run gradient descent with a fixed step-size. Plot the value of the cost function at each iteration and find a reasonable step-size for fast convergence</li>

 <li>Repeat the previous part using gradient descent with momentum.</li>

 <li>Implement gradient descent with Armijo line search. This procedure is as follows: Assume that we are at the point <em>x<sub>k </sub></em>and have a search direction <em>p<sub>k </sub></em>(for gradient descent <em>p<sub>k </sub></em>= −∇<em>f</em>(<em>x<sub>k</sub></em>)). Then, the Armijo line search procedure is:

  <ul>

   <li>Pick an initial step-size <em>t</em></li>

   <li>Initialize the parameters 0 <em>&lt; ρ &lt; </em>1 and 0 <em>&lt; c &lt; </em>1 (typical values are <em>c </em>= 10<sup>−4 </sup>and <em>ρ </em>= 0<em>.</em>9)</li>

   <li>While <em>f</em>(<em>x<sub>k </sub></em>+ <em>tp<sub>k</sub></em>) <em>&gt; f</em>(<em>x<sub>k</sub></em>) + <em>ct</em>∇<em>f</em>(<em>x<sub>k</sub></em>)<em><sup>T </sup>p<sub>k</sub></em>, do <em>t </em>← <em>ρt</em></li>

   <li>Terminate the procedure if <em>f</em>(<em>x<sub>k </sub></em>+ <em>tp<sub>k</sub></em>) ≤ <em>f</em>(<em>x<sub>k</sub></em>) + <em>ct</em>∇<em>f</em>(<em>x<sub>k</sub></em>)<em><sup>T </sup>p<sub>k</sub></em></li>

  </ul></li>

</ul>

The test statement in the while loop is the Armijo condition. If <em>p<sub>k </sub></em>= −∇<em>f</em>(<em>x<sub>k</sub></em>), then the test is accepted when. In general, the second term is negative as long as <em>p<sub>k </sub></em>is a descent direction. One can prove this linesearch procedure will terminate.

Find a good estimate for the initial step-size by trial and error. A simple idea is to use the final step-size from the previous step, but this can be unnecessarily small. You may want to do this, but increase the step-size by a factor of 2.

<h1>3.   Newton’s Method is Affine Invariant</h1>

In this question, we will prove the affine invariance of Newton’s method. Let <em>f </em>: R<em><sup>n </sup></em>→ R be a convex function. Consider an affine transform <em>y </em>→ <em>Ay </em>+ <em>b</em>, where <em>A </em>∈ R<em><sup>n</sup></em><sup>×<em>n </em></sup>is invertible and <em>b </em>∈ R<em><sup>n</sup></em>. Define the function <em>g </em>: R<em><sup>n </sup></em>→ R by <em>g</em>(<em>y</em>) = <em>f</em>(<em>Ay </em>+ <em>b</em>). Denote by <em>x</em><sup>(<em>k</em>) </sup>the <em>k<sup>th </sup></em>iterate of Newton’s method performed on <em>f</em>. Denote <em>y</em><sup>(<em>k</em>) </sup>the <em>k<sup>th </sup></em>iterate of Newton’s method performed on <em>g</em>.

<ul>

 <li>Show that if <em>x</em>(<em>k</em>) = <em>Ay</em>(<em>k</em>) + <em>b</em>, then <em>x</em>(<em>k</em>+1) = <em>Ay</em>(<em>k</em>+1) + <em>b</em>.</li>

 <li>Show that Newton’s decrement does not depend on the coordinates, i.e., show that <em>λ</em>(<em>x</em><sup>(<em>k</em>)</sup>) = <em>λ</em>(<em>y</em><sup>(<em>k</em>)</sup>), where <em>λ</em>(<em>x</em>) = (∇<em>f</em>(<em>x</em>)<em><sup>T </sup></em>∇<sup>2</sup><em>f</em>(<em>x</em>)<sup>−1</sup>∇<em>f</em>(<em>x</em>))<sup>1<em>/</em>2</sup>.</li>

</ul>

Together, this implies that Newton’s method is affine invariant. As an important consequence, Newton’s method cannot be improved by a change of coordinates, unlike gradient descent.

<h1>4.   Newton’s Method for Convex Optimization</h1>

<ul>

 <li>Implement Newton’s method for the logistic regression problem in Problem 1. Plot the value of the cost function at each iteration and find a reasonable step-size for fast convergence.</li>

 <li>Implement randomized Newton’s method with uniform sampling sketch, i.e., sampling rows of <em>H</em><sup>1<em>/</em>2 </sup>uniformly at random where <em>H </em>and <em>H</em><sup>1<em>/</em>2 </sup>denote Hessian and its square-root respectively. Plot the value of the cost function at each iteration and find a reasonable step-size and sketch-size for fast convergence.</li>

</ul>

<h1>5.   Fast Johnson-Lindenstrauss Transform (FJLT) using Hadamard Matrices</h1>

<ul>

 <li>Construct an 128 × 1024 FJLT matrix as follows</li>

</ul>

Set <em>m </em>= 128 and <em>n </em>= 1024

Define

Construct <em>H</em><sub>10 </sub>∈ R<sup>1024<em>,</em>1024 </sup>recursively via

Generate <em>D </em>as an <em>n</em>×<em>n </em>diagonal matrix of uniformly random ±1 variables (Rademacher distribution)

Generate an <em>m </em>× <em>n </em>uniform sub-sampling matrix <em>P </em>scaled with  (uniform sampling sketch)

Form the FJLT matrix.

<ul>

 <li>Verify that <em>S<sup>T </sup>S </em>is a multiple of identity. Scale <em>S </em>appropriately if needed to obtain <em>S<sup>T </sup>S </em>= <em>I</em>.</li>

 <li>Generate a data matrix <em>A </em>of size 1024 × 10 using i.i.d. standard Gaussian variables. Plot the singular values of <em>A </em>and singular values of <em>SA</em>.</li>

 <li>In part (c), <em>SA </em>is a Johnson-Lindenstrauss embedding of 10 vectors (1024 dimensional column vectors of <em>A</em>) to dimension 128. Verify that the pairwise distances are approximately preserved, i.e., there exists an</li>

</ul>

(2)

where <em>A<sub>i </sub></em>is the i-th column of <em>A</em>. Find <sup>∗</sup>, the smallest value of <em> </em>that satisfy (2) for a single realization of the random construction. Note that the matrices <em>D </em>and <em>P </em>are constructed randomly, while <em>H </em>is deterministic. What is the minimum, maximum, and mean value of <sup>∗ </sup>in 100 random realizations of the construction?

Hint: The JL embedding property in (2) specifies 2<em>d</em><sup>2 </sup>linear inequalities of the form

, hence the smallest.

<ul>

 <li>Generate a data matrix <em>A </em>of size 1024 × 10, and a vector <em>b </em>of size 1024 × 1 using i.i.d. standard Gaussian variables. Solve the least squares problem</li>

</ul>

<em>x</em><sup>∗ </sup>= argmin<em> . </em><em>x</em>

Apply the FJLT to <em>A </em>and <em>b </em>as <em>SA </em>and <em>Sb</em>. Solve the sketched least squares problem

<em>x</em>˜ := argmin<em> . </em><em>x</em>

Find the Euclidean distance between the solutions, i.e., k<em>x</em>˜−<em>x</em><sup>∗</sup>k<sub>2</sub>, between their predictions, i.e., k<em>Ax</em>˜ − <em>Ax</em><sup>∗</sup>k<sub>2 </sub>and the approximation ratio () of the objective value.