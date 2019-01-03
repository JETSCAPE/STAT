Bayesian parameter estimation for relativistic heavy-ion collisions
===================================================================

Docs: [Online Documentation](http://hic-param-est.readthedocs.io/en/latest/)

Introduction
------------

In the Statistical Analysis portion of the the school, we will be exploring Computer Emulation and Calibration through Gaussian Processes. The workshop will contain background material on all aspects of the analysis, from design of the experiment to parameter estimation. Students will have the opportunity to work through guided exercises in Python to further their understanding of the material. In the end, the students will perform a complete analysis on example data, and will leave with the tools and software to perform similar analyses on their own data

Please do the following prior to the first school working session, to enable us to get an efficient start. If you have any questions, please contact Jake Coleman (jake.coleman32@gmail.com) and Weiyao Ke (weiyaoke@gmail.com).

Get start with Python and Machine Learning
------------------------------------------

This is the version tailored for the JetScape Winter School 2019. It is writen in Python 3. For scientific computing with Python, we recommand reading the Scipy Lecture Notes available [online](https://www.scipy-lectures.org/), especially section 1.3 NumPy and 1.4 Matplotlib for array manipulations and plotting functions.
You are also encouraged to explore the documentation of the [scikit-learn](https://scikit-learn.org/stable/) library for Machine Learning in Python.


Installation
------------

1. Install Miniconda for Python 3.6 by following the [Regular Installation instructions](https://conda.io/docs/user-guide/install/index.html)

2. If you don't have R, download R from [here] https://cran.cnr.berkeley.edu/

3. Open an R Console instance by opening the R app or by typing R in the command line.

4. In the R console, type the command `install.packages('lhs')` and pick an appropriate download mirror if prompted. To ensure the package was properly installed, type `library(lhs)` in the R console. If that command runs without error, the package is installed. Close the R console by typing `quit()`.

5. Clone the git repository with branch `WS2018` from [here](https://github.com/keweiyao/hic-param-est-2017) by `git clone -b WS2018 https://github.com/keweiyao/hic-param-est-2017`

6. Open Terminal (OSX, Linux) or Windows Command Prompt (Windows).

7. Navigate to the downloaded/cloned git repository.

8. Type: `conda create -n hic-param-est numpy h5py scikit-learn pyyaml jupyter matplotlib pathlib pandas python=3.6`. This line creates a Python 3 environment called `hic-param-est` and downloads some necessary packages.

9. Type: `source activate hic-param-est`. This switches you into your newly created environment.

10. Type: `pip install emcee hic`. This installs some more packages that are unavailable through conda.

11. Type: jupyter notebook. This will open Jupyter iPython Notebook in a web browser.
In Jupyter, open `WinterSchoolQuestions.ipynb`, and run the first cell. If it runs without error, then you should be properly set up for the program.





