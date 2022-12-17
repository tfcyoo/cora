## Dependencies
* scklearn https://scikit-learn.org/stable/install.html
* pytorch https://pytorch.org/
* numpy https://numpy.org/install/
* pandas https://pandas.pydata.org/docs/getting_started/install.html
* matplotlib https://matplotlib.org/stable/users/installing/index.html


## How to run the code from command line:

python cora.py


## Approach

 - 3 Models are considered:
    - Neural Networks (NN)
    - Random Forest (RF)
    - Naive Bayes (NB)


 - Target variable encoding:

    - Neural_Networks => 0
    - Rule_Learning => 1
    - Reinforcement_Learning => 2
    - Probabilistic_Methods => 3
    - Theory => 4
    - Genetic_Algorithms => 5
    - Case_Based => 6


 - Remove paper_id from dataset since it carries no information

 - Models' training & tuning(Neural Networks, Random Forest) using Cross-validation with random split

 - Save the results on the test set in csv file for the 3 models

 - keep only the results of the best performing (accuracy) model