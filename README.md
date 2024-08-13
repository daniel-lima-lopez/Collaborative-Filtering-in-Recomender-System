# Collaborative-Filtering-in-Recomender-System
This repository presents an implementation of a movie recommender system based on Collaborative Filtering. The dataset used in this work, [ml-latest-small](https://grouplens.org/datasets/movielens/), contains over 100,000 ratings performed by 600 users considering 9,000 movies.

## Installation
Clone this repository:
```bash
git clone git@github.com:daniel-lima-lopez/Collaborative-Filtering-in-Recomender-System.git
```
move to installation directory:
```bash
cd Collaborative-Filtering-in-Recomender-System
```

## Method description
Given a system with $M$ users and $N$ items, a Recommender System is characterized  by a matrix $R\in\mathbb{R}^{M\times N}$, denominated interaction matrix. Each $r_{u,i}\in R$ represents the preference of user $u$ for item $i$.

The Collaborative Filtering technique leverages the information of similar users to predict possible  new preferences. This process is divided in the following steps. For a user $u$:
1. Obtain the vector of preferences $r^{(u)}=\{r_{u1}, r_{u2},\dots, u_{uN}\}$ (u-th row).
2. Identify the k-Nearest Neighbors $r^{u_k}$ to $r^{(u)}$, that is, the users with similar preferences as $u$.
3. For each unknown preference $r_{uj}$, calculate the mean of the k-Nearest Neighbors preferences for item $j$, as long as these users have a defined preference for this item.

## Experiments
The notebook [exp1.ipynb](exp1.ipynb) presents 10-folds cross-validation experiments with different k-values. On each experiment, it is measure the Mean Squared Error (MSE) between the test and predicted R-matrix. The best result is achieved with $k=3$, as shown below:
<img src="imgs/MSE_ks.png" alt="drawing" width="600"/>

A simple experiment with this k value is performed on the Notebook [exp2.ipynb](exp2.ipynb). It is presented some examples of users' favorite movies, accompanied by the method's prediction of possible new preferences:
```
Top u1 movies:
 - Mad Max: Fury Road (2015)
 - Wolf of Wall Street, The (2013)
 - The Jinx: The Life and Deaths of Robert Durst (2015)
 - Step Brothers (2008)
 - Warrior (2011)
Top u1 preds:
 - Citizen Kane (1941)
 - Lock, Stock & Two Smoking Barrels (1998)
 - Adventures of Robin Hood, The (1938)
 - Wolf Man, The (1941)
 - Go (1999)

Top u4 movies:
 - Once Were Warriors (1994)
 - Schindler's List (1993)
 - In the Name of the Father (1993)
 - Snow White and the Seven Dwarfs (1937)
 - Pinocchio (1940)
Top u4 preds:
 - Run Lola Run (Lola rennt) (1998)
 - Dr. Horrible's Sing-Along Blog (2008)
 - Crazy, Stupid, Love. (2011)
 - Avengers, The (2012)
 - Inception (2010)

 Top u232 movies:
 - Requiem for a Dream (2000)
 - Shawshank Redemption, The (1994)
 - Eternal Sunshine of the Spotless Mind (2004)
 - Three Billboards Outside Ebbing, Missouri (2017)
 - Forrest Gump (1994)
Top u232 preds:
 - Twelve Monkeys (a.k.a. 12 Monkeys) (1995)
 - From Dusk Till Dawn (1996)
 - Fistful of Dollars, A (Per un pugno di dollari) (1964)
 - Unleashed (Danny the Dog) (2005)
 - Blade Runner (1982)
```
