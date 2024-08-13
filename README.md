# Collaborative-Filtering-in-Recomender-System
This repository presents an implementation of a movie recomender system based on Collaborative Filtering. The dataset used in this work, [ml-latest-small](https://grouplens.org/datasets/movielens/), contains over 100,000 ratings performed by 600 users considering 9,000 movies.

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
Give a system with $M$ users and $N$ items, a Recomender System is caracterized by a matrix $R\in\mathbb{R}^{M\times N}$, denominated interaction matrix. Each $r_{u,i}\in R$ represents the preference of user $u$ for item $i$.

The Collaborative Filtering technique leverages the information of similar users to predict posible new preferences. This process is divided in the following steps. For a user $u$:
1. Obtain the vector of preferences $r^{(u)}=\{r_{u1}, r_{u2},\dots, u_{uN}\}$ (u-th row).
2. Identify the k-Nearest Neighbors $r^{u_k}$ to $r^{(u)}$, that is the users with similir preferences as $u$.
3. For each unknown preference $r_{uj}$, calculate the mean of the k-Nearest Neighbors preferences for item $j$, as long as these users have a defined preference for this item.

## Experiments
The notebook [exp1.ipynb](exp1.ipynb) presents 10-folds cross-validation experiments with different k-values. On each experiment it is measeure the Mean Squared Error (MSE) between the test and predicted R-matrix:
<img src="imgs/MSE_ks.png" alt="drawing" width="600"/>

The best result is acchieved with $k=3$.