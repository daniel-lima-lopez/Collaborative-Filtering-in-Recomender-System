# Movie Recomendation System
Recommendation systems offers several advantages in various industries, including personalized product or content suggestions, enhancing customer experience and user engagement. This technique enable businesses to target marketing effors more efectively, by analizing customer behavior and preferences. Furtheremore, sectors like retal, entertainment and finance leverage this approach for gaining competitive advantage and optimizing user interactions.

This repository presents the implementation of a movie recommendation system that leverages K-means clustering for movie classification and Collaborative Filtering for personalized user recommendations. This project is aimed at enhancing user engagement and driving insights into consumer preferences. The data used can be found at [Movie Lens Dataset](https://www.kaggle.com/datasets/aigamer/movie-lens-dataset?select=tags.csv).

The objectives of this project are:
- Gain valuable insights into user preferences.
- Generate a data set with the most valuable information to classify movies according to their characteristics.
- Make personalized movie recommendations based on the interests of each user.

The repository is organized as follows:
1. Exploratory Data Analysis (EDA).
2. Model implementation.
3. Recomendation examples.

## 1. Exploratory Data Analysis (EDA)
First, we estar by loading the required libraries and reading the movies data:


```python
# load libraries
import pandas as pd
import numpy as np
from pandasql import sqldf
import matplotlib.pyplot as plt
import seaborn as sb

# data from movies
movies_data = pd.read_csv('Data/movies.csv')
movies_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movieId</th>
      <th>title</th>
      <th>genres</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Toy Story (1995)</td>
      <td>Adventure|Animation|Children|Comedy|Fantasy</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Jumanji (1995)</td>
      <td>Adventure|Children|Fantasy</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Grumpier Old Men (1995)</td>
      <td>Comedy|Romance</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Waiting to Exhale (1995)</td>
      <td>Comedy|Drama|Romance</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Father of the Bride Part II (1995)</td>
      <td>Comedy</td>
    </tr>
  </tbody>
</table>
</div>



Subsequently, the information is reorganized, separating the name and year of each film into different columns, as well as the genres in which it is classified:


```python
# extraction of year on each movie
titles = []
years = []
for ti in movies_data['title'].values:
    ti = ti.strip()
    auxy = ti[-5:-1]
    # extraction of year on each movie
    if np.char.isnumeric(auxy):
        titles.append(ti[:-7])
        years.append(ti[-5:-1])
    else:
        titles.append(ti)
        years.append('-')

movies_data['title'] = titles
movies_data['year'] = years

# identification of unique genres
gens = movies_data['genres'].values
auxg = []
for gi in gens:
    auxg += gi.split('|')
auxg = np.unique(auxg)
auxg = auxg[1:] # drop 'no genre listed'

# identification of genres on each movie
aux_dic = {}
for gi in auxg:
    aux_dic[gi] = [0]*len(gens)

for i, gis in enumerate(gens):
    split = gis.split('|')
    if split !=['(no genres listed)']:
        for si in split:
            aux_dic[si][i] = 1
        
# add features to dataframe
for ki in aux_dic.keys():
    movies_data[ki] = aux_dic[ki]

# drop previous genre feature
movies_data = movies_data.drop(['genres'], axis=1)
movies_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movieId</th>
      <th>title</th>
      <th>year</th>
      <th>Action</th>
      <th>Adventure</th>
      <th>Animation</th>
      <th>Children</th>
      <th>Comedy</th>
      <th>Crime</th>
      <th>Documentary</th>
      <th>...</th>
      <th>Film-Noir</th>
      <th>Horror</th>
      <th>IMAX</th>
      <th>Musical</th>
      <th>Mystery</th>
      <th>Romance</th>
      <th>Sci-Fi</th>
      <th>Thriller</th>
      <th>War</th>
      <th>Western</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Toy Story</td>
      <td>1995</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Jumanji</td>
      <td>1995</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Grumpier Old Men</td>
      <td>1995</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Waiting to Exhale</td>
      <td>1995</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Father of the Bride Part II</td>
      <td>1995</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 22 columns</p>
</div>



The rating data, which contains 100,000 movie reviews, is read. This information is leveraged to improve the movie data by adding two new columns: the number of ratings for each movie and its average rating score (minimum 0.0 and maximum 5.0).


```python
# data reading
ratings_data = pd.read_csv('Data/ratings.csv')

# count and average calculation
mean_ratings = sqldf(''' 
    SELECT movieID, COUNT(userID) as ratings, AVG(rating) as avg_rating 
    FROM ratings_data GROUP BY movieID
''')

# joining data with movies dataframe
movies_data = sqldf(''' 
    WITH T AS 
        (SELECT * from movies_data LEFT JOIN
        mean_ratings on movies_data.movieId=mean_ratings.movieId)
    SELECT movieId, title, year, ratings, avg_rating, Action, Adventure, Animation,
       Children, Comedy, Crime, Documentary, Drama, Fantasy,
       [Film-Noir], Horror, IMAX, Musical, Mystery, Romance,
       [Sci-Fi], Thriller, War, Western FROM T
''')
# write final movies dataframe
movies_data.to_csv('Data/movies_full_data.csv', index=False)
movies_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movieId</th>
      <th>title</th>
      <th>year</th>
      <th>ratings</th>
      <th>avg_rating</th>
      <th>Action</th>
      <th>Adventure</th>
      <th>Animation</th>
      <th>Children</th>
      <th>Comedy</th>
      <th>...</th>
      <th>Film-Noir</th>
      <th>Horror</th>
      <th>IMAX</th>
      <th>Musical</th>
      <th>Mystery</th>
      <th>Romance</th>
      <th>Sci-Fi</th>
      <th>Thriller</th>
      <th>War</th>
      <th>Western</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Toy Story</td>
      <td>1995</td>
      <td>215.0</td>
      <td>3.920930</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Jumanji</td>
      <td>1995</td>
      <td>110.0</td>
      <td>3.431818</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Grumpier Old Men</td>
      <td>1995</td>
      <td>52.0</td>
      <td>3.259615</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Waiting to Exhale</td>
      <td>1995</td>
      <td>7.0</td>
      <td>2.357143</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Father of the Bride Part II</td>
      <td>1995</td>
      <td>49.0</td>
      <td>3.071429</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 24 columns</p>
</div>



With the information extracted we can study the general content of the data. For example, the distribution of films published in each year, where it can be observed that most of the data belongs to films published from 1990 onwards. However, older films are also present, including records from 1902.


```python
# number of movies released per year
year_data = sqldf(''' 
    SELECT * FROM movies_data ORDER BY year
''')

# movies distribution per year
movie_year = sqldf('''  
    SELECT year, COUNT(year) AS freq FROM movies_data
    GROUP BY year
''')

# distribution plot
fig1, ax1 = plt.subplots()
fig1.set_size_inches(20, 3.5)
years = movie_year['year'].values
counts = movie_year['freq'].values
ax1.bar(years, counts)
plt.xticks(years, rotation = 70)
plt.show()
```


    
![png](README_files/README_8_0.png)
    


Regarding the genres of each film, the following figure represents the distribution of genres across all films. It should be noted that Drama and Comedy are the most viewed genres.


```python
import squarify
# movies distribution per genre
freqs = [] # frequency
for gi in auxg: # itereate over genres
    auxF = sqldf(f''' 
        SELECT * FROM movies_data WHERE [{gi}]=1
    ''')
    freqs.append(len(auxF))

# Treemap
plt.axis("off")
squarify.plot(sizes=freqs, label=auxg,
                text_kwargs = {'fontsize': 8, 'color': 'white'},
                pad=0.2, ec= 'black',
                color = sb.color_palette("flare", len(freqs)))
```




    <Axes: >




    
![png](README_files/README_10_1.png)
    


Below are highest average rating drama films of recent years, including films such as The Shawshank Redemption, Fight Club and Goodfellas:


```python
# top drama movies in the last years, considering movies with at least 15 reviews
top_drama = sqldf('''  
    SELECT movieId, title, year, avg_rating FROM movies_data
    WHERE ratings >= 15 AND Drama=1 AND year>=1990
    ORDER BY avg_rating DESC
''')
top_drama.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movieId</th>
      <th>title</th>
      <th>year</th>
      <th>avg_rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>318</td>
      <td>Shawshank Redemption, The</td>
      <td>1994</td>
      <td>4.429022</td>
    </tr>
    <tr>
      <th>1</th>
      <td>475</td>
      <td>In the Name of the Father</td>
      <td>1993</td>
      <td>4.300000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2959</td>
      <td>Fight Club</td>
      <td>1999</td>
      <td>4.272936</td>
    </tr>
    <tr>
      <th>3</th>
      <td>48516</td>
      <td>Departed, The</td>
      <td>2006</td>
      <td>4.252336</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1213</td>
      <td>Goodfellas</td>
      <td>1990</td>
      <td>4.250000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1719</td>
      <td>Sweet Hereafter, The</td>
      <td>1997</td>
      <td>4.250000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>58559</td>
      <td>Dark Knight, The</td>
      <td>2008</td>
      <td>4.238255</td>
    </tr>
    <tr>
      <th>7</th>
      <td>527</td>
      <td>Schindler's List</td>
      <td>1993</td>
      <td>4.225000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1245</td>
      <td>Miller's Crossing</td>
      <td>1990</td>
      <td>4.225000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>3275</td>
      <td>Boondock Saints, The</td>
      <td>2000</td>
      <td>4.220930</td>
    </tr>
  </tbody>
</table>
</div>



Taking the most popular Drama movie (The Shawshank Redemption wiith movideId 318) as an example, we can study the most watched genres by users who have watched this movie. The following figure shows the distribution of genres watched by these users, note that Drama remains the most watched genre, because users watch movies of similar genres.


```python
# distribution of genres wathced by users who liked Shawshank Redemption, The (moviedId 318)
# viewers of selected movie
viewers_data = sqldf(''' 
    WITH T2 AS
        (WITH T AS
            (SELECT userId FROM ratings_data
            WHERE  movieId=318)
        SELECT * FROM T LEFT JOIN ratings_data
            on T.userId=ratings_data.userId)
    SELECT userId, movieId, rating from T2 WHERE movieId!=318
''')

# information of movies watched by users and reviews with at least 4.0
bests = sqldf(''' 
    WITH T AS (
        SELECT DISTINCT movieId FROM viewers_data
        WHERE rating>=5.0
        ORDER BY movieId
              )
    SELECT * FROM T LEFT JOIN movies_data
    ON T.movieId=movies_data.movieId
''')
bests = bests.drop(['movieId'], axis=1)

# genres count
freqs = []
for gi in auxg:
    auxF = sqldf(f''' 
        SELECT * FROM bests WHERE [{gi}]=1
    ''')
    freqs.append(len(auxF))

# Treemap
plt.axis("off")
squarify.plot(sizes=freqs, label=auxg,
                text_kwargs = {'fontsize': 8, 'color': 'white'},
                pad=0.2, ec= 'black',
                color = sb.color_palette("flare", len(freqs)))
```




    <Axes: >




    
![png](README_files/README_14_1.png)
    


## 2. Model implementation.
The implementation of the model requires training the k-Means algorithm on the characteristics of the films and calculating the collaborative filtering matrices of each cluster. This process is performed once and both the trained model and the set of matrices are stored for later use to make personalized recommendations.

Regarding the training of the k-Means algorithm, a training dataset is first constructed with the features of each movie, including the year of publication, number of ratings, average rating, and genres. Next, a pipeline is implemented that includes a Column Transformer and the k-Means algorithm. The column transformer class transforms each column depending on whether the attribute is numeric or nominal. Finally, the k-Means algorithm is trained on the movie features and the model is written to a pickle file.


```python
# loading libraries
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.cluster import KMeans
import pickle


# generation of training data
train_data = sqldf(''' 
    SELECT movieId, year, ratings, avg_rating, Action, Adventure, Animation,
       Children, Comedy, Crime, Documentary, Drama, Fantasy,
       [Film-Noir], Horror, IMAX, Musical, Mystery, Romance,
       [Sci-Fi], Thriller, War, Western FROM movies_data    
''')

# data procesing
train_data = train_data.dropna()
Xs = train_data.drop(['movieId'], axis=1)

# procesing of numerical features (year, ratings, avg_rating)
numeric_features = ['year', 'ratings', 'avg_rating']
numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])

# procesing of categorical features (no transformation)
categorical_features = ['Action', 'Adventure', 'Animation',
       'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
       'Film-Noir', 'Horror', 'IMAX', 'Musical', 'Mystery', 'Romance',
       'Sci-Fi', 'Thriller', 'War', 'Western']
categorical_transformer = FunctionTransformer() # transformacion identidad

# preprocesador de ambos tipos
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# k-means algorithm
kmeans = KMeans(n_clusters=5, random_state=1, n_init="auto")

model = Pipeline(steps=[('preprocessor', preprocessor),
                      ('k-means', kmeans)])
model.fit(Xs)

# saving model
with open('model_k_means.pkl','wb') as f:
    pickle.dump(model,f)
model
```




<style>#sk-container-id-1 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: black;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-1 {
  color: var(--sklearn-color-text);
}

#sk-container-id-1 pre {
  padding: 0;
}

#sk-container-id-1 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-1 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-1 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-1 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-1 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-1 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-1 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-1 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-1 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-1 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-1 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-1 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-1 label.sk-toggleable__label {
  cursor: pointer;
  display: block;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
}

#sk-container-id-1 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-1 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-1 div.sk-label label.sk-toggleable__label,
#sk-container-id-1 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-1 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-1 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-1 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-1 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-1 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 1ex;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-1 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-1 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-1 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-1 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>Pipeline(steps=[(&#x27;preprocessor&#x27;,
                 ColumnTransformer(transformers=[(&#x27;num&#x27;,
                                                  Pipeline(steps=[(&#x27;scaler&#x27;,
                                                                   StandardScaler())]),
                                                  [&#x27;year&#x27;, &#x27;ratings&#x27;,
                                                   &#x27;avg_rating&#x27;]),
                                                 (&#x27;cat&#x27;, FunctionTransformer(),
                                                  [&#x27;Action&#x27;, &#x27;Adventure&#x27;,
                                                   &#x27;Animation&#x27;, &#x27;Children&#x27;,
                                                   &#x27;Comedy&#x27;, &#x27;Crime&#x27;,
                                                   &#x27;Documentary&#x27;, &#x27;Drama&#x27;,
                                                   &#x27;Fantasy&#x27;, &#x27;Film-Noir&#x27;,
                                                   &#x27;Horror&#x27;, &#x27;IMAX&#x27;, &#x27;Musical&#x27;,
                                                   &#x27;Mystery&#x27;, &#x27;Romance&#x27;,
                                                   &#x27;Sci-Fi&#x27;, &#x27;Thriller&#x27;, &#x27;War&#x27;,
                                                   &#x27;Western&#x27;])])),
                (&#x27;k-means&#x27;, KMeans(n_clusters=5, random_state=1))])</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" ><label for="sk-estimator-id-1" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;&nbsp;Pipeline<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.pipeline.Pipeline.html">?<span>Documentation for Pipeline</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></label><div class="sk-toggleable__content fitted"><pre>Pipeline(steps=[(&#x27;preprocessor&#x27;,
                 ColumnTransformer(transformers=[(&#x27;num&#x27;,
                                                  Pipeline(steps=[(&#x27;scaler&#x27;,
                                                                   StandardScaler())]),
                                                  [&#x27;year&#x27;, &#x27;ratings&#x27;,
                                                   &#x27;avg_rating&#x27;]),
                                                 (&#x27;cat&#x27;, FunctionTransformer(),
                                                  [&#x27;Action&#x27;, &#x27;Adventure&#x27;,
                                                   &#x27;Animation&#x27;, &#x27;Children&#x27;,
                                                   &#x27;Comedy&#x27;, &#x27;Crime&#x27;,
                                                   &#x27;Documentary&#x27;, &#x27;Drama&#x27;,
                                                   &#x27;Fantasy&#x27;, &#x27;Film-Noir&#x27;,
                                                   &#x27;Horror&#x27;, &#x27;IMAX&#x27;, &#x27;Musical&#x27;,
                                                   &#x27;Mystery&#x27;, &#x27;Romance&#x27;,
                                                   &#x27;Sci-Fi&#x27;, &#x27;Thriller&#x27;, &#x27;War&#x27;,
                                                   &#x27;Western&#x27;])])),
                (&#x27;k-means&#x27;, KMeans(n_clusters=5, random_state=1))])</pre></div> </div></div><div class="sk-serial"><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-2" type="checkbox" ><label for="sk-estimator-id-2" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;preprocessor: ColumnTransformer<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.compose.ColumnTransformer.html">?<span>Documentation for preprocessor: ColumnTransformer</span></a></label><div class="sk-toggleable__content fitted"><pre>ColumnTransformer(transformers=[(&#x27;num&#x27;,
                                 Pipeline(steps=[(&#x27;scaler&#x27;, StandardScaler())]),
                                 [&#x27;year&#x27;, &#x27;ratings&#x27;, &#x27;avg_rating&#x27;]),
                                (&#x27;cat&#x27;, FunctionTransformer(),
                                 [&#x27;Action&#x27;, &#x27;Adventure&#x27;, &#x27;Animation&#x27;,
                                  &#x27;Children&#x27;, &#x27;Comedy&#x27;, &#x27;Crime&#x27;, &#x27;Documentary&#x27;,
                                  &#x27;Drama&#x27;, &#x27;Fantasy&#x27;, &#x27;Film-Noir&#x27;, &#x27;Horror&#x27;,
                                  &#x27;IMAX&#x27;, &#x27;Musical&#x27;, &#x27;Mystery&#x27;, &#x27;Romance&#x27;,
                                  &#x27;Sci-Fi&#x27;, &#x27;Thriller&#x27;, &#x27;War&#x27;, &#x27;Western&#x27;])])</pre></div> </div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-3" type="checkbox" ><label for="sk-estimator-id-3" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">num</label><div class="sk-toggleable__content fitted"><pre>[&#x27;year&#x27;, &#x27;ratings&#x27;, &#x27;avg_rating&#x27;]</pre></div> </div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-4" type="checkbox" ><label for="sk-estimator-id-4" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;StandardScaler<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.preprocessing.StandardScaler.html">?<span>Documentation for StandardScaler</span></a></label><div class="sk-toggleable__content fitted"><pre>StandardScaler()</pre></div> </div></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-5" type="checkbox" ><label for="sk-estimator-id-5" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">cat</label><div class="sk-toggleable__content fitted"><pre>[&#x27;Action&#x27;, &#x27;Adventure&#x27;, &#x27;Animation&#x27;, &#x27;Children&#x27;, &#x27;Comedy&#x27;, &#x27;Crime&#x27;, &#x27;Documentary&#x27;, &#x27;Drama&#x27;, &#x27;Fantasy&#x27;, &#x27;Film-Noir&#x27;, &#x27;Horror&#x27;, &#x27;IMAX&#x27;, &#x27;Musical&#x27;, &#x27;Mystery&#x27;, &#x27;Romance&#x27;, &#x27;Sci-Fi&#x27;, &#x27;Thriller&#x27;, &#x27;War&#x27;, &#x27;Western&#x27;]</pre></div> </div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-6" type="checkbox" ><label for="sk-estimator-id-6" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;FunctionTransformer<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.preprocessing.FunctionTransformer.html">?<span>Documentation for FunctionTransformer</span></a></label><div class="sk-toggleable__content fitted"><pre>FunctionTransformer()</pre></div> </div></div></div></div></div></div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-7" type="checkbox" ><label for="sk-estimator-id-7" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;KMeans<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.cluster.KMeans.html">?<span>Documentation for KMeans</span></a></label><div class="sk-toggleable__content fitted"><pre>KMeans(n_clusters=5, random_state=1)</pre></div> </div></div></div></div></div></div>



The following table presents the top 10 rated movies found in each cluster after training the k-Mean algorithm considering 5 clusters:


```python
# prediction of the class of each movie
preds = model.predict(Xs)

# table with the classification of each movie
movies_class = pd.DataFrame({'movieId':train_data['movieId'].values,
                             'class': preds})

# information join
auxC = sqldf(''' 
    SELECT movies_class.movieId, title, year, ratings, avg_rating, class FROM movies_class LEFT JOIN movies_data
    ON movies_class.movieId=movies_data.movieId
''')

# top 10 movies on each cluster, considering at least 20 reviews
aux_dic = {'ranking': [i for i in range(1,11)]}
for ci in range(5): # tenemos 5 clusters
    aux_top = sqldf(f''' 
        SELECT * FROM auxC
        WHERE class={ci} AND ratings>=20
        ORDER BY avg_rating DESC
    ''')
    aux_dic[f'class_{ci}'] = list(aux_top['title'].values[0:10])
data_tops = pd.DataFrame(aux_dic)
data_tops
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ranking</th>
      <th>class_0</th>
      <th>class_1</th>
      <th>class_2</th>
      <th>class_3</th>
      <th>class_4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>In the Name of the Father</td>
      <td>Streetcar Named Desire, A</td>
      <td>Shawshank Redemption, The</td>
      <td>Old Boy</td>
      <td>Buffy the Vampire Slayer</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Hoop Dreams</td>
      <td>Sunset Blvd. (a.k.a. Sunset Boulevard)</td>
      <td>Godfather, The</td>
      <td>Grand Day Out with Wallace and Gromit, A</td>
      <td>Joe Dirt</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Logan</td>
      <td>Philadelphia Story, The</td>
      <td>Fight Club</td>
      <td>Howl's Moving Castle (Hauru no ugoku shiro)</td>
      <td>Toys</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Miller's Crossing</td>
      <td>Lawrence of Arabia</td>
      <td>Dr. Strangelove or: How I Learned to Stop Worr...</td>
      <td>Femme Nikita, La (Nikita)</td>
      <td>Angels in the Outfield</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Boondock Saints, The</td>
      <td>Harold and Maude</td>
      <td>Rear Window</td>
      <td>Kiss Kiss Bang Bang</td>
      <td>The Scorpion King</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>Boot, Das (Boat, The)</td>
      <td>Cool Hand Luke</td>
      <td>Godfather: Part II, The</td>
      <td>Laputa: Castle in the Sky (Tenkû no shiro Rapy...</td>
      <td>Scary Movie 3</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>Raging Bull</td>
      <td>Notorious</td>
      <td>Departed, The</td>
      <td>Evil Dead II (Dead by Dawn)</td>
      <td>Superman III</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>Glory</td>
      <td>Manchurian Candidate, The</td>
      <td>Goodfellas</td>
      <td>Army of Darkness</td>
      <td>Richie Rich</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>Cinema Paradiso (Nuovo cinema Paradiso)</td>
      <td>All About Eve</td>
      <td>Casablanca</td>
      <td>Road Warrior, The (Mad Max 2)</td>
      <td>I Know What You Did Last Summer</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>In Bruges</td>
      <td>Third Man, The</td>
      <td>Dark Knight, The</td>
      <td>Cabin in the Woods, The</td>
      <td>Inspector Gadget</td>
    </tr>
  </tbody>
</table>
</div>



Regarding the calculation of the Collaborative Filtering matrices, for each cluster a movie-user matrix is ​​calculated considering only the movies of the cluster and the users who have seen them. Unlike the conventional approach in which the user-movies matrix is ​​calculated for the entire dataset, in this case the complexity of the process is reduced, since the calculated matrices are significantly smaller, facilitating the implementation of the Recommendation System.


```python
# calculation of the user-movie rating matrix of each movie cluster
frames = []
for ci in [0,1,2,3,4]: # iterate over each cluster
    # movies info of current cluster
    auxJ = sqldf(f''' 
        WITH T1 AS (
            SELECT movieId FROM movies_class WHERE class={ci}
        )
        SELECT userId, T1.movieId, rating FROM T1 LEFT JOIN ratings_data
        ON T1.movieId=ratings_data.movieId ORDER BY userId, T1.movieId
    ''')

    # dataframe creation
    u_ids = np.unique(auxJ['userId'].values) # number of users
    m_ids = np.unique(auxJ['movieId'].values) # number of movies
    aux_dic = {}
    for mi in m_ids:
        aux_dic[f'{mi}'] = [0.0]*len(u_ids)
    data_ci = pd.DataFrame(index=u_ids, data=aux_dic)

    # writing of each users ratings
    for ui in u_ids:
        data_ui = sqldf(f''' 
            SELECT movieId, rating FROM auxJ
            WHERE userId={ui}            
        ''')
        for mi, ri in zip(data_ui['movieId'].values, data_ui['rating'].values):
            data_ci.at[ui, f'{mi}'] = ri
    
    # dataframe wirting
    data_ci.to_csv(f'cf_matrices/cf_matrix_{ci}.csv', index=True)
```

The recommendation system is implemented by leveraging both the trained k-Means model and the previously calculated collaborative filtering matrices. The implementation of the model is shown below, which considers the following points:
1. The recommendation system loads the pre-trained model and the collaborative filtering matrices. Thus, we ensure that the loading does not require retraining the kmeans algorithm or recalculating the user-movies matrices, which would result in an inefficient process.
2. The model predictions are implemented in the recommend function, which considers the following points:
    - The function receives a dictionary with a new user's ratings as inputs and calculates the centroid of his highest rated movies.
    - The k-Means algorithm is used to classify the centroid of the movies in a cluster.
    - The user-movie matrix of this cluster is used to apply the Collaborative Filtering technique, which consists of finding the most similar users based on their movie ratings and predicting a rating vector based on these users' ratings.
    



```python
# loading libraries
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.cluster import KMeans
import pickle
import pandas as pd
from pandasql import sqldf
from sklearn.neighbors import NearestNeighbors

class RecomenderSystem:
    def __init__(self, k=5):
        # load model
        with open('model_k_means.pkl', 'rb') as f:
            self.model = pickle.load(f)
        
        # load cf matrices
        self.cf_matrices = []
        for i in range(5):
            self.cf_matrices.append(pd.read_csv(f'cf_matrices/cf_matrix_{i}.csv', index_col=0))

        # k value for k Nearest Neighbor
        self.k = k

    
    def recomend(self, ratings):
        # read movies dataframe
        movies_data = pd.read_csv('Data/movies_full_data.csv')
        
        # identify top movies and their ids
        movies = []
        ids = []
        for ki in list(ratings.keys()):        
            if ratings[ki]>=4.0: # at leats 4.0 rating
                movies.append(ki)
                aux_ids = sqldf(f'''
                    SELECT movieId FROM movies_data
                    WHERE title="{ki}"
                ''')
                ids.append(aux_ids.values[0,0])

        # extract vector of each movie
        vectors = movies_data[movies_data['movieId'].isin(ids)]
        vectors = vectors.drop(['movieId', 'title'], axis=1)

        # centroid calculation
        cent = np.mean(vectors.values, axis=0)
        Fcent = pd.DataFrame(data=[cent], columns=('year','ratings','avg_rating','Action','Adventure','Animation','Children','Comedy','Crime','Documentary','Drama','Fantasy','Film-Noir','Horror','IMAX','Musical','Mystery','Romance','Sci-Fi','Thriller','War','Western'))
        
        # k-means classification
        ci = self.model.predict(Fcent)[0]
        
        # identification of the movies that actually belongs to this cluster
        cluster_movies = list(self.cf_matrices[ci].columns)
        auxb = [f'{mi}' in cluster_movies for mi in ids]
        movies = np.array(movies)[auxb]
        ids = np.array(ids)[auxb]
        
        # user-movies vector
        user_vector = Fcent = pd.DataFrame(data=[[0.0]*len(cluster_movies)], columns=tuple(cluster_movies))
        rs = [ratings[mi] for mi in movies]
        for id, ri in zip(ids, rs):
            user_vector.at[0, f'{id}'] = ri
        
        # get the k nearest users in Collaboprative Filtering matrix
        Xs = self.cf_matrices[ci].values # users vectors
        nn = NearestNeighbors(n_neighbors=self.k).fit(Xs)
        _, indices = nn.kneighbors(user_vector.values)
        
        # calculate the mean ratings considering only non zero values
        aux_p = Xs[indices[0]]
        pred_ratings = []
        for pi in range(aux_p.shape[1]):
            aux_r = aux_p[:,pi]
            non_zeros = np.sum(aux_r!=0.0)
            mean = np.sum(aux_r)/np.max([non_zeros, 1]) # prevent division by zero
            pred_ratings.append(mean)
        
        # identify the top recomendations, considering non-zero ratings
        auxF = pd.DataFrame(data = [pred_ratings], columns=tuple(cluster_movies))
        top_movieIds = []
        top_ratings = []
        for id in cluster_movies: # iterate over movies in cluster
            ri = auxF[id].values[0]
            if ri != 0: # non zero ratings
                top_movieIds.append(id)
                top_ratings.append(ri)
        pred_movies_ratings = pd.DataFrame({'movieId': top_movieIds,
                                            'rating': top_ratings})
        preds = sqldf(''' 
            SELECT pred_movies_ratings.movieId, movies_data.title, pred_movies_ratings.rating FROM pred_movies_ratings
            LEFT JOIN movies_data on pred_movies_ratings.movieId=movies_data.movieId ORDER BY pred_movies_ratings.rating DESC
        ''')
        print(preds)
```

## 3. Recomendation examples.
Below are some examples of the recommendation system. Note that it receives as input a dictionary of movie ratings of a new user. The system returns a set of recommendations and makes a prediction of the rating that the user would assign to each recommendation, in this way an informed decision can be made about which movies to recommend.


```python
test = RecomenderSystem()
ratings = {'Godfather, The': 4.5, 'The Island':4.5, 'Rocky II':4.0, 'Batman: Gotham Knight':4.5}
test.recomend((ratings))
```

      movieId                title  rating
    0    3671      Blazing Saddles    4.00
    1    6502        28 Days Later    4.00
    2   33794        Batman Begins    3.50
    3    2005         Goonies, The    2.00
    4    4226              Memento    2.00
    5    1302      Field of Dreams    1.25
    6     527     Schindler's List    0.50
    7    3949  Requiem for a Dream    0.50
    8   68954                   Up    0.50

