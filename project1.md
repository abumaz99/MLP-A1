# Machine Learning in Python - Project 1

Due Friday, March 6th by 5 pm.

Adrian Lee, Abu Mazhar, Elliot Kovanda, Gianluca Bianchi

## 0. Setup


```python
# Install required packages
!pip install -q -r requirements.txt
```


```python
# Add any additional libraries or submodules below

# Display plots inline
%matplotlib inline

# Data libraries
import pandas as pd
import numpy as np

# Plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Plotting defaults
plt.rcParams['figure.figsize'] = (8,5)
plt.rcParams['figure.dpi'] = 80

# sklearn modules
import sklearn

import datetime
```


```python
# Load data
d = pd.read_csv("the_office.csv")
#importing second more up to date dataframe
d2 = pd.read_csv("the_office_series.csv")
```

## 1. Introduction

*This section should include a brief introduction to the task and the data (assume this is a report you are delivering to a client). If you use any additional data sources, you should introduce them here and discuss why they were included.*

*Briefly outline the approaches being used and the conclusions that you are able to draw.*


```python

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
      <th>Unnamed: 0</th>
      <th>Season</th>
      <th>EpisodeTitle</th>
      <th>About</th>
      <th>Ratings</th>
      <th>Votes</th>
      <th>Viewership</th>
      <th>Duration</th>
      <th>Date</th>
      <th>GuestStars</th>
      <th>Director</th>
      <th>Writers</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>pilot</td>
      <td>The premiere episode introduces the boss and s...</td>
      <td>7.5</td>
      <td>4936</td>
      <td>11.20</td>
      <td>23</td>
      <td>24 March 2005</td>
      <td>NaN</td>
      <td>Ken Kwapis</td>
      <td>Ricky Gervais |Stephen Merchant and Greg Daniels</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>diversity day</td>
      <td>Michael's off color remark puts a sensitivity ...</td>
      <td>8.3</td>
      <td>4801</td>
      <td>6.00</td>
      <td>23</td>
      <td>29 March 2005</td>
      <td>NaN</td>
      <td>Ken Kwapis</td>
      <td>B. J. Novak</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>1</td>
      <td>health care</td>
      <td>Michael leaves Dwight in charge of picking the...</td>
      <td>7.8</td>
      <td>4024</td>
      <td>5.80</td>
      <td>22</td>
      <td>5 April 2005</td>
      <td>NaN</td>
      <td>Ken Whittingham</td>
      <td>Paul Lieberstein</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>1</td>
      <td>alliance</td>
      <td>Just for a laugh, Jim agrees to an alliance wi...</td>
      <td>8.1</td>
      <td>3915</td>
      <td>5.40</td>
      <td>23</td>
      <td>12 April 2005</td>
      <td>NaN</td>
      <td>Bryan Gordon</td>
      <td>Michael Schur</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>1</td>
      <td>basketball</td>
      <td>Michael and his staff challenge the warehouse ...</td>
      <td>8.4</td>
      <td>4294</td>
      <td>5.00</td>
      <td>23</td>
      <td>19 April 2005</td>
      <td>NaN</td>
      <td>Greg Daniels</td>
      <td>Greg Daniels</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>183</th>
      <td>183</td>
      <td>9</td>
      <td>stairmageddon</td>
      <td>Dwight shoots Stanley with a bull tranquilizer...</td>
      <td>8.0</td>
      <td>1985</td>
      <td>3.83</td>
      <td>22</td>
      <td>11 April 2013</td>
      <td>NaN</td>
      <td>Matt Sohn</td>
      <td>Dan Sterling</td>
    </tr>
    <tr>
      <th>184</th>
      <td>184</td>
      <td>9</td>
      <td>paper airplane</td>
      <td>The employees hold a paper airplane competitio...</td>
      <td>8.0</td>
      <td>2007</td>
      <td>3.25</td>
      <td>22</td>
      <td>25 April 2013</td>
      <td>NaN</td>
      <td>Jesse Peretz</td>
      <td>Halsted Sullivan | Warren Lieberstein</td>
    </tr>
    <tr>
      <th>185</th>
      <td>185</td>
      <td>9</td>
      <td>livin' dream</td>
      <td>Dwight becomes regional manager after Andy qui...</td>
      <td>9.0</td>
      <td>2831</td>
      <td>3.51</td>
      <td>42</td>
      <td>2 May 2013</td>
      <td>Michael Imperioli</td>
      <td>Jeffrey Blitz</td>
      <td>Niki Schwartz-Wright</td>
    </tr>
    <tr>
      <th>186</th>
      <td>186</td>
      <td>9</td>
      <td>a.a.r.m.</td>
      <td>Dwight prepares for a marriage proposal and hi...</td>
      <td>9.5</td>
      <td>3914</td>
      <td>4.56</td>
      <td>43</td>
      <td>9 May 2013</td>
      <td>NaN</td>
      <td>David Rogers</td>
      <td>Brent Forrester</td>
    </tr>
    <tr>
      <th>187</th>
      <td>187</td>
      <td>9</td>
      <td>finale</td>
      <td>One year later, Dunder Mifflin employees past ...</td>
      <td>9.8</td>
      <td>10515</td>
      <td>5.69</td>
      <td>51</td>
      <td>16 May 2013</td>
      <td>Joan Cusack, Ed Begley Jr, Rachel Harris, Nanc...</td>
      <td>Ken Kwapis</td>
      <td>Greg Daniels</td>
    </tr>
  </tbody>
</table>
<p>186 rows × 12 columns</p>
</div>



## 2. Exploratory Data Analysis and Feature Engineering

Before any major analysis or feature engineering we had to make sure the data we had was fit to use by means of cleaning it. The main effort was required when dealing with the 'episode name' and 'about' columns, ensuring that we had consistency with the spelling of words, as well as merging columns which contained the same information.


```python
# data cleaning for episode names

# use lower case
d.episode_name = d.episode_name.apply(lambda x: x.lower())
d2.EpisodeTitle = d2.EpisodeTitle.apply(lambda x: x.lower())

# unify the spellings in the two data frames
d.episode_name = d.episode_name.str.replace('surveilance', 'surveillance')
d.episode_name = d.episode_name.str.replace('cover', 'cover-up')
d.episode_name = d.episode_name.str.replace('a.a.r.m', 'a.a.r.m.')
d.episode_name = d.episode_name.str.replace(' \(parts 1&2\)', '')
d2.EpisodeTitle = d2.EpisodeTitle.str.replace(': part 1', ' (part 1)')
d2.EpisodeTitle = d2.EpisodeTitle.str.replace(': part 2', ' (part 2)')
d.episode_name = d.episode_name.str.replace('&', 'and')
d.episode_name = d.episode_name.str.replace('s\*x', 'sex')

# remove the word "the" from episode names
d.episode_name = d.episode_name.str.replace('the ', '')
d2.EpisodeTitle = d2.EpisodeTitle.str.replace('the ', '')

# merge the `about` column of episodes that contains two parts
d2.loc[d2.EpisodeTitle.str.contains('delivery \(part 2\)'),'About'] = ' '.join(d2[d2.EpisodeTitle.str.contains('delivery')]['About'])
d2.loc[d2.EpisodeTitle.str.contains('niagara \(part 2\)'),'About'] = ' '.join(d2[d2.EpisodeTitle.str.contains('niagara')]['About'])

# only keep rows for the second part
d2 = d2.loc[~d2.EpisodeTitle.str.contains('delivery \(part 1\)')]
d2 = d2.loc[~d2.EpisodeTitle.str.contains('niagara \(part 1\)')]

# remove "(part 2)" from the episode names
d2.loc[d2.EpisodeTitle.str.contains('delivery \(part 2\)'), "EpisodeTitle"] = 'delivery'
d2.loc[d2.EpisodeTitle.str.contains('niagara \(part 2\)'), "EpisodeTitle"] = 'niagara'

# check whether there are still differences in episode names
set(d.episode_name.tolist()).symmetric_difference(set(d2.EpisodeTitle.tolist()))
```




    set()



Now we have the clean data we merge the two data frames, using the most up to date values for imdb rating and votes.


```python
# select columns of interest from the supplementary data frame
d2 = d2[["Ratings", "EpisodeTitle", "About", "Viewership", "Duration", "GuestStars", "Director"]]

# replace `imdb_rating` in the original dataframe with `Ratings` from the additional data frame, since the latter has a higher sample size
d.imdb_rating = d2.Ratings

# combine the two data frames using episode names
d = pd.merge(d, d2, left_on='episode_name', right_on='EpisodeTitle')

# drop `EpisodeTitle`
d = d.drop('EpisodeTitle', axis = 1)
```

### Feature Engineering: `n_lines`, `n_directions`, `n_words`, and `n_speak_char`

On inspection of the aforementioned variables we can see that they are correlated with eachother. This is an issue due to the fact that if we include all of them in our model we will likely obtain an unstable model which will not perfrom well on our test data. This correlation is seen in the pairplot below. The most highly correlated variables are those with the smallest confidence interval, in this case the 'n_lines' and 'n_words' variables. In order to account for this we introduce a new varibale, 'words_per_line', which encapsulates both of these features whilst reducing the variation of our model.

The confidence intervals between all the other variables were very large in comparison and we subsequentely decided that they were not correlated enough to be removed or altered.


```python
sns.pairplot(data=d,
            y_vars=["n_lines", "n_directions", "n_words", "n_speak_char"],
            x_vars=["n_lines", "n_directions", "n_words", "n_speak_char"],
            kind="reg");
```


![png](project1_files/project1_14_0.png)



```python
# combine `n_words` and `n_lines` into one feature
d['words_per_line'] = d.n_words/d.n_lines
```

### Feature Engineering: `air_date`, `month`


```python
words on air date
```


```python
# transform `air_date` into a `date`
d.air_date = pd.to_datetime(d.air_date)

# extract month from `air_date`
d['month'] = pd.DatetimeIndex(d.air_date).month

# imdb ratings for each month
imdb_rating_month = {}

for month in d.month:
    imdb_rating_month[str(month)] = d.imdb_rating[d.month == month].tolist()

imdb_rating_month = pd.DataFrame.from_dict(imdb_rating_month, orient='index').transpose()

# rank violin plot by mean imdb for each `main_char`
months_imdb_ranked = pd.DataFrame({'months': imdb_rating_month.columns,
          'mean_imdb_rating': [mean for mean in np.mean(imdb_rating_month)]}).sort_values(by='mean_imdb_rating', ascending=False)['months'].tolist()

# melt data frame
imdb_rating_month_melted = imdb_rating_month.melt()

# create plot
sns.violinplot(y='variable', x='value', data=imdb_rating_month_melted, color='lightgrey',order=months_imdb_ranked);
plt.title('imdb ratings for each month');
plt.xlabel('imdb rating');
plt.ylabel('month');
```


![png](project1_files/project1_18_0.png)


### Feature Engineering: `GuestStars` and `main_chars`

The first issue we encountered with these columns was the fact that they were both object types containing strings, which we cannot directly use in our model. The first step was to change `GuestStar` into a column which simply indicated the presence of a guest star or not. We chose this way as oppose to keeping the original names of the guest stars in order to ...


```python
# change GuestStars from names into whether each episode features a guest star (binary: yes/no)
d.loc[pd.notnull(d.GuestStars),"GuestStars"] = 1
d.loc[pd.isnull(d.GuestStars),"GuestStars"] = 0

d.GuestStars = d.GuestStars.astype('int64');
```

We repated this process for the `main_char` variable so that each main character now has their own column indicating whether they were present in an episode or not.


```python
# split `main_chars` column into multiple columns
main_chars_split = d.main_chars.str.split(expand=True, pat=';')
# find out the total number of episode each main character starred in
main_chars = pd.unique(main_chars_split.values.ravel('K'))
main_chars = main_chars[main_chars != None] # remove Nan's

for char in main_chars:
    d[char] = None
    
# one indicator variable for each `main_char`
for char in main_chars:
    for i in np.arange(0, len(d.main_chars), 1):
        if (char in d.main_chars[i]):
            d.at[i,char] = 1
        else:
            d.at[i,char] = 0
```

Upon analysis of the number of times each main character appeared in each episode we found that the impact of each appearance on the imdb rating was negligible. This is due to the fact that many of the characters may appeared in an episode but only briefly in comparison to others. In order to differentiate between the impact characters had on the imdb rating futher we considered the `About` column. This contained infomation about the plot and which characters were a key part of a given episode. Our idea was that if we found episodes about particular characters scored noticably higher or lower than the rest we could be certian that they were integral variables for our model.

The violion plot below shows the imdb rating by character and we can clearly see that episodes where in which 'Michael' is a key character in the plot score bertter than the rest, whilst ones where 'Erin' mentioned perform worse.


```python
# count no. of occurences of each word in `About`

descriptions = str.split(' '.join(d.About))

descriptions_words_count = {}

# initialise dictionary
for word in descriptions:
    descriptions_words_count[word] = 0 

# count
for word in descriptions:
    descriptions_words_count[word] += 1
    
# count no. of occurences of each `main_chars` in `About`
descriptions_words_count = pd.Series(descriptions_words_count)

# imdb ratings for each main character
imdb_rating_char = {}

for char in main_chars:
    imdb_rating_char[char] = d.imdb_rating[d.main_chars.str.contains(char)].tolist()
imdb_rating_char = pd.DataFrame.from_dict(imdb_rating_char, orient='index').transpose()
# rank violin plot by mean imdb for each `main_char`
main_chars_imdb_ranked = pd.DataFrame({'names': imdb_rating_char.columns,
          'mean_imdb_rating': [mean for mean in np.mean(imdb_rating_char)]}).sort_values(by='mean_imdb_rating', ascending=False)['names'].tolist()

# melt data frame
imdb_rating_char_melted = imdb_rating_char.melt()

# create plot
sns.violinplot(y='variable', x='value', data=imdb_rating_char_melted, color='lightgrey',order=main_chars_imdb_ranked);
plt.title('imdb ratings for each main character');
plt.xlabel('imdb rating');
plt.ylabel('');
```


![png](project1_files/project1_25_0.png)



```python
further talk on characters and why we included only above
```


```python
for char in main_chars:
    d[str('no_of_mentions_' + char)] = 0
    
# count the number of occurences of the names of each of the top 5 characters in 'About' 
for i in np.arange(0,len(d.About),1):
    for char in main_chars:
        if char in d.About[i]:
            d.at[i,str('no_of_mentions_' + char)] = d.About[i].count(char)  
#regression plots for noticable characters            
for char in ['no_of_mentions_Darryl',
            'no_of_mentions_Kevin',
            'no_of_mentions_Erin',
            'no_of_mentions_Michael']:
    sns.regplot(x=char, y='imdb_rating', data=d);
    plt.title(char.split("_")[-1]);
    plt.xlabel('number of mentions');
    plt.ylabel('imdb ratings');
    plt.show();
```


![png](project1_files/project1_27_0.png)



![png](project1_files/project1_27_1.png)



![png](project1_files/project1_27_2.png)



![png](project1_files/project1_27_3.png)


### Feature Engineering: `director` and `writer`

With directors we had an issue that we could not consider each director indivdually, due to the fact that there were so many which would make our model far more complex than it needed to be. On observing the data we found that there were directors who directed several episodes and some which only directed a single episode. In order to use this information to best help with our recommendation to NBC Univeral, whilst still keeping our model simple enough, we chose to split the directors into groups A,B,C and D. These correspond to directors who have written: 1 to 2, 3 to 6, 7 to 13 and 14 to 15 episodes respectively. 


```python
# split `director` column into multiple columns
director_split = d.director.str.split(expand=True, pat=';')

directors = pd.unique(director_split.values.ravel('K'))
directors = directors[directors != None] # remove Nan's

# number of episodes each director took part in
director_count = [director_split[director_split==director].count().sum() for director in directors]
# create data frame with directors' names and number of episodes featured in
director_count = pd.DataFrame({'name': directors, 'count': director_count}).set_index('name')
# sort values
director_count = director_count.sort_values(by='count', ascending=False)

# take an average of episodes directed for episodes with more than one director
director_split_count = director_split

for director in director_count.index.tolist():
    director_split_count[director_split_count == director] = director_count.loc[director,"count"]
    
director_split_avg = np.mean(director_split_count,axis=1).astype('int')


# generate new column for the groups
d['director_grouped'] = 0

d.loc[(director_split_avg >= 1)&(director_split_avg <= 2), 'director_grouped'] = 'A'
d.loc[(director_split_avg >= 3)&(director_split_avg <= 6), 'director_grouped'] = 'B'
d.loc[(director_split_avg >= 7)&(director_split_avg <= 13), 'director_grouped'] = 'C'
d.loc[(director_split_avg >= 14)&(director_split_avg <= 15), 'director_grouped'] = 'D'
```

We repeated this process with the `writer` column, this time splitting into groups A,B,C,D and E,  respectively correspoding to: 1 to 4, 5 to 9, 10 to 13, 14 to 17 and 18 to 21 episodes written.


```python
# split `writer` column into multiple columns
writer_split = d.writer.str.split(expand=True, pat=';')

writers = pd.unique(writer_split.values.ravel('K'))
writers = writers[writers != None] # remove Nan's

# number of episodes each writer took part in
writer_count = [writer_split[writer_split==writer].count().sum() for writer in writers]
# create data frame with writers' names and number of episodes featured in
writer_count = pd.DataFrame({'name': writers, 'count': writer_count}).set_index('name')
# sort values
writer_count = writer_count.sort_values(by='count', ascending=False)

# take an average of episodes written for episodes with more than one writer
writer_split_count = writer_split

for writer in writer_count.index.tolist():
    writer_split_count[writer_split_count == writer] = writer_count.loc[writer,"count"]
    
writer_split_avg = np.mean(writer_split_count,axis=1).astype('int')


# generate new column for the groups
d['writer_grouped'] = 0

d.loc[(writer_split_avg >= 1)&(writer_split_avg <= 4), 'writer_grouped'] = 'A'
d.loc[(writer_split_avg >= 5)&(writer_split_avg <= 9), 'writer_grouped'] = 'B'
d.loc[(writer_split_avg >= 10)&(writer_split_avg <= 13), 'writer_grouped'] = 'C'
d.loc[(writer_split_avg >= 14)&(writer_split_avg <= 17), 'writer_grouped'] = 'D'
d.loc[(writer_split_avg >= 18)&(writer_split_avg <= 21), 'writer_grouped'] = 'E'
```

### One-Hot Encoding and Final Inspection of the Data Frame


```python
#d = pd.get_dummies(d, columns = ['director_grouped', 'writer_grouped'])

# only keep columns that are of interest
d = d[['imdb_rating', 'total_votes', 'Viewership', 'Duration', 'GuestStars', 'words_per_line',
       'n_lines', 'n_directions', 'n_words', 'n_speak_char', 'month',
   'no_of_mentions_Kevin', 'no_of_mentions_Erin', 'no_of_mentions_Michael',
   'director_grouped_A', 'director_grouped_B', 'director_grouped_C', 'director_grouped_D',
   'writer_grouped_A', 'writer_grouped_B', 'writer_grouped_C', 'writer_grouped_D', 'writer_grouped_E']]
```

*Include a detailed discussion of the data with a particular emphasis on the features of the data that are relevant for the subsequent modeling. Including visualizations of the data is strongly encouraged - all code and plots must also be described in the write up. Think carefully about whether each plot needs to be included in your final draft - your report should include figures but they should be as focused and impactful as possible.*

*Additionally, this section should also implement and describe any preprocessing / feature engineering of the data. Specifically, this should be any code that you use to generate new columns in the data frame `d`. All of this processing is explicitly meant to occur before we split the data in to training and testing subsets. Processing that will be performed as part of an sklearn pipeline can be mentioned here but should be implemented in the following section.*

*All code and figures should be accompanied by text that provides an overview / context to what is being done or presented.*

## 3. Model Fitting and Tuning

*In this section you should detail your choice of model and describe the process used to refine and fit that model. You are strongly encouraged to explore many different modeling methods (e.g. linear regression, regression trees, lasso, etc.) but you should not include a detailed narrative of all of these attempts. At most this section should mention the methods explored and why they were rejected - most of your effort should go into describing the model you are using and your process for tuning and validatin it.*

*For example if you considered a linear regression model, a classification tree, and a lasso model and ultimately settled on the linear regression approach then you should mention that other two approaches were tried but do not include any of the code or any in depth discussion of these models beyond why they were rejected. This section should then detail is the development of the linear regression model in terms of features used, interactions considered, and any additional tuning and validation which ultimately led to your final model.* 

*This section should also include the full implementation of your final model, including all necessary validation. As with figures, any included code must also be addressed in the text of the document.*

## 4. Discussion & Conclusions


*In this section you should provide a general overview of your final model, its performance, and reliability. You should discuss what the implications of your model are in terms of the included features, predictive performance, and anything else you think is relevant.*

*This should be written with a target audience of a NBC Universal executive who is with the show and  university level mathematics but not necessarily someone who has taken a postgraduate statistical modeling course. Your goal should be to convince this audience that your model is both accurate and useful.*

*Finally, you should include concrete recommendations on what NBC Universal should do to make their reunion episode a popular as possible.*

*Keep in mind that a negative result, i.e. a model that does not work well predictively, that is well explained and justified in terms of why it failed will likely receive higher marks than a model with strong predictive performance but with poor or incorrect explinations / justifications.*

## 5. Convert Document


```python
# Run the following to render to PDF
!jupyter nbconvert --to markdown project1.ipynb
```
