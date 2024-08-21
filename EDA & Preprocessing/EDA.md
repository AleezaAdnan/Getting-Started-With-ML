
# EDA

When you first hear the term EDA, you might think welp another jargon I need to study about but do not panick, we have you covered.

Imagine your wolf pack decides to watch a movie you haven’t heard of. There is absolutely no debate about that,it will lead to a state where you find yourself puzzled with lot of questions which needs to be answered in order to make a decision. Being a good chieftain the first question you would ask, what is the cast and crew of the movie? As a regular practice,you would also watch the trailer of the movie on YouTube. Furthermore,you’d find out ratings and reviews the movie has received from the audience.

Whatever investigating measures you would take before finally buying popcorn for your clan in theater,is nothing but what data scientists in their lingo call **‘Exploratory Data Analysis’**. Now you think fine Exploratory Data Analysis but what does that mean? Not to worry, as the name indicates it's exploring the data which involves studying, and visualizing information to derive important insights. To find patterns, trends, and relationships in the data, it makes use of statistical tools and visualizations. This helps to formulate hypotheses and direct additional investigations as discussed in the earlier analogy.


## Tools

Python provides various libraries used for EDA such as:      

1.**NumPy**   
2.**Pandas**   
3.**Matplotlib**  
4.**Seaborn**  
5.**Plotly**   

When starting out, we first need to import the following libraries before using them.

```
# importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

## Types of EDA

In this section we are going to firstly, discuss the types of EDA followed by an example in the next section.

**1. Univariate Analysis**  

This type of data analysis is the most basic since it only uses one variable to gather information. Understanding the underlying sample distribution and data, as well as drawing conclusions about the population, are the usual objectives of univariate non-graphical EDA. The analysis also includes outlier detection.  

The following are some features of population distribution:

>- ***Central tendency***: Typical or middle values are related to the distribution's central tendency, or location. The most widely used statistics to quantify central tendency are mean, median, and occasionally mode; mean is the most commonly used. The median might be favoured in cases with skewed distribution or where outliers are a source of worry.  
>- ***Spread***: Spread tells us how far away from the centre we are in terms of looking for and locating information values. Two helpful spread metrics are the variance and the quality deviation. The variance is the root of the variance since it is the mean of the square of the individual deviations.  
>- ***Skewness and kurtosis***: The distribution's skewness and kurtosis are two more helpful univariate descriptors. In contrast to a normal distribution, skewness is the measure of imbalance and kurtosis may be a more nuanced indicator of peakedness.


**2. Multivariate Non-graphical**  

The multivariate non-graphical EDA technique is usually used in statistical or cross-tabulation contexts to illustrate the relationship between two or more variables.  

>- For categorical data, an extension of tabulation called cross-tabulation is extremely useful. For 2 variables, cross-tabulation is preferred by making a two-way table with column headings that match the amount of one-variable and row headings that match the amount of the opposite two variables, then filling the counts with all subjects that share an equivalent pair of levels.  
>- For each categorical variable and one quantitative variable, we create statistics for quantitative variables separately for every level of the specific variable then compare the statistics across the amount of categorical variable.  
>- Comparing the means is an off-the-cuff version of ANOVA (a statistical formula used to compare variances across the means (or average) of different groups) and comparing medians may be a robust version of one-way ANOVA.  

**3. Univariate graphical**  

Non-graphical methods are quantitative and objective, they are not able to give the complete picture of the data; therefore, graphical methods are used more as they involve a degree of subjective analysis, also are required. Common sorts of univariate graphics are:

>- ***Histogram:*** The foremost basic graph is a histogram, which may be a barplot during which each bar represents the frequency (count) or proportion (count/total count) of cases for a variety of values. Histograms are one of the simplest ways to quickly learn a lot about your data, including central tendency, spread, modality, shape and outliers.  
>- ***Stem-and-leaf plots:*** An easy substitute for a histogram may be stem-and-leaf plots. It shows all data values and therefore the shape of the distribution.  
>- ***Boxplots:*** Another very useful univariate graphical technique is that the boxplot. Boxplots are excellent at presenting information about central tendency and show robust measures of location and spread also as providing information about symmetry and outliers, although they will be misleading about aspects like multimodality. One among the simplest uses of boxplots is within the sort of side-by-side boxplots.  
>- ***Quantile-normal plots:*** The ultimate univariate graphical EDA technique is that the most intricate. it’s called the quantile-normal or QN plot or more generally the quantile-quantile or QQ plot. it’s wont to see how well a specific sample follows a specific theoretical distribution. It allows detection of non-normality and diagnosis of skewness and kurtosis

**4. Multivariate graphical**  

Multivariate graphical data uses graphics to display relationships between two or more sets of knowledge. The sole one used commonly may be a grouped barplot with each group representing one level of 1 of the variables and every bar within a gaggle representing the amount of the opposite variable.

Other common sorts of multivariate graphics are:

>- ***Scatterplot:*** For 2 quantitative variables, the essential graphical EDA technique is that the scatterplot , sohas one variable on the x-axis and one on the y-axis and therefore the point for every case in your dataset.  
>- ***Run chart:***  It’s a line graph of data plotted over time.  
>- ***Heat map:***  It’s a graphical representation of data where values are depicted by color.  
>- ***Multivariate chart:*** It’s a graphical representation of the relationships between factors and response.  
>- ***Bubble chart:*** It’s a data visualization that displays multiple circles (bubbles) in two-dimensional plot.  

Apart from these functions described above, EDA can also:  

>- ***Perform k-means clustering***: Perform k-means clustering: it’s an unsupervised learning algorithm where the info points are assigned to clusters, also referred to as k-groups, k-means clustering is usually utilized in market segmentation, image compression, and pattern recognition.  
>- EDA is often utilized in predictive models like ***linear regression***.  
>- It is also utilized in ***univariate***, ***bivariate***, and ***multivariate*** visualization for summary statistics, establishing relationships between each variable, and understanding how different fields within the data interact with one another.


