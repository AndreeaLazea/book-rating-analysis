from matplotlib import pyplot as plt
import seaborn as sns

# This function creates a correlation matrix to visualize the relationships between different attributes
# in the dataset. It uses the seaborn library to create a heatmap.

def correlation_matrix(data):
    corr_matrix= data[['year', 'num_pages', 'ratings_count', 'text_reviews_count', 'average_rating']].corr()
    plt.figure(figsize=(10, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin = -1, vmax = 1)
    plt.title("Correlation Matrix")
    plt.show()

# This function creates a pair plot to visualize the relationships between different attributes
# in the dataset. It uses the seaborn library to create a pair plot.

def pair_plot(data):

    # hue helps to color the points based on the average rating
    sns.pairplot(data, hue="average_rating", palette='husl')
    plt.title("Pair Plot for Book Attributes")
    plt.show()

# This function creates a box plot to visualize the distribution of ratings count
# for high rating vs low rating books
def box_plot(data):
    plt.figure(figsize=(10,6))
    sns.boxenplot(x='high_rating', y='ratings_count', data=data)
    plt.title("Distribution of Ratings Count for High Rating vs Low Rating Books")
    plt.xlabel("High Rating (1) vs Low Rating (0)")
    plt.ylabel("Ratings Count")
    plt.show()


#Kernel Density Plot
# This function creates a kernel density plot to visualize the distribution of average ratings
def plot_density(data):
    plt.figure(figsize=(10,6))
    sns.kdeplot(data[data['high_rating'] == 1]['average_rating'], label = 'High Rating', fill=True)
    sns.kdeplot(data[data['high_rating'] == 0]['average_rating'], label = 'Low Rating', fill=True)
    plt.title("Kernel Density Plot of Average Ratings")
    plt.xlabel("Average Rating")
    plt.ylabel("Density")

#Plot Time Series
# This function creates a time series plot of average ratings over the years
def plot_time_series(data):
    plt.figure(figsize=(12,6))
    time_data = data.groupby('year')['average_rating'].mean()
    time_data.plot()
    plt.title("Time Series of Average Rating Over Years")
    plt.xlabel("Year")
    plt.ylabel("Average Rating")
    plt.grid(True)
    plt.show()

#Scatter Plot Matrix
# This function creates a scatter plot matrix
#  to visualize the relationships between different attributes
def scatter_plot_matrix(data):
    sns.set_theme(style="ticks")
    sns.pairplot(data[['average_rating', 'num_pages', 'ratings_count', 'text_reviews_count', 'year']], diag_kind='kde')
    plt.suptitle("Scatter Plot Matrix of Book Attributes")
    plt.show()



