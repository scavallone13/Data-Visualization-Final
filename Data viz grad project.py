#!/usr/bin/env python
# coding: utf-8

# In[16]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


# In[17]:


kindle_sales=pd.read_csv('C:/Users/scava/OneDrive/Desktop/kindle_data-v2.csv')


# In[18]:


kindle_sales.head()


# In[19]:


# Assuming your DataFrame is named 'kindle_sales'
kindle_sales['best_seller_numeric'] = kindle_sales['isBestSeller'].map({True: 1, False: 0})


# In[20]:


plt.figure(figsize=(12, 6))  # Adjust figure size if needed
sns.barplot(x='category_name', y='best_seller_numeric', data=kindle_sales)
plt.title('Book Genre vs. Best Seller Status')
plt.xlabel('Genre')
plt.ylabel('Best Seller (1=True, 0=False)')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
plt.tight_layout()
plt.show()


# In[21]:


# Calculate the average best-seller status for each genre
genre_avg_bestseller = kindle_sales.groupby('category_name')['best_seller_numeric'].mean().sort_values(ascending=False)

# Get the top 3 genres
top_3_genres = genre_avg_bestseller.index[:3]

# Create the bar plot
plt.figure(figsize=(12, 6))
barplot = sns.barplot(x='category_name', y='best_seller_numeric', data=kindle_sales, order=genre_avg_bestseller.index)
plt.title('Book Genre vs. Best Seller Status (Top 3 Highlighted)')
plt.xlabel('Genre')
plt.ylabel('Average Best Seller Status')
plt.xticks(rotation=45, ha='right')

# Highlight the top 3 bars
for i, bar in enumerate(barplot.patches):
    if bar.get_x() + bar.get_width() / 2 in [genre_avg_bestseller.index.get_loc(genre) for genre in top_3_genres]:
        bar.set_color('red')  # Or any color you prefer

plt.tight_layout()
plt.show()


# In[22]:


import matplotlib.pyplot as plt
import seaborn as sns

# Sort by stars and select top 1000
top_1000_books = kindle_sales.sort_values(by=['stars'], ascending=False).head(1000)

# Create scatter plot
plt.figure(figsize=(10, 6))  # Adjust figure size if needed
sns.scatterplot(x='price', y='stars', data=top_1000_books)  # Changed y to 'price'
plt.title('Top 1000 Books: Stars vs. Price')  # Updated title
plt.xlabel('Star Rating')
plt.ylabel('Price')  # Updated y-axis label
plt.show()


# In[23]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Group by publisher and count bestsellers
publisher_bestsellers = kindle_sales.groupby('soldBy')['best_seller_numeric'].sum().reset_index()

# Sort by bestseller count
publisher_bestsellers = publisher_bestsellers.sort_values(by=['best_seller_numeric'], ascending=False)

# Create bar plot
plt.figure(figsize=(12, 6))
sns.barplot(x='soldBy', y='best_seller_numeric', data=publisher_bestsellers)
plt.title('Bestseller Counts by Publishing House')
plt.xlabel('Publishing House')
plt.ylabel('Number of Bestsellers')
plt.xticks(rotation=90, ha='right')  # Rotate x-axis labels for readability
plt.tight_layout()
plt.show()


# In[24]:


# Create a contingency table (cross-tabulation)
contingency_table = pd.crosstab(kindle_sales['isKindleUnlimited'], kindle_sales['isBestSeller'])

# Create a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(contingency_table, annot=True, fmt="d", cmap="YlGnBu")
plt.title('Correlation between Kindle Unlimited and Bestseller Status')
plt.xlabel('isBestSeller')
plt.ylabel('isKindleUnlimited')
plt.show()


# In[25]:


# Assuming your DataFrame is named 'kindle_sales'
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Group and calculate bestseller percentage
genre_bestseller_percentage = kindle_sales.groupby('category_name')['isBestSeller'].mean().reset_index()
genre_bestseller_percentage.rename(columns={'isBestSeller': 'bestseller_percentage'}, inplace=True)

# Sort options
sorted_asc = genre_bestseller_percentage.sort_values('bestseller_percentage', ascending=True)
sorted_desc = genre_bestseller_percentage.sort_values('bestseller_percentage', ascending=False)
original = genre_bestseller_percentage

# Create figure with default (original)
fig = go.Figure()

# Original bars (default)
fig.add_trace(go.Bar(
    x=original['category_name'],
    y=original['bestseller_percentage'],
    name='Original'
))

# Dropdown to change order
fig.update_layout(
    title='Bestseller Percentage by Genre',
    xaxis_title='Genre',
    yaxis_title='Bestseller Percentage',
    updatemenus=[
        dict(
            buttons=list([
                dict(label='Original Order',
                     method='update',
                     args=[{'x': [original['category_name']],
                            'y': [original['bestseller_percentage']]}]),
                dict(label='Sort Ascending',
                     method='update',
                     args=[{'x': [sorted_asc['category_name']],
                            'y': [sorted_asc['bestseller_percentage']]}]),
                dict(label='Sort Descending',
                     method='update',
                     args=[{'x': [sorted_desc['category_name']],
                            'y': [sorted_desc['bestseller_percentage']]}]),
            ]),
            direction="down",
        )
    ]
)

fig.show()


# In[28]:


# Count total books per genre
genre_book_counts = kindle_sales.groupby('category_name')['asin'].count().reset_index()
genre_book_counts.rename(columns={'asin': 'total_books'}, inplace=True)


# In[29]:


import plotly.graph_objects as go

# Create sorted versions of the data
sorted_asc = genre_book_counts.sort_values('total_books', ascending=True)
sorted_desc = genre_book_counts.sort_values('total_books', ascending=False)
original = genre_book_counts

# Create base figure
fig = go.Figure()

# Add original bar chart (default)
fig.add_trace(go.Bar(
    x=original['category_name'],
    y=original['total_books'],
    name='Original'
))

# Add dropdown for sorting
fig.update_layout(
    title='Total Books by Genre',
    xaxis_title='Genre',
    yaxis_title='Total Books',
    updatemenus=[
        dict(
            buttons=list([
                dict(label='Original Order',
                     method='update',
                     args=[{'x': [original['category_name']],
                            'y': [original['total_books']]}]),
                dict(label='Sort Ascending',
                     method='update',
                     args=[{'x': [sorted_asc['category_name']],
                            'y': [sorted_asc['total_books']]}]),
                dict(label='Sort Descending',
                     method='update',
                     args=[{'x': [sorted_desc['category_name']],
                            'y': [sorted_desc['total_books']]}]),
            ]),
            direction="down",
            showactive=True
        )
    ]
)

fig.show()


# In[30]:


import plotly.express as px
import pandas as pd

# Assuming your DataFrame is named 'kindle_sales'
# Sort by stars (or any other metric) and select top 100 books
top_100_books = kindle_sales.sort_values(by=['stars'], ascending=False).head(100)

# Create scatter plot
fig = px.scatter(top_100_books,
                 x='publishedDate',
                 y='stars',  # You can change this to another metric
                 hover_data=['author', 'title','category_name', 'productURL'],  # Data to show on hover
                 title='Top 100 Books by Publication Date')

fig.show()


# In[31]:


import pandas as pd
import plotly.graph_objects as go

# Group and prepare the two metrics
genre_total_books = kindle_sales.groupby('category_name')['asin'].count().reset_index()
genre_total_books.rename(columns={'asin': 'total_books'}, inplace=True)

genre_bestseller_pct = kindle_sales.groupby('category_name')['isBestSeller'].mean().reset_index()
genre_bestseller_pct.rename(columns={'isBestSeller': 'bestseller_percentage'}, inplace=True)

# Merge the two metrics into one DataFrame
combined = pd.merge(genre_total_books, genre_bestseller_pct, on='category_name')

# Sorts for total books
total_orig = combined[['category_name', 'total_books']]
total_asc = total_orig.sort_values('total_books', ascending=True)
total_desc = total_orig.sort_values('total_books', ascending=False)

# Sorts for bestseller %
bspct_orig = combined[['category_name', 'bestseller_percentage']]
bspct_asc = bspct_orig.sort_values('bestseller_percentage', ascending=True)
bspct_desc = bspct_orig.sort_values('bestseller_percentage', ascending=False)

# Base figure with original total books
fig = go.Figure()
fig.add_trace(go.Bar(x=total_orig['category_name'], y=total_orig['total_books'], name='Total Books'))

# Update menu for switching metric
fig.update_layout(
    updatemenus=[
        # Metric selection
        dict(
            buttons=list([
                dict(label='Total Books',
                     method='update',
                     args=[{'x': [total_orig['category_name']],
                            'y': [total_orig['total_books']],
                            'name': 'Total Books'},
                           {'title': 'Total Books by Genre',
                            'yaxis': {'title': 'Total Books'}}]),
                dict(label='Bestseller Percentage',
                     method='update',
                     args=[{'x': [bspct_orig['category_name']],
                            'y': [bspct_orig['bestseller_percentage']],
                            'name': 'Bestseller %'},
                           {'title': 'Bestseller Percentage by Genre',
                            'yaxis': {'title': 'Bestseller Percentage within each Genre'}}])
            ]),
            direction="down",
            showactive=True,
            x=0.1,
            xanchor="left",
            y=1.2,
            yanchor="top"
        ),
        # Sort order selection
        dict(
            buttons=list([
                dict(label='Original Order',
                     method='update',
                     args=[{'x': [total_orig['category_name']],
                            'y': [total_orig['total_books']]}]),
                dict(label='Sort Ascending',
                     method='update',
                     args=[{'x': [total_asc['category_name']],
                            'y': [total_asc['total_books']]}]),
                dict(label='Sort Descending',
                     method='update',
                     args=[{'x': [total_desc['category_name']],
                            'y': [total_desc['total_books']]}]),
            ]),
            direction="down",
            showactive=True,
            x=0.45,
            xanchor="left",
            y=1.2,
            yanchor="top"
        )
    ],
    title='Total Books by Genre',
    xaxis_title='Genre',
    yaxis_title='Total Books',
)

fig.show()


# In[24]:


import pandas as pd
import plotly.graph_objects as go

# Step 1: Compute metrics
genre_total_books = kindle_sales.groupby('category_name')['asin'].count().reset_index(name='total_books')
genre_bestseller_pct = kindle_sales.groupby('category_name')['isBestSeller'].mean().reset_index(name='bestseller_percentage')
genre_avg_rating = kindle_sales.groupby('category_name')['stars'].mean().reset_index(name='average_rating')

# Step 2: Merge all metrics
combined = genre_total_books.merge(genre_bestseller_pct, on='category_name').merge(genre_avg_rating, on='category_name')

# Step 3: Prepare sorted versions
def get_sorted(metric):
    return {
        'original': combined[['category_name', metric]],
        'asc': combined[['category_name', metric]].sort_values(metric, ascending=True),
        'desc': combined[['category_name', metric]].sort_values(metric, ascending=False)
    }

metrics = {
    'total_books': get_sorted('total_books'),
    'bestseller_percentage': get_sorted('bestseller_percentage'),
    'average_rating': get_sorted('average_rating')
}

# Step 4: Initial figure (Total Books - original)
fig = go.Figure()
fig.add_trace(go.Bar(
    x=metrics['total_books']['original']['category_name'],
    y=metrics['total_books']['original']['total_books'],
    name='Total Books'
))

# Step 5: Layout & dropdowns
fig.update_layout(
    title='Total Books by Genre',
    xaxis_title='Genre',
    yaxis_title='Total Books',
    updatemenus=[
        # Metric selection
        dict(
            buttons=[
                dict(label='Total Books',
                     method='update',
                     args=[{'x': [metrics['total_books']['original']['category_name']],
                            'y': [metrics['total_books']['original']['total_books']],
                            'name': 'Total Books'},
                           {'title': 'Total Books by Genre',
                            'yaxis': {'title': 'Total Books'}}]),
                dict(label='Bestseller Percentage',
                     method='update',
                     args=[{'x': [metrics['bestseller_percentage']['original']['category_name']],
                            'y': [metrics['bestseller_percentage']['original']['bestseller_percentage']],
                            'name': 'Bestseller %'},
                           {'title': 'Bestseller Percentage by Genre',
                            'yaxis': {'title': 'Bestseller Percentage'}}]),
                dict(label='Average Star Rating',
                     method='update',
                     args=[{'x': [metrics['average_rating']['original']['category_name']],
                            'y': [metrics['average_rating']['original']['average_rating']],
                            'name': 'Avg Rating'},
                           {'title': 'Average Star Rating by Genre',
                            'yaxis': {'title': 'Average Rating'}}])
            ],
            direction="down",
            showactive=True,
            x=0.05,
            xanchor="left",
            y=1.2,
            yanchor="top"
        ),
        # Sort order (Note: Static - updates only total_books by default)
        dict(
            buttons=[
                dict(label='Original Order',
                     method='update',
                     args=[{'x': [metrics['total_books']['original']['category_name']],
                            'y': [metrics['total_books']['original']['total_books']]}]),
                dict(label='Sort Ascending',
                     method='update',
                     args=[{'x': [metrics['total_books']['asc']['category_name']],
                            'y': [metrics['total_books']['asc']['total_books']]}]),
                dict(label='Sort Descending',
                     method='update',
                     args=[{'x': [metrics['total_books']['desc']['category_name']],
                            'y': [metrics['total_books']['desc']['total_books']]}])
            ],
            direction="down",
            showactive=True,
            x=0.4,
            xanchor="left",
            y=1.2,
            yanchor="top"
        )
    ]
)

fig.show()


# In[26]:


import pandas as pd
import plotly.graph_objects as go

# Step 1: Compute main metrics
genre_total_books = kindle_sales.groupby('category_name')['asin'].count().reset_index(name='total_books')
genre_bestseller_pct = kindle_sales.groupby('category_name')['isBestSeller'].mean().reset_index(name='bestseller_percentage')
genre_avg_rating = kindle_sales.groupby('category_name')['stars'].mean().reset_index(name='average_rating')

# Step 2: Compute top 100 breakdown (based on star rating)
top100 = kindle_sales.sort_values('stars', ascending=False).head(100)
top100_genre_counts = top100.groupby('category_name')['asin'].count().reset_index(name='top100_count')

# Step 3: Merge main metrics
combined = genre_total_books.merge(genre_bestseller_pct, on='category_name').merge(genre_avg_rating, on='category_name')

# Step 4: Helper function to sort metrics
def get_sorted(metric):
    return {
        'original': combined[['category_name', metric]],
        'asc': combined[['category_name', metric]].sort_values(metric, ascending=True),
        'desc': combined[['category_name', metric]].sort_values(metric, ascending=False)
    }

# Step 5: Build metrics dictionary
metrics = {
    'total_books': get_sorted('total_books'),
    'bestseller_percentage': get_sorted('bestseller_percentage'),
    'average_rating': get_sorted('average_rating'),
    'top100': {
        'original': top100_genre_counts.sort_values('category_name'),
        'asc': top100_genre_counts.sort_values('top100_count', ascending=True),
        'desc': top100_genre_counts.sort_values('top100_count', ascending=False)
    }
}

# Step 6: Initial bar chart (Total Books - original)
fig = go.Figure()
fig.add_trace(go.Bar(
    x=metrics['total_books']['original']['category_name'],
    y=metrics['total_books']['original']['total_books'],
    name='Total Books'
))

# Step 7: Layout with dropdowns
fig.update_layout(
    title='Total Books by Genre',
    xaxis_title='Genre',
    yaxis_title='Total Books',
    updatemenus=[
        # Metric dropdown
        dict(
            buttons=[
                dict(label='Total Books',
                     method='update',
                     args=[{'x': [metrics['total_books']['original']['category_name']],
                            'y': [metrics['total_books']['original']['total_books']],
                            'name': 'Total Books'},
                           {'title': 'Total Books by Genre',
                            'yaxis': {'title': 'Total Books'}}]),

                dict(label='Bestseller Percentage',
                     method='update',
                     args=[{'x': [metrics['bestseller_percentage']['original']['category_name']],
                            'y': [metrics['bestseller_percentage']['original']['bestseller_percentage']],
                            'name': 'Bestseller %'},
                           {'title': 'Bestseller Percentage by Genre',
                            'yaxis': {'title': 'Bestseller Percentage'}}]),

                dict(label='Average Star Rating',
                     method='update',
                     args=[{'x': [metrics['average_rating']['original']['category_name']],
                            'y': [metrics['average_rating']['original']['average_rating']],
                            'name': 'Avg Rating'},
                           {'title': 'Average Star Rating by Genre',
                            'yaxis': {'title': 'Average Rating'}}]),

                dict(label='Top 100 Genre Breakdown',
                     method='update',
                     args=[{'x': [metrics['top100']['original']['category_name']],
                            'y': [metrics['top100']['original']['top100_count']],
                            'name': 'Top 100 Count'},
                           {'title': 'Top 100 Bestsellers by Genre (Based on Stars)',
                            'yaxis': {'title': 'Count in Top 100'}}])
            ],
            direction="down",
            showactive=True,
            x=0.05,
            xanchor="left",
            y=1.25,
            yanchor="top"
        ),

        # Sort dropdown (static: applies only to Total Books)
        dict(
            buttons=[
                dict(label='Original Order',
                     method='update',
                     args=[{'x': [metrics['total_books']['original']['category_name']],
                            'y': [metrics['total_books']['original']['total_books']]}]),
                dict(label='Sort Ascending',
                     method='update',
                     args=[{'x': [metrics['total_books']['asc']['category_name']],
                            'y': [metrics['total_books']['asc']['total_books']]}]),
                dict(label='Sort Descending',
                     method='update',
                     args=[{'x': [metrics['total_books']['desc']['category_name']],
                            'y': [metrics['total_books']['desc']['total_books']]}])
            ],
            direction="down",
            showactive=True,
            x=0.4,
            xanchor="left",
            y=1.25,
            yanchor="top"
        )
    ]
)

fig.show()


# In[31]:


import pandas as pd
import plotly.express as px

# 1. Extract publication year
kindle_sales['year'] = pd.to_datetime(kindle_sales['publishedDate'], errors='coerce').dt.year

# 2. Filter out rows with missing years or genres
filtered = kindle_sales.dropna(subset=['year', 'category_name'])

# 3. Group by year and genre to count bestsellers
bestseller_counts = (
    filtered.groupby(['year', 'category_name'])['isBestSeller']
    .sum()
    .reset_index(name='bestseller_count')
)

# 4. Calculate percent change within each genre
bestseller_counts['percent_change'] = (
    bestseller_counts
    .sort_values(['category_name', 'year'])
    .groupby('category_name')['bestseller_count']
    .pct_change() * 100
)

# 5. Drop NaN percent change rows (first year in each genre)
bestseller_trends = bestseller_counts.dropna(subset=['percent_change'])

# 6. Plot line chart
fig = px.line(
    bestseller_trends,
    x='year',
    y='percent_change',
    color='category_name',
    title='Year-over-Year Percent Change in Bestseller Count per Genre',
    labels={'percent_change': 'Percent Change (%)', 'year': 'Year', 'category_name': 'Genre'}
)

# 7. Force x-axis to start at 1990
fig.update_layout(
    xaxis=dict(range=[1995, bestseller_trends['year'].max()]),
    yaxis_tickformat=".1f"
)

fig.show()


# In[32]:


import pandas as pd
import plotly.express as px

# Step 1: Parse year from publishedDate
kindle_sales['year'] = pd.to_datetime(kindle_sales['publishedDate'], errors='coerce').dt.year
kindle_sales = kindle_sales.dropna(subset=['year', 'category_name'])

# Step 2: Define top 100 books *per year* based on stars
top100_by_year = (
    kindle_sales
    .sort_values(['year', 'stars'], ascending=[True, False])
    .groupby('year')
    .head(100)
)

# Step 3: Count number of top books per genre per year
genre_trends_top100 = (
    top100_by_year
    .groupby(['year', 'category_name'])
    .size()
    .reset_index(name='top100_count')
)

# Optional: Filter for more recent years
genre_trends_top100 = genre_trends_top100[genre_trends_top100['year'] >= 1990]

# Step 4: Line chart of genre counts over time
fig = px.line(
    genre_trends_top100,
    x='year',
    y='top100_count',
    color='category_name',
    title='Genre Representation in Top 100 Books Over Time',
    labels={'top100_count': 'Books in Top 100', 'year': 'Year', 'category_name': 'Genre'}
)

fig.update_layout(
    xaxis=dict(range=[1990, genre_trends_top100['year'].max()])
)

fig.show()


# In[33]:


import pandas as pd
import plotly.graph_objects as go
import numpy as np

# Mock data (replace with your actual kindle_sales DataFrame)
np.random.seed(0)
years = np.random.choice(range(1980, 2025), size=1000)
genres = np.random.choice(['Romance', 'Mystery', 'Sci-Fi', 'Non-Fiction', 'Fantasy'], size=1000)
stars = np.random.uniform(3.0, 5.0, size=1000)
isBestSeller = np.random.choice([0, 1], size=1000)
kindle_sales = pd.DataFrame({
    'publishedDate': pd.to_datetime(years, format='%Y'),
    'category_name': genres,
    'stars': stars,
    'isBestSeller': isBestSeller
})

# Extract year
kindle_sales['year'] = pd.to_datetime(kindle_sales['publishedDate'], errors='coerce').dt.year
kindle_sales = kindle_sales.dropna(subset=['year', 'category_name'])

# ---- Metric 1: Total Books ----
genre_total_books = kindle_sales.groupby('category_name')['isBestSeller'].count().reset_index(name='total_books')

# ---- Metric 2: Bestseller Percentage ----
genre_bestseller_pct = kindle_sales.groupby('category_name')['isBestSeller'].mean().reset_index(name='bestseller_percentage')

# ---- Metric 3: Average Rating ----
genre_avg_rating = kindle_sales.groupby('category_name')['stars'].mean().reset_index(name='average_rating')

# ---- Metric 4: Top 100 All-Time Genre Breakdown ----
top100 = kindle_sales.sort_values('stars', ascending=False).head(100)
top100_genre_counts = top100.groupby('category_name')['stars'].count().reset_index(name='top100_count')

# ---- Metric 5: Top 100 per Year Genre Breakdown ----
top100_by_year = kindle_sales.sort_values(['year', 'stars'], ascending=[True, False]).groupby('year').head(100)
genre_trends_top100 = top100_by_year.groupby(['year', 'category_name']).size().reset_index(name='top100_count')
genre_trends_top100 = genre_trends_top100[genre_trends_top100['year'] >= 1990]

# ---- Metric 6: Percent Change in Bestseller Count ----
bestseller_counts = kindle_sales.groupby(['year', 'category_name'])['isBestSeller'].sum().reset_index(name='bestseller_count')
bestseller_counts['percent_change'] = (
    bestseller_counts.sort_values(['category_name', 'year'])
    .groupby('category_name')['bestseller_count']
    .pct_change() * 100
)
bestseller_trends = bestseller_counts.dropna(subset=['percent_change'])
bestseller_trends = bestseller_trends[bestseller_trends['year'] >= 1990]

# ---- Create Figure ----
fig = go.Figure()

# Add bar traces
fig.add_trace(go.Bar(x=genre_total_books['category_name'], y=genre_total_books['total_books'], name='Total Books', visible=True))
fig.add_trace(go.Bar(x=genre_bestseller_pct['category_name'], y=genre_bestseller_pct['bestseller_percentage'], name='Bestseller %', visible=False))
fig.add_trace(go.Bar(x=genre_avg_rating['category_name'], y=genre_avg_rating['average_rating'], name='Avg Rating', visible=False))
fig.add_trace(go.Bar(x=top100_genre_counts['category_name'], y=top100_genre_counts['top100_count'], name='Top 100 Genre Count', visible=False))

# Add line traces: Top 100 over time
for genre in genre_trends_top100['category_name'].unique():
    df = genre_trends_top100[genre_trends_top100['category_name'] == genre]
    fig.add_trace(go.Scatter(x=df['year'], y=df['top100_count'], mode='lines', name=f'{genre} (Top 100 Trend)', visible=False))

# Add line traces: Bestseller % change over time
for genre in bestseller_trends['category_name'].unique():
    df = bestseller_trends[bestseller_trends['category_name'] == genre]
    fig.add_trace(go.Scatter(x=df['year'], y=df['percent_change'], mode='lines', name=f'{genre} (% Change)', visible=False))

# Visibility masks
n_top100 = len(genre_trends_top100['category_name'].unique())
n_change = len(bestseller_trends['category_name'].unique())

visibility = {
    'Total Books': [True] + [False]*(3 + n_top100 + n_change),
    'Bestseller Percentage': [False, True] + [False]*(2 + n_top100 + n_change),
    'Average Rating': [False, False, True] + [False]*(1 + n_top100 + n_change),
    'Top 100 Genre Breakdown': [False]*3 + [True] + [False]*(n_top100 + n_change),
    'Top 100 Over Time': [False]*4 + [True]*n_top100 + [False]*n_change,
    '% Change Bestsellers': [False]*(4 + n_top100) + [True]*n_change
}

# Add dropdown menu
fig.update_layout(
    updatemenus=[
        dict(
            buttons=[dict(label=k, method='update', args=[{'visible': v}, {'title': k}]) for k, v in visibility.items()],
            direction="down",
            x=0.01,
            xanchor="left",
            y=1.2,
            yanchor="top"
        )
    ],
    title="Total Books",
    xaxis_title="Genre or Year",
    yaxis_title="Value",
    height=600
)

fig.show()


# In[34]:


import pandas as pd
import plotly.graph_objects as go
import numpy as np

# Mock data (replace with your actual kindle_sales DataFrame)
np.random.seed(0)
years = np.random.choice(range(1980, 2025), size=1000)
genres = np.random.choice(['Romance', 'Mystery', 'Sci-Fi', 'Non-Fiction', 'Fantasy'], size=1000)
stars = np.random.uniform(3.0, 5.0, size=1000)
isBestSeller = np.random.choice([0, 1], size=1000)
kindle_sales = pd.DataFrame({
    'publishedDate': pd.to_datetime(years, format='%Y'),
    'category_name': genres,
    'stars': stars,
    'isBestSeller': isBestSeller
})

# Extract year from publication date
kindle_sales['year'] = pd.to_datetime(kindle_sales['publishedDate'], errors='coerce').dt.year
kindle_sales = kindle_sales.dropna(subset=['year', 'category_name'])

# --- Metric 1: Top 100 count per genre per year ---
top100_by_year = kindle_sales.sort_values(['year', 'stars'], ascending=[True, False]).groupby('year').head(100)
genre_trends_top100 = top100_by_year.groupby(['year', 'category_name']).size().reset_index(name='top100_count')
genre_trends_top100 = genre_trends_top100[genre_trends_top100['year'] >= 1990]

# --- Metric 2: Percent change in bestseller count ---
bestseller_counts = kindle_sales.groupby(['year', 'category_name'])['isBestSeller'].sum().reset_index(name='bestseller_count')
bestseller_counts['percent_change'] = (
    bestseller_counts.sort_values(['category_name', 'year'])
    .groupby('category_name')['bestseller_count']
    .pct_change() * 100
)
bestseller_trends = bestseller_counts.dropna(subset=['percent_change'])
bestseller_trends = bestseller_trends[bestseller_trends['year'] >= 1990]

# --- Create combined line chart ---
fig = go.Figure()

# Add Top 100 count lines (solid)
for genre in genre_trends_top100['category_name'].unique():
    df = genre_trends_top100[genre_trends_top100['category_name'] == genre]
    fig.add_trace(go.Scatter(
        x=df['year'], y=df['top100_count'], mode='lines', name=f'{genre} - Top 100 Count'
    ))

# Add bestseller % change lines (dashed)
for genre in bestseller_trends['category_name'].unique():
    df = bestseller_trends[bestseller_trends['category_name'] == genre]
    fig.add_trace(go.Scatter(
        x=df['year'], y=df['percent_change'], mode='lines',
        name=f'{genre} - % Change Bestsellers', line=dict(dash='dot')
    ))

# Layout
fig.update_layout(
    title="Genre Trends: Top 100 Book Counts and Bestseller % Change Over Time",
    xaxis_title="Year",
    yaxis_title="Value",
    height=600
)

fig.show()


# In[37]:


print(kindle_sales.dtypes)


# In[33]:


import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Simulate sample data (replace this with your real kindle_sales DataFrame)
np.random.seed(0)
years = np.random.choice(range(1980, 2025), size=1000)
genres = np.random.choice(['Romance', 'Mystery', 'Sci-Fi', 'Non-Fiction', 'Fantasy'], size=1000)
stars = np.random.uniform(3.0, 5.0, size=1000)
isBestSeller = np.random.choice([0, 1], size=1000)
kindle_sales = pd.DataFrame({
    'publishedDate': pd.to_datetime(years, format='%Y'),
    'category_name': genres,
    'stars': stars,
    'isBestSeller': isBestSeller
})

# Preprocessing
kindle_sales['year'] = pd.to_datetime(kindle_sales['publishedDate'], errors='coerce').dt.year
kindle_sales = kindle_sales.dropna(subset=['year', 'category_name'])

# Top 100 per year
top100_by_year = kindle_sales.sort_values(['year', 'stars'], ascending=[True, False]).groupby('year').head(100)
genre_trends_top100 = top100_by_year.groupby(['year', 'category_name']).size().reset_index(name='top100_count')
genre_trends_top100 = genre_trends_top100[genre_trends_top100['year'] >= 1990]

# Bestseller % change
bestseller_counts = kindle_sales.groupby(['year', 'category_name'])['isBestSeller'].sum().reset_index(name='bestseller_count')
bestseller_counts['percent_change'] = (
    bestseller_counts.sort_values(['category_name', 'year'])
    .groupby('category_name')['bestseller_count']
    .pct_change() * 100
)
bestseller_trends = bestseller_counts.dropna(subset=['percent_change'])
bestseller_trends = bestseller_trends[bestseller_trends['year'] >= 1990]

# Create interactive figure
fig = go.Figure()

# Add top 100 count lines (solid)
for genre in genre_trends_top100['category_name'].unique():
    df = genre_trends_top100[genre_trends_top100['category_name'] == genre]
    fig.add_trace(go.Scatter(x=df['year'], y=df['top100_count'], mode='lines', name=f'{genre} - Top 100 Count', visible=True))

# Add bestseller % change lines (dotted)
for genre in bestseller_trends['category_name'].unique():
    df = bestseller_trends[bestseller_trends['category_name'] == genre]
    fig.add_trace(go.Scatter(x=df['year'], y=df['percent_change'], mode='lines', name=f'{genre} - % Change Bestsellers', line=dict(dash='dot'), visible=False))

# Define visibility masks
n_top100 = genre_trends_top100['category_name'].nunique()
n_bestseller = bestseller_trends['category_name'].nunique()
visibility_top100 = [True]*n_top100 + [False]*n_bestseller
visibility_bestseller = [False]*n_top100 + [True]*n_bestseller

# Add dropdown menu
fig.update_layout(
    updatemenus=[
        dict(
            buttons=[
                dict(label='Top 100 Count Over Time',
                     method='update',
                     args=[{'visible': visibility_top100},
                           {'title': 'Top 100 Book Count per Genre Over Time',
                            'yaxis': {'title': 'Top 100 Book Count'}}]),
                dict(label='Bestseller % Change Over Time',
                     method='update',
                     args=[{'visible': visibility_bestseller},
                           {'title': 'Year-over-Year % Change in Bestseller Count per Genre',
                            'yaxis': {'title': 'Percent Change (%)'}}])
            ],
            direction='down',
            showactive=True,
            x=0.01,
            xanchor='left',
            y=1.2,
            yanchor='top'
        )
    ],
    title='Top 100 Book Count per Genre Over Time',
    xaxis_title='Year',
    yaxis_title='Top 100 Book Count',
    height=600
)

fig.show()


# In[43]:


kindle_sales=pd.read_csv('C:/Users/scava/OneDrive/Desktop/kindle_data-v2.csv')


# In[32]:


import pandas as pd
import numpy as np
import plotly.express as px

# Simulated 'author' data – remove this if your real dataset already includes authors
np.random.seed(1)
authors = np.random.choice(
    ['Author ' + str(i) for i in range(1, 100)], size=len(kindle_sales)
)
kindle_sales['author'] = authors

# Count books per author
top_authors = (
    kindle_sales.groupby('author')
    .size()
    .reset_index(name='book_count')
    .sort_values(by='book_count', ascending=False)
    .head(25)
)

fig = px.bar(
    top_authors,
    x='author',
    y='book_count',
    text='author',  # add this line
    title='Top 25 Authors by Number of Books',
    labels={'author': 'Author', 'book_count': 'Number of Books'}
)

fig.update_traces(textangle=0, textposition='outside')  # better label visibility
fig.update_layout(xaxis_tickangle=45)
fig.show()


# In[45]:


kindle_sales['author'].head()


# In[ ]:




