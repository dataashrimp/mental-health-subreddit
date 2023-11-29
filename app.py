# Import Statements
import streamlit as st
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import networkx as nx
import logging
import plotly.express as px
import plotly.graph_objects as go
import squarify
import datetime
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from collections import Counter
from google.cloud import storage
import io
#from bertopic import BERTopic
import json
from google.oauth2.service_account import Credentials

# Load the GCP credentials from Streamlit secrets
GCP_CREDENTIALS_PATH = st.secrets

# Create a credentials object from the dictionary
creds = Credentials.from_service_account_info(GCP_CREDENTIALS_PATH)

# Use the credentials to authenticate with GCP
storage_client = storage.Client(credentials=creds, project='cse6242-groupproject-403600')


# Streamlit page configurations
st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(layout="wide",page_title="Reddit Analysis")

# Download 'vader_lexicon' and create global SentimentIntensityAnalyzer instance
# This is more performant than creating an instance every time we need to analyze sentiment
try:
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')
sentiment_analyzer = SentimentIntensityAnalyzer()

def list_blobs_with_prefix(bucket_name, prefix, gcp_credentials):
    """Lists all the blobs in the bucket that begin with the prefix."""

    creds = Credentials.from_service_account_info(gcp_credentials)
    storage_client = storage.Client(credentials=creds)
    blobs = storage_client.list_blobs(bucket_name, prefix=prefix)
    return blobs

def read_csv_from_gcloud(bucket_name, source_blob_name, gcp_credentials):
    """Reads a CSV file from Google Cloud Storage into a pandas DataFrame."""
    creds = Credentials.from_service_account_info(gcp_credentials)
    storage_client = storage.Client(credentials=creds)
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    data = blob.download_as_bytes()
    data_stream = io.BytesIO(data)
    return pd.read_csv(data_stream)

def load_all_csvs_from_folder(bucket_name, folder_path, gcp_credentials):
    """Loads all CSV files from a specified folder in GCP bucket into a single DataFrame."""
    all_files = list_blobs_with_prefix(bucket_name, folder_path, gcp_credentials=gcp_credentials)
    all_dfs = [read_csv_from_gcloud(bucket_name, file, gcp_credentials) for file in all_files if file.endswith('.csv')]
    return pd.concat(all_dfs, ignore_index=True)

def load_data(data_file):
    """
    Load data from a CSV file.

    Parameters:
    - data_file (str): Path to the CSV file.

    Returns:
    - pd.DataFrame: Loaded DataFrame or None if an error occurs.
    """
    try:
        df = pd.read_csv(data_file)
        return df
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return None

def generate_wordcloud(text):
    """
    Generate a WordCloud image based on the input text.

    Parameters:
    - text (str): Input text for generating the WordCloud.

    Returns:
    - PIL.Image.Image: WordCloud image generated from the input text.
    """

    try:
        stop_words = nltk.corpus.stopwords.words('english')
    except LookupError:
        nltk.download('stopwords')
        stop_words = nltk.corpus.stopwords.words('english') 

    stop_words.extend(['things','the','im',"I'm",'people','much'])
    wordcloud = WordCloud(width=600, height=400, background_color='white', colormap='PRGn',stopwords=stop_words).generate(text)
    return wordcloud.to_image()

def visualize_bertopic_topics(text_data):
    """
     Visualize topics in text data using BERTopic model.

    This function initializes a BERTopic model, fits it on the input text data,
    and visualizes the topics using the BERTopic visualization.

    Parameters:
    - text_data (pd.Series or list): Input text data for topic modeling.

    Returns:
    Intertopic Distance Map
    
    """
    # Initialize BERTopic model
    topic_model = BERTopic()

    # Fit the model
    topics, _ = topic_model.fit_transform(text_data)

    # Visualize topics
    st.subheader("Topic Modeling with BERTopic")
    return topic_model.visualize_topics()

def analyze_subreddit(subreddit, df):
    # Filtering to select subreddit
    subreddit_df = df[df["subreddit"] == subreddit]

    # create columns to put plots side by side
    col1, col2 = st.columns(2)

    # Word Cloud
    with col1:
        st.subheader("Frequently Used Words in Posts")
        wordcloud = generate_wordcloud(" ".join(subreddit_df["title"].str.lower()))
        st.image(wordcloud)

    # Bar Graph of Top Posters
    with col2:
        st.subheader("Top Users Among Subreddit")
        top_posters = subreddit_df["author"].value_counts().sort_values(ascending=False).head(10)
        st.bar_chart(top_posters)

    col3, col4 = st.columns(2)

    # Subreddit Popularity
    with col3:

        # Time Series Plot
        st.subheader("Posts Popularity Over Time")
        subreddit_df['created_utc'] = pd.to_datetime(subreddit_df['created_utc'], unit='s')
        popularity_over_time = subreddit_df[['created_utc', 'num_comments']].set_index('created_utc')
        st.line_chart(popularity_over_time)

    # Sentiment Analysis
    with col4:
        st.subheader("Treemap of Most Popular Posts")
        plot_treemap(subreddit_df)
        
    col5, col6 = st.columns(2)
    with col5:
    ## statistics
        st.subheader(f"Subreddit Statistics")
        total_posts = len(subreddit_df)
        st.write(f"Total Posts: {total_posts}")

        avg_comments = subreddit_df['num_comments'].mean()
        st.write(f"Average Comments per Post: {avg_comments:.2f}")
        
        avg_ups = subreddit_df['ups'].mean()
        st.write(f"Average Number of Up Votes per Post: {avg_ups:.2f}")

        avg_downs = subreddit_df['downs'].mean()
        st.write(f"Average Number of Down Votes per Post: {avg_downs:.2f}")

        avg_score = subreddit_df['score'].median()
        st.write(f"Average Score per Post: {avg_score:.2f}")

        over_18 = subreddit_df['over_18'].mean() * 100
        st.write(f"Percentage of Posts that are NSFW: {over_18:.2f}%")

        try:
            nltk.data.find('sentiment/vader_lexicon')
        except LookupError:
            nltk.download('vader_lexicon')
        
        subreddit_df["sentiment"] = subreddit_df["title"].apply(lambda description: sentiment_analyzer.polarity_scores(description)["compound"])
        avg_sentiment = subreddit_df['sentiment'].mean()
        st.write(f"Average Sentiment of Posts: {avg_sentiment:.2f}")

    with col6:
        st.subheader(f"Top 5 Most Positive Comments")
        subreddit_df["sentiment"] = subreddit_df["title"].apply(lambda description: sentiment_analyzer.polarity_scores(description)["compound"])
        top_positive_comments = subreddit_df.nlargest(5, "sentiment")[["title", "sentiment"]]
        st.table(top_positive_comments)

    st.subheader("Comments that may need to be flagged for removal")
    flagged_comments = subreddit_df[subreddit_df["sentiment"] < -0.5][["title", "sentiment"]].head(5)
    st.table(flagged_comments)

    # Display top users with low average sentiment
    st.subheader("Top Users with Low Average Sentiment")
    top_users_low_sentiment = find_top_users_low_sentiment(subreddit_df)
    st.table(top_users_low_sentiment)

    # st.subheader("Author-Comment Network Graph")
    # G = create_network_graph(subreddit_df.sample(n=50))
    # plot_network_graph(G)



    st.subheader("Top 10 Most Frequent Words")
    # Combine titles and descriptions into a single text
    text_data = ' '.join(subreddit_df['title'].astype(str) + ' ' + subreddit_df['selftext'].astype(str))
    # Tokenize the text
    words = text_data.lower().split()

    # Remove common stop words
    stop_words = stopwords.words('english')
    stop_words.extend(['things','the','im',"I'm",'people','much',"i'm",'really','like',"i've","even"])
    filtered_words = [word for word in words if word not in stop_words]

    # Count word frequencies
    word_counts = Counter(filtered_words)

    # Display top 10 words in a bar chart
    common_words = word_counts.most_common(10)
    word_chart = pd.DataFrame(common_words, columns=['Word', 'Frequency'])
    st.bar_chart(word_chart.set_index('Word'))

    #visualize_bertopic_topics(subreddit_df['title'].astype(str) + ' ' + subreddit_df['selftext'].astype(str))

    

def find_top_users_low_sentiment(subreddit_df):
    user_sentiments = subreddit_df.groupby("author")["sentiment"].mean()
    top_users_low_sentiment = user_sentiments.nsmallest(10).reset_index()
    return top_users_low_sentiment.rename(columns={"sentiment": "Average Sentiment"})

def create_network_graph(subreddit_df):
    G = nx.Graph()
    for _, row in subreddit_df.iterrows():
        author = row["author"]
        comment = row["title"]
        if author and comment:
            G.add_node(author)
            G.add_node(comment)
            G.add_edge(author, comment)
    return G
def plot_network_graph(G):
    pos = nx.spring_layout(G, seed=42)  
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    node_x = []
    node_y = []
    node_text = []
    node_colors = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
        node_colors.append(len(list(G.neighbors(node))))

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            )
        )
    )

    node_trace.text = list(node_text)  
    node_trace.marker.color = list(node_colors) 

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=0, l=0, r=0, t=0),
                    ))

    st.plotly_chart(fig)



def plot_treemap(subreddit_df):
    """
    Plot a treemap visualization for the most popular posts in the subreddit.

    This function creates a treemap using Plotly, displaying the names
    of the most popular posts based on the number of comments.

    Parameters:
    - subreddit_df (pd.DataFrame): DataFrame containing subreddit data.

    Returns:
    Visualization
    """
    top_posts = subreddit_df.nlargest(8, "num_comments")[["title", "num_comments", "ups", "url"]]
    labels = top_posts["title"].tolist()
    sizes = top_posts["num_comments"].tolist()

    fig = px.treemap(names=labels, parents=["Top Posts"] * len(labels), values=sizes)
    st.plotly_chart(fig)

def filter_posts_by_date(posts, date_range):
    start_date, end_date = date_range
    # Convert 'created_utc' column to datetime format
    posts['created_utc'] = pd.to_datetime(posts['created_utc'], unit='s', errors='coerce')

    # Ensure that start_date and end_date are pandas Timestamps
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)

    mask = (
        posts['created_utc'].notna() & 
        (posts['created_utc'] >= start_date) & 
        (posts['created_utc'] <= end_date)
    )
    return posts[mask]

def main():
    st.title("Mental Health Subreddit Community Analysis")
    if st.button("ℹ️ Info"):
        st.sidebar.info(
            """
            This application allows Reddit moderators to gain insights into their community's activities, user behavior, and evaluate the sentiment of posts. Follow the guidelines below.

            Subreddit Selection: Use the sidebar to select the subreddit you want to analyze. 

            Date Range Selection: Adjust the start and end dates in the sidebar to filter posts based on a specific time period.
            
            Key Features:

            - Frequently Used Words: Visualizes the most frequently used words in posts. Identify common themes within your community.
            - Top Posters: Identify  top users contributing to the subreddit. 
            - Posts Popularity Over Time: Analyze the popularity of posts over time to help understand when your subreddit is most active.
            - Sentiment Analysis: Gauge  overall sentiment of posts in your subreddit. 
            - Flagged Comments: Explore comments with low sentiment scores. These may indicate posts that need attention or moderation.
            - Network Analysis: Visualize the relationships between authors and their comments. Identify patterns and connections within your community.
            - Top 10 Most Frequent Words: Identify the most commonly used words in posts. 
            """
        )

    # Sidebar to select subreddit
    bucket_name = 'groupprojectdata'

    data_folder1 = 'data_csv/Anger/'
    data_folder2 = 'data_csv/selfharm/'
    data_folder3 = 'data_csv/alcoholism/'

    df1 = load_all_csvs_from_folder(bucket_name, data_folder1, GCP_CREDENTIALS_PATH)
    df2 = load_all_csvs_from_folder(bucket_name, data_folder2, GCP_CREDENTIALS_PATH)
    df3 = load_all_csvs_from_folder(bucket_name, data_folder3, GCP_CREDENTIALS_PATH)

    df = pd.concat([df1, df2, df3], axis=0)

    df = df[['author','author_flair_text','over_18','num_comments','created_utc','ups','score','post_categories','permalink','selftext','subreddit','title','url','downs']]   
    df = df[df["author"] != '[deleted]']

    df = df[df['created_utc'].notna()]
    df = df[df['subreddit'].notna()]

    if df is not None:
        subreddit = st.sidebar.selectbox("Select a subreddit", df["subreddit"].unique())
        
       # Create date range slider in the sidebar
        # Create date range selector in the sidebar
        start_date = st.sidebar.date_input("Start Date", min_value=pd.to_datetime(df['created_utc'].min(), unit='s'), value=pd.to_datetime(df['created_utc'].min(), unit='s'))

        end_date = st.sidebar.date_input("End Date", max_value=pd.to_datetime(df['created_utc'].max(), unit='s'), value=pd.to_datetime(df['created_utc'].max(), unit='s'))


        # Filter posts based on selected date range
        filtered_posts = filter_posts_by_date(df, (start_date, end_date))
        analyze_subreddit(subreddit, filtered_posts) 
if __name__ == "__main__":
    main()
