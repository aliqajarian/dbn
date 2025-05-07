import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import networkx as nx
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from pathlib import Path

class ReviewVisualizer:
    def __init__(self):
        self.colors = px.colors.qualitative.Set3
        
    def create_review_type_analysis(self, results_df: pd.DataFrame) -> go.Figure:
        """Create a sunburst chart showing review types and their relationships."""
        # Prepare data for sunburst
        review_types = []
        for _, row in results_df.iterrows():
            # Main categories
            if row['speech_analysis']['is_emotional']:
                review_types.append(('Emotional', 'Positive' if row['sentiment_analysis']['textblob_sentiment'] > 0 else 'Negative'))
            if row['speech_analysis']['is_sarcastic']:
                review_types.append(('Sarcastic', 'Positive' if row['sentiment_analysis']['textblob_sentiment'] > 0 else 'Negative'))
            if row['speech_analysis']['is_exaggerated']:
                review_types.append(('Exaggerated', 'Positive' if row['sentiment_analysis']['textblob_sentiment'] > 0 else 'Negative'))
            if row['speech_analysis']['is_personal']:
                review_types.append(('Personal', 'Positive' if row['sentiment_analysis']['textblob_sentiment'] > 0 else 'Negative'))
            if row['speech_analysis']['is_recommendation']:
                review_types.append(('Recommendation', 'Positive' if row['sentiment_analysis']['textblob_sentiment'] > 0 else 'Negative'))
        
        # Create DataFrame for sunburst
        df_sunburst = pd.DataFrame(review_types, columns=['type', 'sentiment'])
        df_sunburst['count'] = 1
        df_sunburst = df_sunburst.groupby(['type', 'sentiment']).count().reset_index()
        
        # Create sunburst chart
        fig = px.sunburst(
            df_sunburst,
            path=['type', 'sentiment'],
            values='count',
            title='Review Types and Sentiment Distribution'
        )
        return fig
    
    def create_emotion_flow(self, results_df: pd.DataFrame) -> go.Figure:
        """Create a sankey diagram showing emotion flow in reviews."""
        # Extract emotion data
        emotions = []
        for _, row in results_df.iterrows():
            sentiment = 'Positive' if row['sentiment_analysis']['textblob_sentiment'] > 0 else 'Negative'
            if row['speech_analysis']['is_emotional']:
                emotions.append(('Emotional', sentiment, 'High' if abs(row['sentiment_analysis']['textblob_sentiment']) > 0.5 else 'Low'))
            if row['speech_analysis']['is_sarcastic']:
                emotions.append(('Sarcastic', sentiment, 'High' if abs(row['sentiment_analysis']['textblob_sentiment']) > 0.5 else 'Low'))
        
        # Create sankey diagram
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=["Emotional", "Sarcastic", "Positive", "Negative", "High", "Low"],
                color=self.colors
            ),
            link=dict(
                source=[0, 0, 1, 1, 2, 2, 3, 3],
                target=[2, 3, 2, 3, 4, 5, 4, 5],
                value=[len([e for e in emotions if e[0] == 'Emotional' and e[1] == 'Positive']),
                       len([e for e in emotions if e[0] == 'Emotional' and e[1] == 'Negative']),
                       len([e for e in emotions if e[0] == 'Sarcastic' and e[1] == 'Positive']),
                       len([e for e in emotions if e[0] == 'Sarcastic' and e[1] == 'Negative']),
                       len([e for e in emotions if e[1] == 'Positive' and e[2] == 'High']),
                       len([e for e in emotions if e[1] == 'Positive' and e[2] == 'Low']),
                       len([e for e in emotions if e[1] == 'Negative' and e[2] == 'High']),
                       len([e for e in emotions if e[1] == 'Negative' and e[2] == 'Low'])]
            )
        )])
        fig.update_layout(title_text="Emotion Flow in Reviews", font_size=10)
        return fig
    
    def create_review_network(self, results_df: pd.DataFrame) -> go.Figure:
        """Create a network graph showing relationships between review types."""
        # Create graph
        G = nx.Graph()
        
        # Add nodes and edges based on review types
        for _, row in results_df.iterrows():
            types = []
            if row['speech_analysis']['is_emotional']:
                types.append('Emotional')
            if row['speech_analysis']['is_sarcastic']:
                types.append('Sarcastic')
            if row['speech_analysis']['is_exaggerated']:
                types.append('Exaggerated')
            if row['speech_analysis']['is_personal']:
                types.append('Personal')
            if row['speech_analysis']['is_recommendation']:
                types.append('Recommendation')
            
            # Add edges between types
            for i in range(len(types)):
                for j in range(i+1, len(types)):
                    if G.has_edge(types[i], types[j]):
                        G[types[i]][types[j]]['weight'] += 1
                    else:
                        G.add_edge(types[i], types[j], weight=1)
        
        # Create network visualization
        pos = nx.spring_layout(G)
        edge_trace = go.Scatter(
            x=[], y=[], line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')
        
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_trace['x'] += tuple([x0, x1, None])
            edge_trace['y'] += tuple([y0, y1, None])
        
        node_trace = go.Scatter(
            x=[], y=[], text=[], mode='markers+text', hoverinfo='text',
            marker=dict(showscale=True, colorscale='YlGnBu', size=10))
        
        for node in G.nodes():
            x, y = pos[node]
            node_trace['x'] += tuple([x])
            node_trace['y'] += tuple([y])
            node_trace['text'] += tuple([node])
        
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title='Review Type Network',
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40)
                       ))
        return fig
    
    def create_review_embedding(self, results_df: pd.DataFrame) -> go.Figure:
        """Create a 2D embedding of reviews using t-SNE."""
        # Prepare features for embedding
        features = []
        for _, row in results_df.iterrows():
            features.append([
                row['sentiment_analysis']['textblob_sentiment'],
                row['sentiment_analysis']['transformer_sentiment'],
                row['speech_analysis']['emotional_intensity'],
                int(row['speech_analysis']['is_sarcastic']),
                int(row['speech_analysis']['is_exaggerated']),
                int(row['speech_analysis']['is_personal']),
                int(row['speech_analysis']['is_recommendation'])
            ])
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        embeddings = tsne.fit_transform(features)
        
        # Create scatter plot
        fig = px.scatter(
            x=embeddings[:, 0],
            y=embeddings[:, 1],
            color=results_df['is_anomaly'],
            title='Review Embedding Space',
            labels={'x': 't-SNE 1', 'y': 't-SNE 2', 'color': 'Anomaly'}
        )
        return fig
    
    def create_review_wordcloud(self, results_df: pd.DataFrame) -> go.Figure:
        """Create word clouds for different review types."""
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Emotional Reviews', 'Sarcastic Reviews',
                          'Personal Reviews', 'Recommendation Reviews')
        )
        
        # Generate word clouds for each type
        types = ['emotional', 'sarcastic', 'personal', 'recommendation']
        for idx, review_type in enumerate(types):
            # Filter reviews by type
            mask = results_df[f'is_{review_type}']
            texts = results_df[mask]['review_text'].str.cat(sep=' ')
            
            # Generate word cloud
            wordcloud = WordCloud(width=400, height=400,
                                background_color='white').generate(texts)
            
            # Add to subplot
            fig.add_trace(
                go.Image(z=wordcloud.to_array()),
                row=idx//2 + 1, col=idx%2 + 1
            )
        
        fig.update_layout(height=800, width=800, title_text="Review Type Word Clouds")
        return fig
    
    def create_review_timeline(self, results_df: pd.DataFrame) -> go.Figure:
        """Create an interactive timeline of review patterns."""
        # Prepare timeline data
        timeline_data = []
        for _, row in results_df.iterrows():
            timeline_data.append({
                'timestamp': row.get('timestamp', pd.Timestamp.now()),
                'sentiment': row['sentiment_analysis']['textblob_sentiment'],
                'type': 'Emotional' if row['speech_analysis']['is_emotional'] else
                       'Sarcastic' if row['speech_analysis']['is_sarcastic'] else
                       'Personal' if row['speech_analysis']['is_personal'] else
                       'Recommendation' if row['speech_analysis']['is_recommendation'] else 'Other',
                'is_anomaly': row['is_anomaly']
            })
        
        df_timeline = pd.DataFrame(timeline_data)
        
        # Create timeline visualization
        fig = px.scatter(
            df_timeline,
            x='timestamp',
            y='sentiment',
            color='type',
            size='is_anomaly',
            title='Review Timeline Analysis',
            labels={'timestamp': 'Time', 'sentiment': 'Sentiment Score'}
        )
        return fig
    
    def create_all_visualizations(self, results_df: pd.DataFrame, save_path: str = None):
        """Generate all visualizations and save them."""
        visualizations = {
            'review_types': self.create_review_type_analysis(results_df),
            'emotion_flow': self.create_emotion_flow(results_df),
            'review_network': self.create_review_network(results_df),
            'review_embedding': self.create_review_embedding(results_df),
            'wordclouds': self.create_review_wordcloud(results_df),
            'timeline': self.create_review_timeline(results_df)
        }
        
        if save_path:
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)
            for name, fig in visualizations.items():
                fig.write_html(save_path / f'{name}.html')
        
        return visualizations 