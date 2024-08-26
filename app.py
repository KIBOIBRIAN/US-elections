import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import numpy as np
import re
from datetime import datetime

# Set page config
st.set_page_config(page_title="2024 US Presidential Election Prediction Market Analysis", layout="wide")

# Load and preprocess the data
@st.cache_data
def load_data():
    df = pd.read_csv("predictIT.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df = df[df['ContractName'].isin(['Trump', 'Harris'])]
    
    # Remove dollar signs and convert to float
    for col in ['OpenSharePrice', 'HighSharePrice', 'LowSharePrice', 'CloseSharePrice']:
        df[col] = df[col].str.replace('$', '').astype(float)
    
    return df


def parse_date_ranges(df):
    def parse_date(date_str):
        # Remove any leading/trailing whitespace and periods
        date_str = date_str.strip().rstrip('.')
        # Split the range and get the end date
        parts = date_str.split('-')
        end_date = parts[-1].strip()
        
        # If the end date doesn't have a month, add the month from the start date
        if not any(month in end_date for month in ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']):
            start_date = parts[0].strip()
            month = start_date.split('.')[0]
            end_date = f"{month}. {end_date}"
        
        # Parse the date
        return datetime.strptime(end_date + " 2024", "%b. %d %Y")

    # Apply the parse_date function to the 'Dates' column
    df['Date'] = df['Dates'].apply(parse_date)
    
    return df

@st.cache_data
def load_poll_data():
    poll_df = pd.read_csv("project358.csv")
    poll_df = parse_date_ranges(poll_df)
    poll_df['Harris'] = poll_df['Kamala Haris'].str.rstrip('%').astype('float')
    poll_df['Trump'] = poll_df['Donald Trump'].str.rstrip('%').astype('float')
    return poll_df

df = load_data()
poll_df = load_poll_data()

st.title("2024 US Presidential Election Analysis")

# Sidebar for user inputs
st.sidebar.header("Controls")

# Date range selector
start_date, end_date = st.sidebar.date_input(
    "Select date range",
    [df['Date'].min(), df['Date'].max()],
    min_value=df['Date'].min(),
    max_value=df['Date'].max()
)

# Filter data based on date range
mask = (df['Date'] >= pd.Timestamp(start_date)) & (df['Date'] <= pd.Timestamp(end_date))
filtered_df = df.loc[mask]

# Get unique candidates
candidates = filtered_df['ContractName'].unique()

# Multiselect for candidates
selected_candidates = st.sidebar.multiselect("Select candidates to display", candidates, default=candidates[:5])

# Latest probabilities
# st.header("Latest Win Probabilities")
# Poll data visualization with smoothened average line
fig_poll = go.Figure()

for candidate in ['Harris', 'Trump']:
    # Sort the data by date
    sorted_data = poll_df.sort_values('Date')
    
    # Calculate the cumulative average
    sorted_data[f'{candidate}_cum_avg'] = sorted_data[candidate].expanding().mean()
    
    # Apply additional smoothing using a rolling average
    sorted_data[f'{candidate}_smooth'] = sorted_data[f'{candidate}_cum_avg'].rolling(window=5, center=True).mean()
    
    # Plot the original poll data as scatter points
    fig_poll.add_trace(go.Scatter(x=sorted_data['Date'], y=sorted_data[candidate], mode='markers', name=f'{candidate} (Polls)',
                                  marker=dict(color='#b51a00' if candidate == 'Harris' else 'grey', size=5, opacity=0.5)))
    
    # Plot the smoothened average line
    fig_poll.add_trace(go.Scatter(x=sorted_data['Date'], y=sorted_data[f'{candidate}_smooth'], mode='lines', name=f'{candidate} (Trend)',
                                  line=dict(color='#b51a00' if candidate == 'Harris' else 'grey', width=3)))

fig_poll.update_layout(title="Who’s ahead in the national polls?", xaxis_title="Date", yaxis_title="Support (%)")
col1, col2 = st.columns(2)

col1.plotly_chart(fig_poll, use_container_width=True)

latest_date = filtered_df['Date'].max()
latest_data = filtered_df[filtered_df['Date'] == latest_date]

# Calculate total share price to normalize probabilities
total_share_price = latest_data['CloseSharePrice'].sum()

# Create a pie chart of latest probabilities
prob_df = latest_data[['ContractName', 'CloseSharePrice']].copy()
prob_df['Probability'] = prob_df['CloseSharePrice'] / total_share_price * 100
prob_df = prob_df.sort_values('Probability', ascending=False)

fig_pie = px.pie(prob_df, values='Probability', names='ContractName', title=f"Win Probabilities as of {latest_date.date()}")
fig_pie.update_traces(
    marker=dict(colors=['#b51a00', 'grey']),  # Use brand color for first slice
    textfont=dict(color='white')  # Set the value text color to white
)
fig_pie.update_layout(
    font_color="white",
)

col2.plotly_chart(fig_pie, use_container_width=True)
# latest_date = filtered_df['Date'].max()
# latest_data = filtered_df[filtered_df['Date'] == latest_date]

# # Calculate total share price to normalize probabilities
# total_share_price = latest_data['CloseSharePrice'].sum()

# # Create a dataframe for probabilities
# prob_df = latest_data[['ContractName', 'CloseSharePrice']].copy()
# prob_df['Probability'] = prob_df['CloseSharePrice'] / total_share_price * 100
# prob_df = prob_df.sort_values('Probability', ascending=False)

# # Create a horizontal bar chart that looks like tabs
# fig_tabs = go.Figure()

# colors = ['#1f77b4', '#b51a00', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

# for i, (index, row) in enumerate(prob_df.iterrows()):
#     fig_tabs.add_trace(go.Bar(
#         y=[row['ContractName']],
#         x=[row['Probability']],
#         orientation='h',
#         marker=dict(color=colors[i % len(colors)]),
#         text=[f"{row['Probability']:.1f}%"],
#         textposition='inside',
#         insidetextanchor='middle',
#         name=row['ContractName'],
#         hoverinfo='text',
#         hovertext=f"{row['ContractName']}: {row['Probability']:.1f}%"
#     ))

# fig_tabs.update_layout(
#     title=f"Win Probabilities as of {latest_date.date()}",
#     barmode='stack',
#     height=400,
#     yaxis=dict(
#         showticklabels=False,
#         showgrid=False,
#         zeroline=False
#     ),
#     xaxis=dict(
#         showticklabels=False,
#         showgrid=False,
#         zeroline=False,
#         range=[0, 100]
#     ),
#     showlegend=False,
#     margin=dict(l=0, r=0, t=40, b=0)
# )

# # Add candidate names as annotations
# for i, (index, row) in enumerate(prob_df.iterrows()):
#     fig_tabs.add_annotation(
#         x=0,
#         y=i,
#         text=row['ContractName'],
#         showarrow=False,
#         xanchor='left',
#         yanchor='middle',
#         xshift=5,
#         font=dict(color='white' if row['Probability'] > 10 else 'black')
#     )

# col2.plotly_chart(fig_tabs, use_container_width=True)


# Create the main visualization with dual y-axes
fig = make_subplots(specs=[[{"secondary_y": True}]])

for candidate in selected_candidates:
    candidate_data = filtered_df[filtered_df['ContractName'] == candidate]
    fig.add_trace(go.Scatter(x=candidate_data['Date'], y=candidate_data['CloseSharePrice'] * 100,
                             mode='lines', name=f"{candidate} (Probability)",
                             line=dict(color='#b51a00' if candidate == 'Harris' else 'grey')), secondary_y=False)
    fig.add_trace(go.Bar(x=candidate_data['Date'], y=candidate_data['TradeVolume'],
                         name=f"{candidate} (Volume)", opacity=0.3,
                         marker_color='#b51a00' if candidate == 'Harris' else 'grey'), secondary_y=True)

fig.update_layout(height=600, title_text=" Market Data Over Time")
fig.update_xaxes(title_text="Date")
fig.update_yaxes(title_text="Win Probability (%)", secondary_y=False)
fig.update_yaxes(title_text="Trade Volume", secondary_y=True)

st.plotly_chart(fig, use_container_width=True)

