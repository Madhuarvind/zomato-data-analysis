import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import altair as alt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from st_aggrid import AgGrid, GridOptionsBuilder

# --- Streamlit Page Config ---
st.set_page_config(page_title="Zomato Analytics Dashboard", layout="wide")

# --- Custom Style ---
st.markdown("""
    <style>
    .main {
        background: url('https://images.unsplash.com/photo-1540189549336-e6e99c3679fe') no-repeat center center fixed;
        background-size: cover;
    }
    .block-container {
        background-color: rgba(255, 255, 255, 0.9);
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 0 12px rgba(0,0,0,0.2);
    }
    .stMetric {
        background-color: #f7f7f7;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 2px 2px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    .stMetric label {
        font-weight: bold;
        color: #E23744;
        font-size: 18px;
    }
    </style>
""", unsafe_allow_html=True)

# --- Load and preprocess data ---
@st.cache_data
def load_data():
    df = pd.read_csv('zomato_cleaned.csv')
    df['rate_cleaned'] = df['rate'].replace(['NEW', '-'], None)
    df['rate_cleaned'] = df['rate_cleaned'].astype(str).str.extract(r'(\d+\.\d+)').astype(float)
    df['approx_cost(for two people)'] = df['approx_cost(for two people)'].astype(str).str.replace(',', '', regex=False)
    df['approx_cost(for two people)'] = df['approx_cost(for two people)'].astype(float)
    df = df.dropna(subset=['rate_cleaned', 'approx_cost(for two people)', 'votes'])
    X = df[['rate_cleaned', 'approx_cost(for two people)', 'votes']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['cluster'] = kmeans.fit_predict(X_scaled)
    return df

df = load_data()

# --- Sidebar Filters ---
st.sidebar.title("\U0001F50D Filters")
type_filter = st.sidebar.multiselect("Select Restaurant Type", options=df['listed_in(type)'].unique(), default=df['listed_in(type)'].unique())
cluster_filter = st.sidebar.multiselect("Select Cluster", options=sorted(df['cluster'].unique()), default=sorted(df['cluster'].unique()))
price_min, price_max = int(df['approx_cost(for two people)'].min()), int(df['approx_cost(for two people)'].max())
price_range = st.sidebar.slider("Select Price Range", price_min, price_max, (price_min, price_max))

# --- Apply Filters ---
filtered_df = df[
    (df['listed_in(type)'].isin(type_filter)) &
    (df['cluster'].isin(cluster_filter)) &
    (df['approx_cost(for two people)'] >= price_range[0]) &
    (df['approx_cost(for two people)'] <= price_range[1])
]

# --- KPI Metrics ---
st.title("\U0001F37D Zomato Restaurant Analytics Dashboard")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("#### ⭐ Average Rating")
    st.markdown(f"<h2 style='color:#27ae60'>{round(filtered_df['rate_cleaned'].mean(), 2)}</h2>", unsafe_allow_html=True)
with col2:
    st.markdown("#### 💰 Avg Cost for Two")
    st.markdown(f"<h2 style='color:#2980b9'>₹{int(filtered_df['approx_cost(for two people)'].mean())}</h2>", unsafe_allow_html=True)
with col3:
    st.markdown("#### 🗳️ Total Votes")
    st.markdown(f"<h2 style='color:#8e44ad'>{int(filtered_df['votes'].sum())}</h2>", unsafe_allow_html=True)

# --- Visuals ---
st.markdown("---")
st.subheader("\U0001F4CA Top 10 Most Common Restaurants")
top_restaurants = filtered_df['name'].value_counts().head(10)
fig1, ax1 = plt.subplots()
top_restaurants.plot(kind='bar', color='tomato', ax=ax1)
plt.xticks(rotation=45)
st.pyplot(fig1)

st.subheader("\U0001F6F0 Delivery vs Dine-in Distribution")
fig2, ax2 = plt.subplots()
filtered_df['listed_in(type)'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax2)
ax2.set_ylabel('')
st.pyplot(fig2)

st.subheader("\U0001F4C8 Average Rating by Price Band and Type")
filtered_df['price_band'] = pd.cut(filtered_df['approx_cost(for two people)'], bins=5)
pivot = filtered_df.pivot_table(index='price_band', columns='listed_in(type)', values='rate_cleaned', aggfunc='mean')
fig3, ax3 = plt.subplots()
sns.heatmap(pivot, annot=True, cmap='YlGnBu', fmt='.2f', ax=ax3)
st.pyplot(fig3)

st.subheader("\U0001F7E0 Interactive Bubble Chart: Cost vs Rating")
fig4 = px.scatter(
    filtered_df,
    x='approx_cost(for two people)',
    y='rate_cleaned',
    size='votes',
    color='cluster',
    hover_name='name',
    hover_data=['listed_in(type)', 'votes'],
    template="plotly_white",
    title="Votes Bubble Chart"
)
st.plotly_chart(fig4, use_container_width=True)

st.subheader("\U0001F9F9 Price Distribution by Type")
fig5, ax5 = plt.subplots(figsize=(10, 6))
sns.boxplot(data=filtered_df, x='listed_in(type)', y='approx_cost(for two people)', palette='pastel', ax=ax5)
plt.xticks(rotation=45)
st.pyplot(fig5)

st.subheader("\U0001F3D7 Clustered Cost vs Rating")
fig6 = px.scatter(
    filtered_df,
    x='approx_cost(for two people)',
    y='rate_cleaned',
    color='cluster',
    hover_name='name',
    template="plotly_dark",
    title="Clusters by Rating and Price"
)
st.plotly_chart(fig6, use_container_width=True)

# --- Altair Bar Chart ---
st.subheader("\U0001F4B8 Avg Rating by Type (Altair)")
bar_chart = alt.Chart(filtered_df).mark_bar().encode(
    x=alt.X('listed_in(type):N', sort='-y'),
    y='average(rate_cleaned):Q',
    color='cluster:N',
    tooltip=['listed_in(type)', 'average(rate_cleaned)']
).properties(width=700).interactive()
st.altair_chart(bar_chart, use_container_width=True)

# --- Interactive Table ---
st.subheader("\U0001F4CB Interactive Restaurant Table")
gb = GridOptionsBuilder.from_dataframe(filtered_df[['name', 'rate_cleaned', 'votes', 'approx_cost(for two people)', 'listed_in(type)', 'cluster']])
gb.configure_pagination()
gb.configure_side_bar()
gb.configure_default_column(filterable=True, sortable=True, resizable=True)
grid_options = gb.build()
AgGrid(filtered_df, gridOptions=grid_options, height=350, fit_columns_on_grid_load=True)
