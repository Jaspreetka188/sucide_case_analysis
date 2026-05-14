import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import gaussian_kde

from prophet import Prophet
import logging

logging.getLogger("cmdstanpy").disabled = True


# ---------------- Style Enhancements ----------------
st.markdown("""
<style>
.stApp { background-color: #f8f9fa; }
.sidebar .stButton > button { width: 100%; }
.big-title { font-size: 28px; font-weight:700; color:#2E8B57; }
.section-header { font-size:22px; font-weight:600; margin-top:20px; }
</style>
""", unsafe_allow_html=True)

# ---------------- Load & Clean ----------------
@st.cache_data
def load_and_clean(path="sucide_case.csv"):
    df_raw = pd.read_csv(path)
    df = df_raw.copy()

    # Clean column names
    def clean_col(c):
        c = c.lower().strip()
        c = c.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")
        return c

    df.columns = [clean_col(c) for c in df.columns]

    # Rename variants
    rename_map = {
        "suicides/100k_pop": "suicides_per_100k",
        "suicides_100kpop": "suicides_per_100k",
        "gdp_per_capita_$": "gdp_per_capita",
        "gdp_for_year_$": "gdp_for_year"
    }
    df.rename(columns=rename_map, inplace=True)

    # Convert numeric
    num_cols = ["year", "suicides_no", "population", "suicides_per_100k", "gdp_per_capita", "gdp_for_year"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop duplicates
    df.drop_duplicates(inplace=True)

    # Cast categories
    for c in ["country", "sex", "age", "generation"]:
        if c in df.columns:
            df[c] = df[c].astype("category")

    return df_raw, df

try:
    df_raw, df = load_and_clean()
except FileNotFoundError:
    st.error("CSV file missing. Please add `sucide_case.csv` in the app folder.")
    st.stop()

# ---------------- Title ----------------
st.set_page_config(
    page_title="Global Suicide Analytics",
    page_icon="🌍",
    layout="wide"
)
st.markdown("""
<style>

/* Main Title */
.main-title{
    font-size:52px;
    font-weight:800;
    color:#1E293B;
    margin-bottom:18px;
    line-height:1.2;
    letter-spacing:0.5px;
}

/* Description Text */
.main-desc{
    font-size:28px;
    color:#475569;
    line-height:1.8;
    font-weight:400;
    max-width:1400px;
}

/* Optional spacing */
.block-container{
    padding-top:2rem;
    padding-left:2rem;
    padding-right:2rem;
}

</style>
""", unsafe_allow_html=True)
st.markdown("""
<div class="main-title">
🌍 Global Suicide Rate Analysis Dashboard
</div>

<div class="main-desc">
An interactive AI-powered analytics platform designed to explore,
visualize, and predict worldwide suicide trends using machine learning,
forecasting models, and modern data visualization techniques.

Analyze country-wise patterns, gender comparisons, age-group distributions,
and future predictions through intelligent interactive dashboards.
</div>
""", unsafe_allow_html=True)

st.divider()

# =====================================================
# METRIC CARDS
# =====================================================

col1, col2, col3, col4 = st.columns(4)

col1.metric("🌍 Countries", df["country"].nunique())
col2.metric("📅 Years", f"{df['year'].min()} - {df['year'].max()}")
col3.metric("📊 Total Records", f"{len(df):,}")
col4.metric("📈 Total Suicides", f"{df['suicides_no'].sum():,}")

st.divider()


# ---------------- Sidebar ----------------
st.sidebar.header("🔎 Filters")

dark_mode = st.sidebar.checkbox("Dark Mode")
if dark_mode:
    st.markdown("""
    <style>
      .stApp { background-color:#0f1724; color:#ddd; }
    </style>
    """, unsafe_allow_html=True)

# Year slider
years = sorted(df["year"].unique())
year_range = st.sidebar.slider("Select Year Range", min(years), max(years), (min(years), max(years)))

# Country selector
countries = sorted(df["country"].cat.categories.tolist())
selected_countries = st.sidebar.multiselect("Select Countries", countries, default=[])

# Age filter
ages = sorted(df["age"].cat.categories.tolist())
selected_ages = st.sidebar.multiselect("Age Groups", ages)

# Sex filter
sexes = sorted(df["sex"].cat.categories.tolist())
selected_sex = st.sidebar.multiselect("Sex", sexes)

# GDP filter
if "gdp_per_capita" in df.columns:
    gdp_min, gdp_max = int(df["gdp_per_capita"].min()), int(df["gdp_per_capita"].max())
    gdp_range = st.sidebar.slider("GDP per Capita Range", gdp_min, gdp_max, (gdp_min, gdp_max))
else:
    gdp_range = None

if st.sidebar.button("Reset Filters"):
    st.experimental_rerun()

# ---------------- Apply Filters ----------------
df_filtered = df[
    (df["year"].between(year_range[0], year_range[1])) &
    (df["country"].isin(selected_countries) if selected_countries else True) &
    (df["age"].isin(selected_ages) if selected_ages else True) &
    (df["sex"].isin(selected_sex) if selected_sex else True)
]

if gdp_range:
    df_filtered = df_filtered[df_filtered["gdp_per_capita"].between(gdp_range[0], gdp_range[1])]

st.success(f"Filtered Rows: {len(df_filtered):,}")

st.divider()

# ---------------- Preprocessing Section ----------------
st.header("🧹 Data Preprocessing Summary")

with st.expander("Show Preprocessing Details"):
    st.markdown("""
    ### ✔ Steps Done
    - Cleaned column names  
    - Standardized GDP and suicide rate columns  
    - Converted numeric values  
    - Removed duplicates  
    - Categorical encoding for faster filtering  

    ### ✔ Cleaned Data Preview
    """)
    st.dataframe(df.head(8))

# Missing values visual
st.subheader("Missing Value Summary")
missing = df.isnull().sum()
missing_df = pd.DataFrame({"Column": missing.index, "Missing": missing.values})
missing_df["%"] = (missing_df["Missing"] / len(df) * 100).round(2)
st.dataframe(missing_df[missing_df["Missing"] > 0])

# ---------------- EDA Section ----------------
st.header("📊 Exploratory Data Analysis")

# 1️⃣ Line Chart:Suicides over Years
import streamlit as st
import matplotlib.pyplot as plt

# Your grouped dataframe
df_year = df.groupby("year")["suicides_no"].sum()

# Create a figure
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(df_year.index, df_year.values, marker='o', linestyle='-', color='blue')
ax.set_title("Total Suicides per Year")
ax.set_xlabel("Year")
ax.set_ylabel("Number of Suicides")
ax.grid(True)

# Display in Streamlit
st.pyplot(fig)
# 3️⃣ Suicide Rate by Generation
import streamlit as st
import plotly.graph_objects as go

# Title
st.subheader("Total Suicides Over Time by Sex")

# Group and reshape the data
df_sex_year = df.groupby(["year", "sex"])["suicides_no"].sum().unstack()

# Custom color mapping
color_map = {
    "male": "royalblue",
    "female": "deeppink"
}

# Initialize figure
fig = go.Figure()

# Add traces
for sex in df_sex_year.columns:
    fig.add_trace(go.Scatter(
        x=df_sex_year.index,
        y=df_sex_year[sex],
        name=sex.capitalize(),
        stackgroup='one',
        mode='lines',
        line=dict(width=0.5, color=color_map.get(sex, "gray")),
        hovertemplate=f"Sex: {sex}<br>Year: %{{x}}<br>Suicides: %{{y}}"
    ))

# Peak year
peak_year = df_sex_year.sum(axis=1).idxmax()
peak_value = df_sex_year.sum(axis=1).max()

# Add vertical line + annotation
fig.add_vline(x=peak_year, line_dash="dash", line_color="gray")
fig.add_annotation(
    x=peak_year, y=peak_value,
    text=f"Peak Year: {peak_year}",
    showarrow=True,
    arrowhead=1,
    ax=0, ay=-40,
    bgcolor="lightgray",
    bordercolor="black"
)

# Update layout
fig.update_layout(
    xaxis_title="Year",
    yaxis_title="Number of Suicides",
    hovermode="x unified",
    template="plotly_white",
    width=950,
    height=550,
    font=dict(size=14),
    legend=dict(
        title="Sex",
        orientation="h",
        y=1.1,
        x=0.5,
        xanchor="center"
    ),
    xaxis=dict(dtick=2, showgrid=True, gridcolor='lightgrey'),
    yaxis=dict(showgrid=True, gridcolor='lightgrey')
)

# Display in Streamlit
st.plotly_chart(fig, use_container_width=True)


# 4️⃣ 

st.subheader("Top 10 Countries by Total Suicides")

# Prepare top 10 countries data
top_countries = df.groupby("country")["suicides_no"].sum().nlargest(10).sort_values(ascending=True)
top_df = top_countries.reset_index()
top_df.columns = ["Country", "Total Suicides"]

# Optional: Add emoji flags
flag_map = {
    "United States": "🇺🇸", "Russia": "🇷🇺", "Japan": "🇯🇵", "India": "🇮🇳",
    "France": "🇫🇷", "Germany": "🇩🇪", "Ukraine": "🇺🇦", "Brazil": "🇧🇷",
    "Republic of Korea": "🇰🇷", "Poland": "🇵🇱"
}
top_df["Country"] = top_df["Country"].apply(lambda x: f"{flag_map.get(x, '')} {x}")

# Create Plotly bar chart
fig = px.bar(
    top_df,
    x="Total Suicides",
    y="Country",
    orientation="h",
    color="Total Suicides",
    color_continuous_scale=px.colors.sequential.Viridis,
    text="Total Suicides",
    title="Top 10 Countries by Total Suicides",
)

# Update trace aesthetics
fig.update_traces(
    texttemplate="%{text:,}",
    textposition="inside",
    insidetextanchor="start",
    marker_line_color='black',
    marker_line_width=1.2,
    marker=dict(
        line=dict(width=1, color="black"),
        opacity=0.9
    )
)

# Layout customization
fig.update_layout(
    width=950,
    height=600,
    template="plotly_white",
    font=dict(family="Arial Black", size=15),
    title_font=dict(size=24, family="Arial Black"),
    coloraxis_showscale=False,
    xaxis_title="Total Suicides",
    yaxis_title="Country",
    xaxis=dict(showgrid=True, gridcolor="lightgrey"),
    yaxis=dict(categoryorder="total ascending"),
    margin=dict(l=120, r=40, t=70, b=40),
    plot_bgcolor="#f8f9fa"
)

# Display in Streamlit
st.plotly_chart(fig, use_container_width=True)



# 5️⃣ 
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.subheader("Suicide Analysis by Sex and Generation")

# Grouping and sorting
sex_totals = df.groupby("sex")["suicides_no"].sum().reset_index()
generation_totals = df.groupby("generation", observed=True)["suicides_no"].sum().reset_index()

# Create subplots
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=("Total Suicides by Sex", "Total Suicides by Generation")
)

# Bar for Sex
fig.add_trace(
    go.Bar(
        x=sex_totals["sex"],
        y=sex_totals["suicides_no"],
        marker_color=["#636EFA", "#EF553B"],
        text=[f"{val:,}" for val in sex_totals["suicides_no"]],
        textposition="auto",
        width=0.6,
        showlegend=False
    ),
    row=1, col=1
)

# Bar for Generation
fig.add_trace(
    go.Bar(
        x=generation_totals["generation"],
        y=generation_totals["suicides_no"],
        marker=dict(
            color=generation_totals["suicides_no"],
            colorscale="Plasma",
            showscale=True,
            colorbar=dict(
                title="Suicides",
                x=1.05,
                len=0.8
            )
        ),
        text=[f"{val:,}" for val in generation_totals["suicides_no"]],
        textposition="auto",
        width=0.6,
        showlegend=False
    ),
    row=1, col=2
)

# Layout
fig.update_layout(
    title="Suicide Analysis by Sex and Generation",
    title_font_size=24,
    height=500,
    width=1000,
    template="plotly_white",
    font=dict(family="Arial", size=14),
    margin=dict(l=50, r=90, t=80, b=50)
)

# Display using Streamlit
st.plotly_chart(fig, use_container_width=True)

# 6️⃣ Distribution KDE + Histogram
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.subheader("Polar Area Chart: Total Suicides by Age Group")

# Grouping data by age
age_data = df.groupby("age")["suicides_no"].sum().sort_index()

# Prepare values and labels
values = age_data.values.tolist()
labels = age_data.index.tolist()

# Repeat first element to close circular chart
values += [values[0]]
labels += [labels[0]]

# Custom color scale
custom_colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692']
colors = custom_colors[:len(labels)]

# Create polar plot
fig = go.Figure()

fig.add_trace(go.Scatterpolar(
    r=values,
    theta=labels,
    mode='lines+markers+text',
    fill='toself',
    line=dict(color='mediumvioletred', width=3),
    marker=dict(size=10, color='lightpink', line=dict(width=2, color='crimson')),
    text=[f'{val:,}' for val in values],
    textposition='top center',
    hovertemplate='<b>Age Group:</b> %{theta}<br><b>Total Suicides:</b> %{r:,}<extra></extra>',
    name='Total Suicides'
))

# Layout customization
fig.update_layout(
    title="Polar Area Chart: Total Suicides by Age Group",
    title_font_size=24,
    polar=dict(
        bgcolor="rgba(0,0,0,0)",
        radialaxis=dict(showticklabels=True, ticks='', gridcolor='lightgray', linecolor='gray'),
        angularaxis=dict(tickfont=dict(size=12), direction='clockwise')
    ),
    font=dict(size=14, family='Arial'),
    showlegend=False,
    template='plotly_white',
    height=650,
    width=750,
    margin=dict(t=100, b=50, l=60, r=60)
)

# Display in Streamlit
st.plotly_chart(fig, use_container_width=True)
#
import streamlit as st
import pandas as pd
import plotly.express as px

st.subheader("Year-wise Suicide Distribution by Age Group (1985–2016)")

# Ensure age column order
age_order = ['5-14 years', '15-24 years', '25-34 years', '35-54 years', '55-74 years', '75+ years']
df['age'] = pd.Categorical(df['age'], categories=age_order, ordered=True)

# Group by year and age
age_year_data = df.groupby(['year', 'age'], observed=False)['suicides_no'].sum().reset_index()

# Streamlit slider for year selection
years = sorted(df['year'].unique())
selected_year = st.slider("Select Year", min_value=int(years[0]), max_value=int(years[-1]), value=int(years[0]), step=1)

# Filter data for selected year
year_data = age_year_data[age_year_data['year'] == selected_year]

# Pie chart for the selected year
fig = px.pie(
    year_data,
    names='age',
    values='suicides_no',
    color='age',
    color_discrete_sequence=px.colors.sequential.Magma,
    title=f"Suicide Distribution by Age Group ({selected_year})",
    width=1000,
    height=700
)

fig.update_traces(
    textinfo='percent+label',
    pull=[0.05] * len(age_order),
    textfont_size=14
)

fig.update_layout(
    title_font_size=24,
    legend_title_text='Age Group',
    legend_font_size=14,
    margin=dict(t=60, b=30, l=40, r=40)
)

st.plotly_chart(fig, use_container_width=True)

#
import streamlit as st
import plotly.express as px
import pandas as pd

st.subheader("Animated Bubble Chart: Population vs Suicides Over Time")

# Sample for performance (or skip sampling if your system handles large data well)
df_sample = df.sample(1000, random_state=42)

# Clean data: drop rows with missing values in key columns
df_sample = df_sample.dropna(subset=["year", "population", "suicides_no", "gdp_per_capita", "generation", "sex"])

# Create animated bubble chart
fig = px.scatter(
    df_sample,
    x="population",
    y="suicides_no",
    size="gdp_per_capita",
    color="generation",  # can switch to "sex" if desired
    animation_frame="year",
    hover_name="country",
    size_max=40,
    color_discrete_sequence=px.colors.qualitative.Set3,
    hover_data={
        "population": True,
        "suicides_no": True,
        "gdp_per_capita": True,
        "generation": True,
        "sex": True,
        "year": False
    },
    title="Animated Bubble Chart: Population vs Suicides Over Time (Size = GDP per Capita)"
)

# Improve layout
fig.update_layout(
    xaxis_title="Population",
    yaxis_title="Number of Suicides",
    template="plotly_white",
    height=650,
    width=1000,
    font=dict(family="Segoe UI", size=14),
    margin=dict(l=60, r=60, t=80, b=60),
    legend_title_text="Generation",
    updatemenus=[{
        "buttons": [
            {
                "args": [None, {"frame": {"duration": 800, "redraw": True},
                                "fromcurrent": True}],
                "label": "▶ Play",
                "method": "animate"
            },
            {
                "args": [[None], {"frame": {"duration": 0, "redraw": True},
                                  "mode": "immediate",
                                  "transition": {"duration": 0}}],
                "label": "⏸ Pause",
                "method": "animate"
            }
        ],
        "direction": "left",
        "pad": {"r": 10, "t": 87},
        "showactive": False,
        "type": "buttons",
        "x": 0.1,
        "xanchor": "right",
        "y": 0,
        "yanchor": "top"
    }]
)

# Log scale for X-axis and grid
fig.update_xaxes(type="log", showgrid=True)
fig.update_yaxes(showgrid=True)

# Display in Streamlit
st.plotly_chart(fig, use_container_width=True)
#
import streamlit as st
import pandas as pd
import plotly.express as px

st.subheader("GDP per Capita vs Total Suicides")

# Ensure numeric column
df["suicides_no"] = pd.to_numeric(df["suicides_no"], errors="coerce")

# Create scatter plot
fig = px.scatter(
    df,
    x="gdp_per_capita",
    y="suicides_no",  # Can replace with any numeric column
    color="sex",       # Or 'generation', 'country', etc.
    hover_data=["country", "year", "population"],
    opacity=0.7,
    size_max=10,
    title="GDP per Capita vs Total Suicides",
    labels={
        "gdp_per_capita": "GDP per Capita ($)",
        "suicides_no": "Total Suicides"
    },
    template="plotly_white"
)

# Customize markers and layout
fig.update_traces(marker=dict(size=8, line=dict(width=1, color='DarkSlateGrey')))
fig.update_layout(
    width=900,
    height=550,
    title_font=dict(size=24, family="Arial Black"),
    font=dict(size=14),
    xaxis=dict(showgrid=True, gridcolor='lightgrey'),
    yaxis=dict(showgrid=True, gridcolor='lightgrey'),
    legend=dict(title="Gender", orientation="h", y=1.1, x=0.5, xanchor="center")
)

# Display in Streamlit
st.plotly_chart(fig, use_container_width=True)
#
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

st.subheader("Mean Values by Age Group (with Error Bars)")

# Let user select numeric column dynamically
numeric_column = st.selectbox(
    "Select numeric column to plot",
    options=[col for col in df.select_dtypes(include='number').columns]
)

# Prepare aggregated data
age_stats = df.groupby("age")[numeric_column].agg(['mean', 'std']).reset_index()

# Ensure age group is sorted properly
age_order = ["5-14 years", "15-24 years", "25-34 years", 
             "35-54 years", "55-74 years", "75+ years"]
age_stats["age"] = pd.Categorical(age_stats["age"], categories=age_order, ordered=True)
age_stats = age_stats.sort_values("age")

# Assign unique colors per age group
colors = px.colors.qualitative.Pastel + px.colors.qualitative.Dark24
age_stats["color"] = colors[:len(age_stats)]

# Build figure
fig = go.Figure()

# Add each age group as a separate point with error bar
for i, row in age_stats.iterrows():
    fig.add_trace(go.Scatter(
        x=[row["age"]],
        y=[row["mean"]],
        mode="markers+text",
        marker=dict(
            size=18,
            color=row["color"],
            line=dict(width=2, color='black'),
            symbol='circle'
        ),
        error_y=dict(
            type='data',
            array=[row["std"]],
            visible=True,
            color='gray',
            thickness=2,
            width=7
        ),
        text=[f"{row['mean']:.1f}"],
        textposition="top center",
        hovertemplate=(
            f"<b>Age Group:</b> {row['age']}<br>"
            f"<b>Mean:</b> {row['mean']:.2f}<br>"
            f"<b>Std Dev:</b> {row['std']:.2f}<extra></extra>"
        ),
        showlegend=False
    ))

# Add connecting dashed line
fig.add_trace(go.Scatter(
    x=age_stats["age"],
    y=age_stats["mean"],
    mode="lines",
    line=dict(color='black', width=2, dash='dot'),
    hoverinfo="skip",
    showlegend=False
))

# Final layout
fig.update_layout(
    title=f"Mean {numeric_column.replace('_',' ').title()} by Age Group (with Error Bars)",
    xaxis_title="Age Group",
    yaxis_title=f"Mean {numeric_column.replace('_',' ').title()}",
    template="plotly_white",
    width=950,
    height=550,
    font=dict(family="Arial Black", size=15),
    xaxis=dict(showgrid=False),
    yaxis=dict(showgrid=True, gridcolor='lightgray', zeroline=False),
    plot_bgcolor="#f6f6f6",
    margin=dict(l=80, r=40, t=80, b=60)
)

# Display in Streamlit
st.plotly_chart(fig, use_container_width=True)
#
import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
import plotly.graph_objects as go

st.subheader("Distribution with KDE and Histogram")

# Let user select numeric column
numeric_column = st.selectbox(
    "Select numeric column to visualize",
    options=[col for col in df.select_dtypes(include='number').columns]
)

# Clean the column
df[numeric_column] = pd.to_numeric(df[numeric_column], errors="coerce")
data = df[numeric_column].dropna()

# Mean and Median
mean_val = data.mean()
median_val = data.median()

# KDE Calculation
kde = gaussian_kde(data)
x_vals = np.linspace(data.min(), data.max(), 500)
kde_vals = kde(x_vals)

# Create Plotly figure
fig = go.Figure()

# Histogram with density normalization
fig.add_trace(go.Histogram(
    x=data,
    nbinsx=30,
    histnorm='probability density',
    name="Histogram",
    marker_color="lightblue",
    opacity=0.7,
    hovertemplate=f"{numeric_column}: "+"%{{x:.2f}}<br>Density: %{y:.4f}<extra></extra>"
))

# KDE Line
fig.add_trace(go.Scatter(
    x=x_vals,
    y=kde_vals,
    mode='lines',
    name='KDE Curve',
    line=dict(color='crimson', width=3)
))

# Mean Line
fig.add_trace(go.Scatter(
    x=[mean_val, mean_val],
    y=[0, max(kde_vals)*1.05],
    mode='lines',
    name=f'Mean: {mean_val:.2f}',
    line=dict(color='green', dash='dash', width=2)
))

# Median Line
fig.add_trace(go.Scatter(
    x=[median_val, median_val],
    y=[0, max(kde_vals)*1.05],
    mode='lines',
    name=f'Median: {median_val:.2f}',
    line=dict(color='orange', dash='dot', width=2)
))

# Layout customization
fig.update_layout(
    title=f"Distribution of {numeric_column.replace('_',' ').title()}",
    xaxis_title=numeric_column.replace('_',' ').title(),
    yaxis_title="Density",
    template="plotly_white",
    width=1000,
    height=550,
    bargap=0.05,
    font=dict(family="Arial", size=14),
    title_font=dict(size=22),
    legend=dict(borderwidth=1, bordercolor="lightgray", bgcolor="white")
)

# Display in Streamlit
st.plotly_chart(fig, use_container_width=True)
#
import streamlit as st
import pandas as pd
import plotly.express as px

st.title("World Choropleth Map")

# Let the user select a numeric column
numeric_column = st.selectbox(
    "Select numeric column to visualize on map",
    options=[col for col in df.select_dtypes(include='number').columns]
)

# Clean numeric column
df[numeric_column] = pd.to_numeric(df[numeric_column], errors="coerce")

# Aggregate by country
country_data = df.groupby("country")[numeric_column].sum().reset_index()

# Create choropleth
fig = px.choropleth(
    country_data,
    locations="country",            # Column with country names
    locationmode="country names",   # or "ISO-3" if using ISO codes
    color=numeric_column,           # Column to color by
    hover_name="country",           # Show country on hover
    color_continuous_scale=px.colors.sequential.Plasma,
    title=f"Choropleth of {numeric_column.replace('_',' ').title()} by Country"
)

# Update layout
fig.update_layout(
    template="plotly_white",
    width=1000,
    height=600,
    title_font=dict(size=24),
    geo=dict(showframe=False, showcoastlines=True),
)

# Display in Streamlit
st.plotly_chart(fig, use_container_width=True)
# =====================================================
# MACHINE LEARNING PREDICTION SECTION
# =====================================================

st.divider()

st.header("🤖 Machine Learning Suicide Prediction")

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Prepare ML Data
ml_df = df.dropna(subset=["year", "population", "suicides_no"])

# Features
X = ml_df[["year", "population"]]

# Target
y = ml_df["suicides_no"]

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = RandomForestRegressor()

model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Accuracy
mae = mean_absolute_error(y_test, predictions)

# Show Accuracy
st.subheader("📈 Model Accuracy")

st.success(f"Mean Absolute Error (MAE): {mae:.2f}")

# -------------------------------------------------
# USER INPUT FOR PREDICTION
# -------------------------------------------------

st.subheader("🔮 Predict Suicide Numbers")

year_input = st.number_input(
    "Enter Future Year",
    min_value=1985,
    max_value=2050,
    value=2025
)

population_input = st.number_input(
    "Enter Population",
    min_value=1000,
    value=500000
)

# Prediction Button
if st.button("Predict Suicide Number"):

    input_df = pd.DataFrame({
        "year": [year_input],
        "population": [population_input]
    })

    result = model.predict(input_df)

    st.success(
        f"Predicted Suicide Number: {result[0]:.2f}"
    )
    # =====================================================
# FUTURE SUICIDE FORECASTING
# =====================================================

st.divider()

st.header("🔮 Future Suicide Rate Forecasting")

st.write(
    "This section predicts future suicide trends using Prophet Forecasting Model."
)

from prophet import Prophet
import logging

logging.getLogger("cmdstanpy").disabled = True
import plotly.graph_objects as go

# -----------------------------
# PREPARE DATA
# -----------------------------
forecast_df = df.groupby("year")["suicides_no"].sum().reset_index()

# Rename columns for Prophet
forecast_df.columns = ["ds", "y"]

# Convert year to datetime
forecast_df["ds"] = pd.to_datetime(forecast_df["ds"], format="%Y")

# -----------------------------
# TRAIN MODEL
# -----------------------------
model = Prophet(
    yearly_seasonality=True,
    changepoint_prior_scale=0.5
)

model.fit(forecast_df)

# -----------------------------
# USER INPUT
# -----------------------------
years_to_predict = st.slider(
    "Select Future Prediction Years",
    min_value=1,
    max_value=20,
    value=10
)

# -----------------------------
# CREATE FUTURE DATA
# -----------------------------
future = model.make_future_dataframe(
    periods=years_to_predict,
    freq='YE'
)

# Predict
forecast = model.predict(future)

# -----------------------------
# SHOW FORECAST DATA
# -----------------------------
st.subheader("📊 Forecast Data")

forecast_table = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(years_to_predict)

forecast_table.columns = [
    "Year",
    "Predicted Suicides",
    "Lower Estimate",
    "Upper Estimate"
]

st.dataframe(forecast_table)

# -----------------------------
# INTERACTIVE PLOTLY GRAPH
# -----------------------------
st.subheader("📈 Forecast Visualization")

fig = go.Figure()

# Actual data
fig.add_trace(go.Scatter(
    x=forecast_df["ds"],
    y=forecast_df["y"],
    mode='lines+markers',
    name='Actual Data'
))

# Predicted data
fig.add_trace(go.Scatter(
    x=forecast["ds"],
    y=forecast["yhat"],
    mode='lines',
    name='Forecast Prediction'
))

# Upper bound
fig.add_trace(go.Scatter(
    x=forecast["ds"],
    y=forecast["yhat_upper"],
    mode='lines',
    line=dict(width=0),
    showlegend=False
))

# Lower bound with fill
fig.add_trace(go.Scatter(
    x=forecast["ds"],
    y=forecast["yhat_lower"],
    mode='lines',
    fill='tonexty',
    fillcolor='rgba(0,100,80,0.2)',
    line=dict(width=0),
    name='Confidence Interval'
))

# Layout
fig.update_layout(
    title="Future Suicide Rate Forecast",
    xaxis_title="Year",
    yaxis_title="Predicted Suicide Numbers",
    template="plotly_white",
    height=600
)

# Show graph
st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# FUTURE INSIGHTS
# -----------------------------
latest_prediction = forecast_table.iloc[-1]["Predicted Suicides"]

st.subheader("🧠 AI Insight")

st.info(
    f"The forecasting model predicts approximately "
    f"{latest_prediction:,.0f} suicide cases in the future forecasted year."
)