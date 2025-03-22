import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import calendar

# Page configuration
st.set_page_config(
    page_title="Personal Fitness Tracker",
    page_icon="💪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {padding: 1rem;}
    .stApp {max-width: 1200px; margin: 0 auto;}
    h1, h2, h3 {color: blue; font-weight: 600;}
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .trend-good {color: #28a745;}
    .trend-bad {color: #dc3545;}
    .trend-neutral {color: #6c757d;}
    .stTabs [data-baseweb="tab-list"] {gap: 24px;}
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: black;
        border-radius: 4px 4px 0 0;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: black;
        border-bottom: 2px solid blue;
    }
</style>
""", unsafe_allow_html=True)

# Define color palette
COLOR_PALETTE = {
    'primary': '#0066cc',
    'secondary': '#6c757d',
    'success': '#28a745',
    'danger': '#dc3545',
    'warning': '#ffc107',
    'info': '#17a2b8',
    'light': '#f8f9fa',
    'dark': '#343a40',
}

# Function to calculate trend and return formatted HTML
def format_trend(current, previous, reverse=False):
    if previous == 0:
        return ""
    
    change = ((current - previous) / previous) * 100
    if abs(change) < 1:  # Less than 1% change
        return f'<span class="trend-neutral">({change:.1f}%)</span>'
    
    if (change > 0 and not reverse) or (change < 0 and reverse):
        return f'<span class="trend-good">▲ ({abs(change):.1f}%)</span>'
    else:
        return f'<span class="trend-bad">▼ ({abs(change):.1f}%)</span>'

# Function to load data
@st.cache_data(ttl=300)
def load_data():
    try:
        if os.path.exists('fitness_data_comprehensive.csv'):
            df = pd.read_csv('fitness_data_comprehensive.csv')
            df['Date'] = pd.to_datetime(df['Date'])
            return df
        else:
            st.error("Fitness data file not found. Please upload a dataset.")
            return None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Initialize session state variables
if 'user_data' not in st.session_state:
    st.session_state.user_data = load_data()

if 'user_profile' not in st.session_state:
    st.session_state.user_profile = {
        'name': 'Alex Johnson',
        'age': 32,
        'height': 175,  # cm
        'join_date': '2024-01-01'
    }

if 'goals' not in st.session_state:
    st.session_state.goals = {
        'weekly_workouts': 5,
        'weekly_calories': 2500,
        'target_weight': 70.0,
        'daily_steps': 10000
    }

# Application title
st.title("Personal Fitness Tracker")

# Navigation sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/exercise.png", width=80)
    st.header(f"Hello, {st.session_state.user_profile['name']}!")
    
    app_mode = st.radio("Navigation", ["Dashboard", "Activity Log", "Body Metrics", "Nutrition", "Sleep & Recovery", "Analysis", "Goals & Progress", "Settings"])
    
    if st.session_state.user_data is not None:
        # Calculate quick stats
        last_7_days = st.session_state.user_data[st.session_state.user_data['Date'] > 
                                             (datetime.now() - timedelta(days=7))]
        
        active_days = len(last_7_days[last_7_days['Duration(min)'] > 0])
        total_calories = last_7_days['Calories_Burned'].sum()
        avg_steps = int(last_7_days['Steps'].mean())
        
        # Display quick stats in sidebar
        st.markdown("### Quick Stats (Last 7 Days)")
        st.markdown(f"🏋️ Active Days: **{active_days}** / 7")
        st.markdown(f"🔥 Calories Burned: **{total_calories}**")
        st.markdown(f"👟 Avg. Steps: **{avg_steps}**")
        
        # Current weight trend
        if 'Weight(kg)' in st.session_state.user_data.columns:
            latest_weight = st.session_state.user_data['Weight(kg)'].iloc[-1]
            st.markdown(f"⚖️ Current Weight: **{latest_weight} kg**")

# Main content area
if st.session_state.user_data is None:
    st.warning("No fitness data found. Please upload a dataset in the Settings page.")
else:
    # Dashboard Page
    if app_mode == "Dashboard":
        st.header("Fitness Dashboard")
        
        # Date range selector
        col1, col2 = st.columns([3, 1])
        with col1:
            date_range = st.selectbox(
                "Select Period", 
                ["Last 7 Days", "Last 14 Days", "Last 30 Days", "Last 90 Days", "All Time"],
                index=0
            )
            
            # Filter data based on selected date range
            if date_range == "Last 7 Days":
                filtered_data = st.session_state.user_data[st.session_state.user_data['Date'] >= 
                                                      (datetime.now() - timedelta(days=7))]
            elif date_range == "Last 14 Days":
                filtered_data = st.session_state.user_data[st.session_state.user_data['Date'] >= 
                                                      (datetime.now() - timedelta(days=14))]
            elif date_range == "Last 30 Days":
                filtered_data = st.session_state.user_data[st.session_state.user_data['Date'] >= 
                                                      (datetime.now() - timedelta(days=30))]
            elif date_range == "Last 90 Days":
                filtered_data = st.session_state.user_data[st.session_state.user_data['Date'] >= 
                                                      (datetime.now() - timedelta(days=90))]
            else:
                filtered_data = st.session_state.user_data
        
        with col2:
            # Summary toggle
            view_mode = st.radio("View", ["Summary", "Details"], horizontal=True)
        
        # Summary Statistics
        st.subheader("Summary Statistics")
        
        # Calculate key metrics
        total_workouts = len(filtered_data[filtered_data['Duration(min)'] > 0])
        total_active_days = len(filtered_data[filtered_data['Duration(min)'] > 0])
        total_duration = filtered_data['Duration(min)'].sum()
        total_calories = filtered_data['Calories_Burned'].sum()
        avg_daily_steps = int(filtered_data['Steps'].mean())
        
        # Previous period for comparison
        if date_range == "Last 7 Days":
            prev_start = datetime.now() - timedelta(days=14)
            prev_end = datetime.now() - timedelta(days=7)
        elif date_range == "Last 14 Days":
            prev_start = datetime.now() - timedelta(days=28)
            prev_end = datetime.now() - timedelta(days=14)
        elif date_range == "Last 30 Days":
            prev_start = datetime.now() - timedelta(days=60)
            prev_end = datetime.now() - timedelta(days=30)
        elif date_range == "Last 90 Days":
            prev_start = datetime.now() - timedelta(days=180)
            prev_end = datetime.now() - timedelta(days=90)
        else:
            prev_start = None
            prev_end = None
        
        if prev_start and prev_end:
            prev_data = st.session_state.user_data[(st.session_state.user_data['Date'] >= prev_start) & 
                                               (st.session_state.user_data['Date'] < prev_end)]
            prev_workouts = len(prev_data[prev_data['Duration(min)'] > 0])
            prev_duration = prev_data['Duration(min)'].sum()
            prev_calories = prev_data['Calories_Burned'].sum()
            prev_steps = int(prev_data['Steps'].mean()) if not prev_data.empty else 0
        else:
            prev_workouts = 0
            prev_duration = 0
            prev_calories = 0
            prev_steps = 0
        
        # Display metrics in columns with trend indicators
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Active Days", f"{total_active_days}", 
                    delta=f"{total_active_days - prev_workouts}")
        with col2:
            st.metric("Total Duration", f"{total_duration} mins", 
                    delta=f"{total_duration - prev_duration} mins")
        with col3:
            st.metric("Calories Burned", f"{total_calories}", 
                    delta=f"{total_calories - prev_calories}")
        with col4:
            st.metric("Avg. Daily Steps", f"{avg_daily_steps}", 
                    delta=f"{avg_daily_steps - prev_steps}")
        
        # Create main dashboard layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Activity Timeline
            st.subheader("Activity Timeline")
            
            # Create summary data by day
            daily_summary = filtered_data.groupby('Date').agg({
                'Calories_Burned': 'sum',
                'Duration(min)': 'sum',
                'Exercise': lambda x: ', '.join(x.unique()) if len(x.unique()) <= 2 else f"{len(x.unique())} activities",
                'Steps': 'mean'
            }).reset_index()
            
            # Create a bar chart for daily activities
            fig = px.bar(
                daily_summary, 
                x='Date', 
                y='Duration(min)',
                color='Exercise',
                hover_data=['Calories_Burned', 'Steps'],
                labels={'Duration(min)': 'Duration (minutes)', 'Date': ''},
                title='Daily Activity Duration'
            )
            
            fig.update_layout(
                xaxis=dict(type='date', tickformat='%a %d %b'),
                yaxis=dict(title='Duration (minutes)'),
                hovermode='x unified',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Weight Trend
            if 'Weight(kg)' in filtered_data.columns:
                st.subheader("Weight Trend")
                
                # Create a line chart for weight trend
                weight_data = filtered_data[['Date', 'Weight(kg)']].dropna()
                
                fig = px.line(
                    weight_data, 
                    x='Date', 
                    y='Weight(kg)',
                    markers=True,
                    labels={'Weight(kg)': 'Weight (kg)', 'Date': ''},
                )
                
                fig.update_layout(
                    xaxis=dict(type='date', tickformat='%d %b'),
                    yaxis=dict(title='Weight (kg)'),
                    hovermode='x unified'
                )
                
                # Add target weight line if it exists
                if 'target_weight' in st.session_state.goals:
                    fig.add_hline(
                        y=st.session_state.goals['target_weight'],
                        line_dash="dash",
                        line_color="green",
                        annotation_text="Target"
                    )
                
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Exercise Distribution
            st.subheader("Exercise Distribution")
            
            exercise_counts = filtered_data['Exercise'].value_counts()
            exercise_counts = exercise_counts[exercise_counts > 0]  # Remove zero counts
            
            fig = px.pie(
                values=exercise_counts.values,
                names=exercise_counts.index,
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            
            fig.update_layout(
                showlegend=True,
                margin=dict(l=0, r=0, t=20, b=0),
                legend=dict(orientation="h", yanchor="bottom", y=-0.3)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Recent Activities
            st.subheader("Recent Activities")
            
            recent = filtered_data.sort_values('Date', ascending=False).head(5)
            
            for _, row in recent.iterrows():
                with st.container():
                    st.markdown(f"""
                    <div style="padding: 10px; border-left: 4px solid {COLOR_PALETTE['primary']}; margin-bottom: 10px; background-color: black;">
                        <strong>{row['Date'].strftime('%a, %b %d')}</strong> - {row['Exercise']}<br/>
                        <small>⏱️ {int(row['Duration(min)'])} mins | 🔥 {int(row['Calories_Burned'])} cal | {row['WorkoutDetails'] if 'WorkoutDetails' in row and pd.notna(row['WorkoutDetails']) else ''}</small>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Additional Insights
        st.subheader("Insights & Trends")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Calories burned by exercise type
            calories_by_exercise = filtered_data.groupby('Exercise')['Calories_Burned'].sum().sort_values(ascending=False)
            
            fig = px.bar(
                x=calories_by_exercise.index,
                y=calories_by_exercise.values,
                labels={'x': 'Exercise Type', 'y': 'Calories Burned'},
                color=calories_by_exercise.values,
                color_continuous_scale='Viridis'
            )
            
            fig.update_layout(
                title='Calories by Exercise Type',
                showlegend=False,
                coloraxis_showscale=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Weekly activity pattern
            if len(filtered_data) >= 7:
                filtered_data['Weekday'] = filtered_data['Date'].dt.day_name()
                weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                
                weekly_pattern = filtered_data.groupby('Weekday')['Duration(min)'].mean().reindex(weekday_order)
                
                fig = px.bar(
                    x=weekly_pattern.index,
                    y=weekly_pattern.values,
                    labels={'x': 'Day of Week', 'y': 'Avg. Duration (min)'},
                    color=weekly_pattern.values,
                    color_continuous_scale='Viridis'
                )
                
                fig.update_layout(
                    title='Weekly Activity Pattern',
                    showlegend=False,
                    coloraxis_showscale=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    # Activity Log Page
    elif app_mode == "Activity Log":
        st.header("Activity Log")
        
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["Log", "Add Workout", "Exercise Analysis"])
        
        # Tab 1: Activity Log
        with tab1:
            # Date filter
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Start Date", 
                                        value=datetime.now() - timedelta(days=30),
                                        max_value=datetime.now())
            with col2:
                end_date = st.date_input("End Date", 
                                       value=datetime.now(),
                                       max_value=datetime.now())
            
            # Filter data by date
            filtered_data = st.session_state.user_data[
                (st.session_state.user_data['Date'] >= pd.to_datetime(start_date)) & 
                (st.session_state.user_data['Date'] <= pd.to_datetime(end_date))
            ].sort_values('Date', ascending=False)
            
            # Exercise type filter
            unique_exercises = ['All'] + sorted(st.session_state.user_data['Exercise'].unique().tolist())
            exercise_filter = st.selectbox("Filter by Exercise Type", unique_exercises)
            
            if exercise_filter != 'All':
                filtered_data = filtered_data[filtered_data['Exercise'] == exercise_filter]
            
            # Display activity log
            if not filtered_data.empty:
                # Summary stats
                active_days = len(filtered_data[filtered_data['Duration(min)'] > 0])
                total_duration = filtered_data['Duration(min)'].sum()
                avg_duration = filtered_data['Duration(min)'].mean()
                total_calories = filtered_data['Calories_Burned'].sum()
                
                st.markdown(f"""
                ### Summary
                - **Period**: {start_date.strftime('%b %d, %Y')} to {end_date.strftime('%b %d, %Y')}
                - **Active Days**: {active_days} days
                - **Total Duration**: {total_duration:.0f} minutes
                - **Average Workout**: {avg_duration:.1f} minutes
                - **Total Calories**: {total_calories:.0f} calories burned
                """)
                
                # Paginated log display
                st.subheader("Workout Log")
                
                items_per_page = 7
                total_pages = (len(filtered_data) + items_per_page - 1) // items_per_page
                
                col1, col2 = st.columns([4, 1])
                with col2:
                    page = st.number_input("Page", min_value=1, max_value=max(1, total_pages), value=1)
                
                start_idx = (page - 1) * items_per_page
                end_idx = min(start_idx + items_per_page, len(filtered_data))
                
                for _, row in filtered_data.iloc[start_idx:end_idx].iterrows():
                    with st.expander(f"{row['Date'].strftime('%a, %b %d, %Y')} - {row['Exercise']} ({int(row['Duration(min)'])} mins)"):
                        col1, col2 = st.columns([3, 2])
                        
                        with col1:
                            st.markdown(f"**Duration:** {int(row['Duration(min)'])} minutes")
                            st.markdown(f"**Calories Burned:** {int(row['Calories_Burned'])}")
                            
                            if 'Distance(km)' in row and row['Distance(km)'] > 0:
                                st.markdown(f"**Distance:** {row['Distance(km)']:.2f} km")
                            
                            if 'Intensity(1-10)' in row and row['Intensity(1-10)'] > 0:
                                st.markdown(f"**Intensity:** {row['Intensity(1-10)']}/10")
                            
                            if 'WorkoutDetails' in row and pd.notna(row['WorkoutDetails']):
                                st.markdown(f"**Details:** {row['WorkoutDetails']}")
                        
                        with col2:
                            if 'Sets' in row and row['Sets'] > 0:
                                st.markdown(f"**Sets:** {int(row['Sets'])}")
                                
                            if 'AvgReps' in row and row['AvgReps'] > 0:
                                st.markdown(f"**Avg. Reps:** {int(row['AvgReps'])}")
                                
                            if 'AvgHR' in row and row['AvgHR'] > 0:
                                st.markdown(f"**Avg. Heart Rate:** {int(row['AvgHR'])} bpm")
                                
                            if 'MaxHR' in row and row['MaxHR'] > 0:
                                st.markdown(f"**Max Heart Rate:** {int(row['MaxHR'])} bpm")
                                
                            if 'RPE(1-10)' in row and row['RPE(1-10)'] > 0:
                                st.markdown(f"**RPE:** {row['RPE(1-10)']}/10")
            else:
                st.info("No workout data available for the selected filters.")
            
            # Export data option
            if not filtered_data.empty:
                export_data = filtered_data.copy()
                export_data['Date'] = export_data['Date'].dt.strftime('%Y-%m-%d')
                
                csv = export_data.to_csv(index=False)
                st.download_button(
                    label="Export Filtered Data",
                    data=csv,
                    file_name=f"fitness_data_{start_date}_to_{end_date}.csv",
                    mime="text/csv",
                )
        
        # Tab 2: Add Workout
        with tab2:
            st.subheader("Add New Workout")
            
            with st.form("workout_form"):
                col1, col2 = st.columns(2)
                
                with col1:
                    workout_date = st.date_input("Date", datetime.now(), max_value=datetime.now())
                    
                    exercise_types = sorted([ex for ex in st.session_state.user_data['Exercise'].unique() if ex != 'Rest'])
                    exercise_type = st.selectbox("Exercise Type", exercise_types)
                    
                    duration = st.number_input("Duration (minutes)", min_value=5, max_value=300, value=45)
                    
                    # Calories calculator based on exercise type and duration
                    calories_map = {
                        "Running": 10, "Cycling": 8, "Swimming": 9, "Weight Training (Upper)": 6, 
                        "Weight Training (Lower)": 6, "Yoga": 4, "HIIT": 12, "Walking": 5
                    }
                    
                    default_calories = int(duration * calories_map.get(exercise_type, 7))
                    calories = st.number_input("Calories Burned", min_value=0, value=default_calories)
                    
                with col2:
                    intensity = st.slider("Intensity (1-10)", min_value=1, max_value=10, value=7)
                    
                    distance = 0.0
                    if exercise_type in ["Running", "Cycling", "Swimming", "Walking"]:
                        distance = st.number_input("Distance (km)", min_value=0.0, max_value=100.0, value=5.0, step=0.1)
                    
                    details = st.text_area("Workout Details", height=100, 
                                        placeholder="Enter any additional workout details...")
                    
                    current_weight = st.number_input("Current Weight (kg)", min_value=40.0, max_value=150.0, 
                                                 value=st.session_state.user_data['Weight(kg)'].iloc[-1] 
                                                 if 'Weight(kg)' in st.session_state.user_data.columns else 70.0,
                                                 step=0.1)
                
                submit_workout = st.form_submit_button("Add Workout")
                
                if submit_workout:
                    # Create new workout entry
                    new_data = {
                        'Date': pd.to_datetime(workout_date),
                        'Exercise': exercise_type,
                        'Duration(min)': duration,
                        'Calories_Burned': calories,
                        'Weight(kg)': current_weight,
                        'Intensity(1-10)': intensity,
                        'Distance(km)': distance,
                        'WorkoutDetails': details
                    }
                    
                    # Add default values for other columns
                    for col in st.session_state.user_data.columns:
                        if col not in new_data:
                            new_data[col] = 0
                    
                    # Add new workout to dataframe
                    st.session_state.user_data = pd.concat([
                        st.session_state.user_data, 
                        pd.DataFrame([new_data])
                    ]).reset_index(drop=True)
                    
                    # Sort by date
                    st.session_state.user_data = st.session_state.user_data.sort_values('Date')
                    
                    # Save to CSV
                    st.session_state.user_data.to_csv('fitness_data_comprehensive.csv', index=False)
                    
                    st.success("Workout added successfully!")
                    st.experimental_rerun()
        
        # Tab 3: Exercise Analysis
        with tab3:
            st.subheader("Exercise Analysis")
            
            # Select exercise type to analyze
            exercise_types = sorted(st.session_state.user_data['Exercise'].unique())
            selected_exercise = st.selectbox("Select Exercise to Analyze", exercise_types)
            
            # Filter data for selected exercise
            exercise_data = st.session_state.user_data[st.session_state.user_data['Exercise'] == selected_exercise]
            
            if not exercise_data.empty:
                # Basic statistics
                total_workouts = len(exercise_data)
                avg_duration = exercise_data['Duration(min)'].mean()
                avg_calories = exercise_data['Calories_Burned'].mean()
                avg_intensity = exercise_data['Intensity(1-10)'].mean() if 'Intensity(1-10)' in exercise_data.columns else 0
                
                st.markdown(f"""
                #### {selected_exercise} Statistics
                - **Total Workouts**: {total_workouts}
                - **Average Duration**: {avg_duration:.1f} minutes
                - **Average Calories**: {avg_calories:.1f}
                - **Average Intensity**: {avg_intensity:.1f}/10
                """)
                
                # Progress over time
                st.subheader("Progress Over Time")
                
                # Create metrics to display
                metrics = ['Duration(min)', 'Calories_Burned']
                
                if 'Distance(km)' in exercise_data.columns:
                    if exercise_data['Distance(km)'].sum() > 0:
                        metrics.append('Distance(km)')
                
                if 'Intensity(1-10)' in exercise_data.columns:
                    metrics.append('Intensity(1-10)')
                
                selected_metric = st.selectbox("Select Metric", metrics)
                
                # Create line chart
                fig = px.line(
                    exercise_data.sort_values('Date'), 
                    x='Date', 
                    y=selected_metric,
                    markers=True,
                    title=f"{selected_metric} Progress for {selected_exercise}"
                )
                
                # Add trend line
                fig.add_traces(
                    px.scatter(
                        exercise_data.sort_values('Date'), 
                        x='Date', 
                        y=selected_metric,
                        trendline="ols"
                    ).data[1]
                )
                
                fig.update_layout(
                    xaxis_title="Date",
                    yaxis_title=selected_metric,
                    legend_title="Legend"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Performance insights
                if total_workouts >= 3:
                    st.subheader("Performance Insights")
                    
                    # Calculate trends
                    first_half = exercise_data.sort_values('Date').iloc[:len(exercise_data)//2]
                    second_half = exercise_data.sort_values('Date').iloc[len(exercise_data)//2:]
                    
                    duration_change = second_half['Duration(min)'].mean() - first_half['Duration(min)'].mean()
                    calories_change = second_half['Calories_Burned'].mean() - first_half['Calories_Burned'].mean()
                    
                    if 'Distance(km)' in exercise_data.columns and exercise_data['Distance(km)'].sum() > 0:
                        distance_change = second_half['Distance(km)'].mean() - first_half['Distance(km)'].mean()
                        
                        if exercise_type in ["Running", "Walking"]:
                            # Calculate pace (min/km)
                            exercise_data['Pace'] = exercise_data['Duration(min)'] / exercise_data['Distance(km)']
                            first_pace = first_half['Duration(min)'].sum() / first_half['Distance(km)'].sum()
                            second_pace = second_half['Duration(min)'].sum() / second_half['Distance(km)'].sum()
                            pace_change = second_pace - first_pace
                            
                            st.markdown(f"""
                            Changes from earlier to recent workouts:
                            - Duration: {duration_change:.1f} minutes {' ⬆️' if duration_change > 0 else ' ⬇️' if duration_change < 0 else ''}
                            - Calories: {calories_change:.1f} calories {' ⬆️' if calories_change > 0 else ' ⬇️' if calories_change < 0 else ''}
                            - Distance: {distance_change:.2f} km {' ⬆️' if distance_change > 0 else ' ⬇️' if distance_change < 0 else ''}
                            - Pace: {abs(pace_change):.2f} min/km {'slower ⬆️' if pace_change > 0 else 'faster ⬇️' if pace_change < 0 else ''}
                            """)
                        else:
                            st.markdown(f"""
                            Changes from earlier to recent workouts:
                            - Duration: {duration_change:.1f} minutes {' ⬆️' if duration_change > 0 else ' ⬇️' if duration_change < 0 else ''}
                            - Calories: {calories_change:.1f} calories {' ⬆️' if calories_change > 0 else ' ⬇️' if calories_change < 0 else ''}
                            - Distance: {distance_change:.2f} km {' ⬆️' if distance_change > 0 else ' ⬇️' if distance_change < 0 else ''}
                            """)
                    else:
                        st.markdown(f"""
                        Changes from earlier to recent workouts:
                        - Duration: {duration_change:.1f} minutes {' ⬆️' if duration_change > 0 else ' ⬇️' if duration_change < 0 else ''}
                        - Calories: {calories_change:.1f} calories {' ⬆️' if calories_change > 0 else ' ⬇️' if calories_change < 0 else ''}
                        """)
            else:
                st.info(f"No data available for {selected_exercise}.")
    
    # Body Metrics Page
    elif app_mode == "Body Metrics":
        st.header("Body Metrics Tracker")
        
        # Check if body measurement columns exist
        if 'Weight(kg)' not in st.session_state.user_data.columns:
            st.warning("Body measurement data is not available in the dataset.")
        else:
            # Create tabs for different body metrics views
            tab1, tab2 = st.tabs(["Weight & Body Composition", "Add Measurements"])
            
            # Tab 1: Weight & Body Composition
            with tab1:
                # Date range filter
                col1, col2 = st.columns(2)
                with col1:
                    start_date = st.date_input("Start Date", 
                                            value=datetime.now() - timedelta(days=30),
                                            max_value=datetime.now(),
                                            key="body_start_date")
                with col2:
                    end_date = st.date_input("End Date", 
                                           value=datetime.now(),
                                           max_value=datetime.now(),
                                           key="body_end_date")
                
                # Filter data by date
                filtered_data = st.session_state.user_data[
                    (st.session_state.user_data['Date'] >= pd.to_datetime(start_date)) & 
                    (st.session_state.user_data['Date'] <= pd.to_datetime(end_date))
                ]
                
                # Weight tracking chart
                st.subheader("Weight Trend")
                
                weight_data = filtered_data[['Date', 'Weight(kg)']].dropna()
                
                if not weight_data.empty:
                    # Calculate weight change metrics
                    current_weight = weight_data['Weight(kg)'].iloc[-1]
                    initial_weight = weight_data['Weight(kg)'].iloc[0]
                    weight_change = current_weight - initial_weight
                    avg_weight = weight_data['Weight(kg)'].mean()
                    min_weight = weight_data['Weight(kg)'].min()
                    max_weight = weight_data['Weight(kg)'].max()
                    
                    # Display weight stats
                    st.markdown(f"""
                    ### Weight Summary
                    - **Current Weight**: {current_weight:.1f} kg
                    - **Initial Weight**: {initial_weight:.1f} kg
                    - **Change**: {weight_change:.1f} kg ({(weight_change/initial_weight)*100:.1f}%)
                    - **Average**: {avg_weight:.1f} kg
                    - **Range**: {min_weight:.1f} kg to {max_weight:.1f} kg
                    """)
                    
                    # Create weight trend chart
                    fig = px.line(
                        weight_data, 
                        x='Date', 
                        y='Weight(kg)',
                        markers=True,
                        labels={'Weight(kg)': 'Weight (kg)', 'Date': ''},
                        title='Weight Trend'
                    )
                    
                    fig.update_layout(
                        xaxis=dict(type='date', tickformat='%d %b'),
                        yaxis=dict(title='Weight (kg)'),
                        hovermode='x unified'
                    )
                    
                    # Add target weight line if it exists
                    if 'target_weight' in st.session_state.goals:
                        fig.add_hline(
                            y=st.session_state.goals['target_weight'],
                            line_dash="dash",
                            line_color="green",
                            annotation_text="Target"
                        )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Body fat chart
                    if 'BodyFat(%)' in filtered_data.columns:
                        st.subheader("Body Composition")
                        
                        bf_data = filtered_data[['Date', 'BodyFat(%)']].dropna()
                        
                        if not bf_data.empty:
                            # Calculate body fat change metrics
                            current_bf = bf_data['BodyFat(%)'].iloc[-1]
                            initial_bf = bf_data['BodyFat(%)'].iloc[0]
                            bf_change = current_bf - initial_bf
                            
                            # Display body fat stats
                            st.markdown(f"""
                            ### Body Fat Summary
                            - **Current Body Fat**: {current_bf:.1f}%
                            - **Initial Body Fat**: {initial_bf:.1f}%
                            - **Change**: {bf_change:.1f}% ({(bf_change/initial_bf)*100:.1f}%)
                            """)
                            
                            # Create body fat trend chart
                            fig = px.line(
                                bf_data, 
                                x='Date', 
                                y='BodyFat(%)',
                                markers=True,
                                labels={'BodyFat(%)': 'Body Fat (%)', 'Date': ''},
                                title='Body Fat Trend'
                            )
                            
                            fig.update_layout(
                                xaxis=dict(type='date', tickformat='%d %b'),
                                yaxis=dict(title='Body Fat (%)'),
                                hovermode='x unified'
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # Weight vs. Exercise correlation
                    st.subheader("Weight vs. Exercise")
                    
                    # Merge weight data with exercise data
                    weekly_exercise = filtered_data.groupby(pd.Grouper(key='Date', freq='W')).agg({
                        'Duration(min)': 'sum',
                        'Calories_Burned': 'sum'
                    }).reset_index()
                    
                    weekly_weight = filtered_data.groupby(pd.Grouper(key='Date', freq='W')).agg({
                        'Weight(kg)': 'mean'
                    }).reset_index()
                    
                    weekly_data = pd.merge(weekly_exercise, weekly_weight, on='Date')
                    
                    if len(weekly_data) > 1:
                        fig = go.Figure()
                        
                        # Add weight line
                        fig.add_trace(go.Scatter(
                            x=weekly_data['Date'],
                            y=weekly_data['Weight(kg)'],
                            name='Weight (kg)',
                            line=dict(color='royalblue', width=3)
                        ))
                        
                        # Add exercise duration bars
                        fig.add_trace(go.Bar(
                            x=weekly_data['Date'],
                            y=weekly_data['Duration(min)'],
                            name='Exercise (min)',
                            marker_color='lightgreen',
                            opacity=0.7,
                            yaxis='y2'
                        ))
                        
                        # Set up layout with dual Y axes
                        fig.update_layout(
                            title='Weekly Weight vs. Exercise Duration',
                            xaxis=dict(title='Week'),
                            yaxis=dict(title='Weight (kg)', side='left', showgrid=False),
                            yaxis2=dict(
                                title='Exercise (min)',
                                side='right',
                                overlaying='y',
                                showgrid=False
                            ),
                            legend=dict(x=0, y=1.1, orientation='h'),
                            hovermode='x unified'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No weight data available for the selected date range.")
            
            # Tab 2: Add Measurements
            with tab2:
                st.subheader("Add Body Measurements")
                
                with st.form("body_form"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        meas_date = st.date_input("Date", datetime.now(), max_value=datetime.now())
                        weight = st.number_input("Weight (kg)", min_value=40.0, max_value=150.0, 
                                              value=st.session_state.user_data['Weight(kg)'].iloc[-1] 
                                              if 'Weight(kg)' in st.session_state.user_data.columns else 70.0,
                                              step=0.1)
                    
                    with col2:
                        body_fat = st.number_input("Body Fat (%)", min_value=3.0, max_value=50.0, 
                                                value=st.session_state.user_data['BodyFat(%)'].iloc[-1] 
                                                if 'BodyFat(%)' in st.session_state.user_data.columns else 20.0,
                                                step=0.1)
                        resting_hr = st.number_input("Resting Heart Rate (bpm)", min_value=30, max_value=120, 
                                                   value=st.session_state.user_data['RestingHR'].iloc[-1] 
                                                   if 'RestingHR' in st.session_state.user_data.columns else 65)
                    
                    notes = st.text_area("Notes", placeholder="Any additional notes about your measurements...")
                    
                    submit_measurements = st.form_submit_button("Add Measurements")
                    
                    if submit_measurements:
                        # Find the row for the specified date if it exists
                        date_mask = st.session_state.user_data['Date'] == pd.to_datetime(meas_date)
                        
                        if date_mask.any():
                            # Update existing row
                            st.session_state.user_data.loc[date_mask, 'Weight(kg)'] = weight
                            
                            if 'BodyFat(%)' in st.session_state.user_data.columns:
                                st.session_state.user_data.loc[date_mask, 'BodyFat(%)'] = body_fat
                            
                            if 'RestingHR' in st.session_state.user_data.columns:
                                st.session_state.user_data.loc[date_mask, 'RestingHR'] = resting_hr
                        else:
                            # Create new row with minimal required data
                            new_data = {
                                'Date': pd.to_datetime(meas_date),
                                'Exercise': 'Rest',  # Default value
                                'Duration(min)': 0,
                                'Calories_Burned': 0,
                                'Weight(kg)': weight
                            }
                            
                            # Add body fat and resting hr if in the dataset
                            if 'BodyFat(%)' in st.session_state.user_data.columns:
                                new_data['BodyFat(%)'] = body_fat
                                
                            if 'RestingHR' in st.session_state.user_data.columns:
                                new_data['RestingHR'] = resting_hr
                            
                            # Add default values for other columns
                            for col in st.session_state.user_data.columns:
                                if col not in new_data:
                                    new_data[col] = 0
                            
                            # Add new row
                            st.session_state.user_data = pd.concat([
                                st.session_state.user_data, 
                                pd.DataFrame([new_data])
                            ]).reset_index(drop=True)
                        
                        # Sort by date and save
                        st.session_state.user_data = st.session_state.user_data.sort_values('Date')
                        st.session_state.user_data.to_csv('fitness_data_comprehensive.csv', index=False)
                        
                        st.success("Measurements added successfully!")
                        st.experimental_rerun()
    
    # Nutrition Page
    elif app_mode == "Nutrition":
        st.header("Nutrition Tracker")
        
        # Check if nutrition columns exist
        has_nutrition = all(col in st.session_state.user_data.columns for col in ['CaloriesConsumed', 'Protein(g)', 'Carbs(g)', 'Fat(g)'])
        
        if not has_nutrition:
            st.warning("Nutrition data is not available in the dataset.")
        else:
            # Create tabs for different nutrition views
            tab1, tab2 = st.tabs(["Nutrition Summary", "Calorie Balance"])
            
            # Tab 1: Nutrition Summary
            with tab1:
                # Date range filter
                col1, col2 = st.columns(2)
                with col1:
                    start_date = st.date_input("Start Date", 
                                            value=datetime.now() - timedelta(days=7),
                                            max_value=datetime.now(),
                                            key="nutrition_start_date")
                with col2:
                    end_date = st.date_input("End Date", 
                                           value=datetime.now(),
                                           max_value=datetime.now(),
                                           key="nutrition_end_date")
                
                # Filter data by date
                filtered_data = st.session_state.user_data[
                    (st.session_state.user_data['Date'] >= pd.to_datetime(start_date)) & 
                    (st.session_state.user_data['Date'] <= pd.to_datetime(end_date))
                ]
                
                # Nutrition summary
                st.subheader("Nutrition Summary")
                
                # Calculate averages
                avg_calories = filtered_data['CaloriesConsumed'].mean()
                avg_protein = filtered_data['Protein(g)'].mean()
                avg_carbs = filtered_data['Carbs(g)'].mean()
                avg_fat = filtered_data['Fat(g)'].mean()
                
                # Calculate macronutrient percentages
                total_macro_calories = avg_protein * 4 + avg_carbs * 4 + avg_fat * 9
                protein_pct = (avg_protein * 4 / total_macro_calories) * 100 if total_macro_calories > 0 else 0
                carbs_pct = (avg_carbs * 4 / total_macro_calories) * 100 if total_macro_calories > 0 else 0
                fat_pct = (avg_fat * 9 / total_macro_calories) * 100 if total_macro_calories > 0 else 0
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Avg. Calories", f"{avg_calories:.0f}")
                col2.metric("Avg. Protein", f"{avg_protein:.0f}g")
                col3.metric("Avg. Carbs", f"{avg_carbs:.0f}g")
                col4.metric("Avg. Fat", f"{avg_fat:.0f}g")
                
                # Macronutrient breakdown
                st.subheader("Macronutrient Breakdown")
                
                # Create pie chart for macronutrient breakdown
                macro_labels = ['Protein', 'Carbohydrates', 'Fat']
                macro_values = [protein_pct, carbs_pct, fat_pct]
                macro_colors = ['#ff9999', '#66b3ff', '#ffcc99']
                
                fig = px.pie(
                    values=macro_values,
                    names=macro_labels,
                    color=macro_labels,
                    color_discrete_map=dict(zip(macro_labels, macro_colors)),
                    title='Average Macronutrient Distribution',
                    hole=0.4
                )
                
                fig.update_layout(
                    annotations=[dict(text=f"{protein_pct:.0f}% / {carbs_pct:.0f}% / {fat_pct:.0f}%", x=0.5, y=0.5, font_size=12, showarrow=False)]
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Daily nutrition trends
                st.subheader("Daily Nutrition Trends")
                
                # Create line chart for calorie intake
                fig = px.line(
                    filtered_data, 
                    x='Date', 
                    y=['CaloriesConsumed', 'Calories_Burned'],
                    labels={'value': 'Calories', 'variable': 'Type'},
                    title='Daily Calorie Intake vs. Burned',
                    color_discrete_map={
                        'CaloriesConsumed': '#ff9999',
                        'Calories_Burned': '#66b3ff'
                    }
                )
                
                fig.update_layout(
                    xaxis=dict(type='date', tickformat='%a %d %b'),
                    yaxis=dict(title='Calories'),
                    hovermode='x unified',
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Macro trends
                macro_data = filtered_data[['Date', 'Protein(g)', 'Carbs(g)', 'Fat(g)']].copy()
                
                fig = px.bar(
                    macro_data,
                    x='Date',
                    y=['Protein(g)', 'Carbs(g)', 'Fat(g)'],
                    labels={'value': 'Grams', 'variable': 'Nutrient'},
                    title='Daily Macronutrient Intake',
                    color_discrete_map={
                        'Protein(g)': '#ff9999',
                        'Carbs(g)': '#66b3ff',
                        'Fat(g)': '#ffcc99'
                    }
                )
                
                fig.update_layout(
                    xaxis=dict(type='date', tickformat='%a %d %b'),
                    yaxis=dict(title='Grams'),
                    barmode='stack',
                    hovermode='x unified',
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Water intake if available
                if 'Water(ml)' in filtered_data.columns:
                    st.subheader("Hydration")
                    
                    avg_water = filtered_data['Water(ml)'].mean()
                    st.metric("Average Water Intake", f"{avg_water:.0f} ml", f"{avg_water/1000:.1f} L")
                    
                    fig = px.bar(
                        filtered_data,
                        x='Date',
                        y='Water(ml)',
                        title='Daily Water Intake',
                        color='Water(ml)',
                        color_continuous_scale='blues'
                    )
                    
                    fig.update_layout(
                        xaxis=dict(type='date', tickformat='%a %d %b'),
                        yaxis=dict(title='Water (ml)'),
                        hovermode='x unified',
                        coloraxis_showscale=False
                    )
                    
                    # Add target line at 2000ml
                    fig.add_hline(y=2000, line_dash="dash", line_color="blue", 
                                annotation_text="Recommended (2L)")
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            # Tab 2: Calorie Balance
            with tab2:
                st.subheader("Calorie Balance")
                
                # Date range same as tab 1
                filtered_data = st.session_state.user_data[
                    (st.session_state.user_data['Date'] >= pd.to_datetime(start_date)) & 
                    (st.session_state.user_data['Date'] <= pd.to_datetime(end_date))
                ]
                
                # Calculate calorie deficit/surplus
                filtered_data['CalorieBalance'] = filtered_data['CaloriesConsumed'] - filtered_data['Calories_Burned']
                filtered_data['CumulativeBalance'] = filtered_data['CalorieBalance'].cumsum()
                
                # Calculate overall balance
                total_consumed = filtered_data['CaloriesConsumed'].sum()
                total_burned = filtered_data['Calories_Burned'].sum()
                total_balance = total_consumed - total_burned
                
                # Display summary metrics
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Calories Consumed", f"{total_consumed:.0f}")
                col2.metric("Total Calories Burned", f"{total_burned:.0f}")
                col3.metric("Overall Balance", 
                          f"{total_balance:.0f}", 
                          f"{total_balance/len(filtered_data):.0f} per day")
                
                # Create daily balance chart
                fig = px.bar(
                    filtered_data,
                    x='Date',
                    y='CalorieBalance',
                    title='Daily Calorie Balance (Consumed - Burned)',
                    color='CalorieBalance',
                    color_continuous_scale=['red', 'white', 'green'],
                    color_continuous_midpoint=0
                )
                
                fig.update_layout(
                    xaxis=dict(type='date', tickformat='%a %d %b'),
                    yaxis=dict(title='Calorie Balance'),
                    hovermode='x unified'
                )
                
                fig.add_hline(y=0, line_dash="solid", line_color="black")
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Create cumulative balance chart
                fig = px.line(
                    filtered_data,
                    x='Date',
                    y='CumulativeBalance',
                    title='Cumulative Calorie Balance',
                    markers=True
                )
                
                fig.update_layout(
                    xaxis=dict(type='date', tickformat='%a %d %b'),
                    yaxis=dict(title='Cumulative Balance (calories)'),
                    hovermode='x unified'
                )
                
                fig.add_hline(y=0, line_dash="solid", line_color="black")
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Estimated weight impact
                st.subheader("Estimated Weight Impact")
                
                # Assuming 7700 calorie deficit/surplus = 1kg weight loss/gain
                estimated_weight_change = total_balance / 7700
                
                st.info(f"""
                ### Weight Change Estimation
                - Total Calorie Balance: {total_balance:.0f} calories {"surplus" if total_balance > 0 else "deficit"}
                - Estimated Weight Impact: {abs(estimated_weight_change):.2f} kg {"gain" if total_balance > 0 else "loss"}
                - Based on the approximation that 7,700 calories = 1 kg of body weight
                """)
                
                # Compare with actual weight change if available
                if 'Weight(kg)' in filtered_data.columns and len(filtered_data) > 1:
                    first_weight = filtered_data['Weight(kg)'].iloc[0]
                    last_weight = filtered_data['Weight(kg)'].iloc[-1]
                    actual_change = last_weight - first_weight
                    
                    st.markdown(f"""
                    ### Actual vs. Estimated Weight Change
                    - Starting weight: {first_weight:.1f} kg
                    - Current weight: {last_weight:.1f} kg
                    - Actual change: {actual_change:.2f} kg
                    - Estimated change: {estimated_weight_change:.2f} kg
                    - Difference: {actual_change - estimated_weight_change:.2f} kg
                    """)
    
    # Sleep & Recovery Page
    elif app_mode == "Sleep & Recovery":
        st.header("Sleep & Recovery")
        
        # Check if sleep columns exist
        has_sleep = all(col in st.session_state.user_data.columns for col in ['SleepHours', 'SleepQuality(%)'])
        
        if not has_sleep:
            st.warning("Sleep data is not available in the dataset.")
        else:
            # Date range filter
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Start Date", 
                                        value=datetime.now() - timedelta(days=14),
                                        max_value=datetime.now(),
                                        key="sleep_start_date")
            with col2:
                end_date = st.date_input("End Date", 
                                       value=datetime.now(),
                                       max_value=datetime.now(),
                                       key="sleep_end_date")
            
            # Filter data by date
            filtered_data = st.session_state.user_data[
                (st.session_state.user_data['Date'] >= pd.to_datetime(start_date)) & 
                (st.session_state.user_data['Date'] <= pd.to_datetime(end_date))
            ]
            
            # Sleep summary
            st.subheader("Sleep Summary")
            
            avg_sleep = filtered_data['SleepHours'].mean()
            avg_quality = filtered_data['SleepQuality(%)'].mean()
            sleep_under_7 = (filtered_data['SleepHours'] < 7).sum()
            sleep_over_8 = (filtered_data['SleepHours'] > 8).sum()
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Avg. Sleep Duration", f"{avg_sleep:.1f} hours")
            col2.metric("Avg. Sleep Quality", f"{avg_quality:.0f}%")
            col3.metric("Sleep Score", f"{(avg_sleep/8 * 50 + avg_quality/100 * 50):.0f}/100")
            
            st.markdown(f"""
            #### Sleep Statistics
            - Days with less than 7 hours: {sleep_under_7} ({sleep_under_7/len(filtered_data)*100:.0f}%)
            - Days with more than 8 hours: {sleep_over_8} ({sleep_over_8/len(filtered_data)*100:.0f}%)
            """)
            
            # Sleep duration chart
            st.subheader("Sleep Duration & Quality")
            
            # Create a figure with dual y-axes
            fig = go.Figure()
            
            # Add sleep duration bars
            fig.add_trace(go.Bar(
                x=filtered_data['Date'],
                y=filtered_data['SleepHours'],
                name='Sleep Hours',
                marker_color='darkblue'
            ))
            
            # Add sleep quality line
            fig.add_trace(go.Scatter(
                x=filtered_data['Date'],
                y=filtered_data['SleepQuality(%)'],
                name='Sleep Quality',
                line=dict(color='orange', width=3),
                yaxis='y2'
            ))
            
            # Set up layout with dual y-axes
            fig.update_layout(
                title='Sleep Duration & Quality',
                xaxis=dict(title='Date', tickformat='%a %d %b'),
                yaxis=dict(title='Sleep Hours', side='left', range=[0, 12]),
                yaxis2=dict(
                    title='Sleep Quality (%)',
                    side='right',
                    range=[0, 100],
                    overlaying='y',
                    tickmode='array',
                    tickvals=[0, 25, 50, 75, 100]
                ),
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                hovermode='x unified'
            )
            
            # Add reference lines
            fig.add_hline(y=7, line_dash="dash", line_color="rgba(0, 0, 255, 0.3)", 
                        annotation_text="Recommended Min.")
            fig.add_hline(y=9, line_dash="dash", line_color="rgba(0, 0, 255, 0.3)", 
                        annotation_text="Recommended Max.")
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Weekly sleep pattern
            st.subheader("Weekly Sleep Pattern")
            
            # Add weekday column
            filtered_data['Weekday'] = filtered_data['Date'].dt.day_name()
            
            # Calculate average sleep by day of week
            weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            sleep_by_day = filtered_data.groupby('Weekday').agg({
                'SleepHours': 'mean',
                'SleepQuality(%)': 'mean'
            }).reindex(weekday_order)
            
            # Create a bar chart for sleep by day of week
            fig = go.Figure()
            
            # Add sleep duration bars
            fig.add_trace(go.Bar(
                x=sleep_by_day.index,
                y=sleep_by_day['SleepHours'],
                name='Sleep Hours',
                marker_color='darkblue'
            ))
            
            # Add sleep quality line
            fig.add_trace(go.Scatter(
                x=sleep_by_day.index,
                y=sleep_by_day['SleepQuality(%)'],
                name='Sleep Quality',
                line=dict(color='orange', width=3),
                yaxis='y2'
            ))
            
            # Set up layout with dual y-axes
            fig.update_layout(
                title='Average Sleep by Day of Week',
                xaxis=dict(title=''),
                yaxis=dict(title='Sleep Hours', side='left', range=[0, 12]),
                yaxis2=dict(
                    title='Sleep Quality (%)',
                    side='right',
                    range=[0, 100],
                    overlaying='y',
                    tickmode='array',
                    tickvals=[0, 25, 50, 75, 100]
                ),
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                hovermode='closest'
            )
            
            # Add reference lines
            fig.add_hline(y=7, line_dash="dash", line_color="rgba(0, 0, 255, 0.3)")
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Recovery metrics
            if 'RestingHR' in filtered_data.columns:
                st.subheader("Recovery Metrics")
                
                # Create resting heart rate chart
                fig = px.line(
                    filtered_data,
                    x='Date',
                    y='RestingHR',
                    title='Resting Heart Rate',
                    markers=True
                )
                
                fig.update_layout(
                    xaxis=dict(title='Date', tickformat='%a %d %b'),
                    yaxis=dict(title='Resting Heart Rate (bpm)'),
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Sleep and exercise correlation
                st.subheader("Sleep and Exercise Correlation")
                
                # Create dataset with previous day's exercise intensity and sleep quality
                sleep_exercise_data = filtered_data[['Date', 'SleepHours', 'SleepQuality(%)', 'Exercise', 'Duration(min)', 'Intensity(1-10)']].copy()
                
                # Shift exercise data to correlate with next night's sleep
                sleep_exercise_data['PrevExercise'] = sleep_exercise_data['Exercise'].shift(1)
                sleep_exercise_data['PrevDuration'] = sleep_exercise_data['Duration(min)'].shift(1)
                sleep_exercise_data['PrevIntensity'] = sleep_exercise_data['Intensity(1-10)'].shift(1)
                
                # Remove first row with NaN values
                sleep_exercise_data = sleep_exercise_data.iloc[1:]
                
                # Create scatter plot of exercise intensity vs sleep quality
                if 'Intensity(1-10)' in sleep_exercise_data.columns and len(sleep_exercise_data) > 5:
                    fig = px.scatter(
                        sleep_exercise_data,
                        x='PrevIntensity',
                        y='SleepQuality(%)',
                        size='PrevDuration',
                        color='PrevExercise',
                        hover_name='Date',
                        title="Previous Day's Exercise vs. Sleep Quality",
                        labels={
                            'PrevIntensity': 'Previous Day Exercise Intensity (1-10)',
                            'SleepQuality(%)': 'Sleep Quality (%)',
                            'PrevDuration': 'Exercise Duration (min)',
                            'PrevExercise': 'Exercise Type'
                        }
                    )
                    
                    fig.update_layout(
                        hovermode='closest'
                    )
                    
                    # Add trend line
                    fig.add_traces(
                        px.scatter(
                            sleep_exercise_data, 
                            x='PrevIntensity', 
                            y='SleepQuality(%)',
                            trendline="ols"
                        ).data[1]
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Calculate correlation
                    correlation = sleep_exercise_data['PrevIntensity'].corr(sleep_exercise_data['SleepQuality(%)'])
                    
                    if abs(correlation) < 0.2:
                        correlation_text = "very weak"
                    elif abs(correlation) < 0.4:
                        correlation_text = "weak"
                    elif abs(correlation) < 0.6:
                        correlation_text = "moderate"
                    elif abs(correlation) < 0.8:
                        correlation_text = "strong"
                    else:
                        correlation_text = "very strong"
                    
                    st.markdown(f"""
                    There is a {correlation_text} {'positive' if correlation > 0 else 'negative'} correlation 
                    ({correlation:.2f}) between exercise intensity and sleep quality in this data.
                    """)
    
    # Analysis Page
    elif app_mode == "Analysis":
        st.header("Fitness Analysis")
        
        # Create tabs for different analysis views
        tab1, tab2, tab3 = st.tabs(["Performance Trends", "Correlations", "Weekly Patterns"])
        
        # Tab 1: Performance Trends
        with tab1:
            st.subheader("Performance Trends")
            
            # Date range filter
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Start Date", 
                                        value=datetime.now() - timedelta(days=60),
                                        max_value=datetime.now(),
                                        key="analysis_start_date")
            with col2:
                end_date = st.date_input("End Date", 
                                       value=datetime.now(),
                                       max_value=datetime.now(),
                                       key="analysis_end_date")
            
            # Filter data by date
            filtered_data = st.session_state.user_data[
                (st.session_state.user_data['Date'] >= pd.to_datetime(start_date)) & 
                (st.session_state.user_data['Date'] <= pd.to_datetime(end_date))
            ]
            
            # Calculate weekly data
            weekly_data = filtered_data.groupby(pd.Grouper(key='Date', freq='W-MON')).agg({
                'Duration(min)': 'sum',
                'Calories_Burned': 'sum',
                'Distance(km)': 'sum',
                'Steps': 'sum',
                'Exercise': lambda x: len([ex for ex in x if ex != 'Rest'])
            }).reset_index()
            
            weekly_data.rename(columns={'Exercise': 'WorkoutCount'}, inplace=True)
            
            # Select trend to visualize
            trend_options = ['Duration(min)', 'Calories_Burned', 'Distance(km)', 'WorkoutCount', 'Steps']
            trend_names = ['Weekly Duration (minutes)', 'Weekly Calories Burned', 'Weekly Distance (km)', 
                          'Weekly Workout Count', 'Weekly Steps']
            
            trend_to_show = st.selectbox("Select Trend to Visualize", 
                                       options=list(zip(trend_options, trend_names)), 
                                       format_func=lambda x: x[1])
            
            # Create trend chart
            fig = px.line(
                weekly_data,
                x='Date',
                y=trend_to_show[0],
                title=f"Weekly Trend: {trend_to_show[1]}",
                markers=True
            )
            
            fig.update_layout(
                xaxis=dict(title='Week Starting'),
                yaxis=dict(title=trend_to_show[1]),
                hovermode='x unified'
            )
            
            # Add trend line
            fig.add_traces(
                px.scatter(
                    weekly_data, 
                    x='Date', 
                    y=trend_to_show[0],
                    trendline="ols"
                ).data[1]
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate and display trend statistics
            if len(weekly_data) >= 2:
                first_half = weekly_data.iloc[:len(weekly_data)//2]
                second_half = weekly_data.iloc[len(weekly_data)//2:]
                
                first_half_avg = first_half[trend_to_show[0]].mean()
                second_half_avg = second_half[trend_to_show[0]].mean()
                
                change = second_half_avg - first_half_avg
                change_pct = (change / first_half_avg) * 100 if first_half_avg > 0 else 0
                
                st.markdown(f"""
                ### Trend Analysis: {trend_to_show[1]}
                
                - First half average: {first_half_avg:.1f}
                - Second half average: {second_half_avg:.1f}
                - Change: {change:.1f} ({change_pct:.1f}%)
                """)
                
                # Interpret the trend
                if change_pct > 10:
                    st.success(f"Your {trend_to_show[1].lower()} is showing a significant upward trend! 📈")
                elif change_pct > 5:
                    st.success(f"Your {trend_to_show[1].lower()} is showing a positive improvement. 👍")
                elif change_pct < -10:
                    st.warning(f"Your {trend_to_show[1].lower()} is showing a significant decrease. 📉")
                elif change_pct < -5:
                    st.warning(f"Your {trend_to_show[1].lower()} is slightly declining. 👎")
                else:
                    st.info(f"Your {trend_to_show[1].lower()} is relatively stable. ➡️")
            
            # Exercise-specific trends
            st.subheader("Exercise-Specific Trends")
            
            # Get unique exercises excluding 'Rest'
            exercise_options = sorted([ex for ex in filtered_data['Exercise'].unique() if ex != 'Rest'])
            
            if exercise_options:
                selected_exercise = st.selectbox("Select Exercise", exercise_options)
                
                # Filter for selected exercise
                exercise_data = filtered_data[filtered_data['Exercise'] == selected_exercise].copy()
                
                if len(exercise_data) >= 3:
                    # Select metric to analyze
                    exercise_metrics = ['Duration(min)', 'Calories_Burned', 'Intensity(1-10)', 'Distance(km)']
                    exercise_metric_names = ['Duration (minutes)', 'Calories Burned', 'Intensity', 'Distance (km)']
                    
                    valid_metrics = [m for m in exercise_metrics if m in exercise_data.columns]
                    valid_names = [n for m, n in zip(exercise_metrics, exercise_metric_names) if m in exercise_data.columns]
                    
                    selected_metric = st.selectbox("Select Metric", 
                                                options=list(zip(valid_metrics, valid_names)),
                                                format_func=lambda x: x[1])
                    
                    # Create trend chart for specific exercise
                    fig = px.scatter(
                        exercise_data,
                        x='Date',
                        y=selected_metric[0],
                        title=f"{selected_exercise}: {selected_metric[1]} Trend",
                        trendline="ols",
                        trendline_color_override="red"
                    )
                    
                    fig.update_layout(
                        xaxis=dict(title='Date'),
                        yaxis=dict(title=selected_metric[1]),
                        hovermode='closest'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Calculate improvement
                    if len(exercise_data) >= 4:
                        first_workouts = exercise_data.sort_values('Date').iloc[:2][selected_metric[0]].mean()
                        last_workouts = exercise_data.sort_values('Date').iloc[-2:][selected_metric[0]].mean()
                        
                        change = last_workouts - first_workouts
                        change_pct = (change / first_workouts) * 100 if first_workouts > 0 else 0
                        
                        # Different interpretation for different metrics
                        if 'Duration' in selected_metric[1] or 'Calories' in selected_metric[1] or 'Distance' in selected_metric[1]:
                            # For these metrics, higher is better
                            if change_pct > 10:
                                st.success(f"Great improvement in {selected_exercise} {selected_metric[1].lower()}! 🎉")
                            elif change_pct > 5:
                                st.success(f"You've shown improvement in {selected_exercise} {selected_metric[1].lower()}. 👍")
                            elif change_pct < -10:
                                st.warning(f"Your {selected_exercise} {selected_metric[1].lower()} has decreased significantly. 📉")
                            elif change_pct < -5:
                                st.warning(f"Your {selected_exercise} {selected_metric[1].lower()} has slightly decreased. 👎")
                            else:
                                st.info(f"Your {selected_exercise} {selected_metric[1].lower()} has remained stable. ➡️")
                        elif 'pace' in selected_metric[1].lower():
                            # For pace, lower is better
                            if change_pct < -10:
                                st.success(f"Great improvement in {selected_exercise} pace! 🎉")
                            elif change_pct < -5:
                                st.success(f"You've shown improvement in {selected_exercise} pace. 👍")
                            elif change_pct > 10:
                                st.warning(f"Your {selected_exercise} pace has slowed significantly. 📉")
                            elif change_pct > 5:
                                st.warning(f"Your {selected_exercise} pace has slightly slowed. 👎")
                            else:
                                st.info(f"Your {selected_exercise} pace has remained stable. ➡️")
                        else:
                            # For other metrics (like intensity), higher could be better but depends on context
                            st.info(f"Change in {selected_exercise} {selected_metric[1].lower()}: {change:.1f} ({change_pct:.1f}%)")
                else:
                    st.info(f"Not enough data for {selected_exercise} trend analysis. At least 3 workouts needed.")
            else:
                st.info("No exercise data available for trend analysis.")
        
        # Tab 2: Correlations
        with tab2:
            st.subheader("Correlation Analysis")
            
            # Use the same filtered data as tab 1
            
            if len(filtered_data) >= 5:
                # Create list of numeric columns to analyze
                numeric_cols = filtered_data.select_dtypes(include=['number']).columns.tolist()
                
                # Remove date and non-meaningful columns
                exclude_cols = ['Date', 'Sets', 'AvgReps']
                numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
                
                # Create more user-friendly names
                col_names = {
                    'Duration(min)': 'Workout Duration',
                    'Calories_Burned': 'Calories Burned',
                    'Weight(kg)': 'Weight',
                    'Steps': 'Daily Steps',
                    'Distance(km)': 'Distance',
                    'Intensity(1-10)': 'Workout Intensity',
                    'RPE(1-10)': 'Perceived Exertion',
                    'AvgHR': 'Avg Heart Rate',
                    'MaxHR': 'Max Heart Rate',
                    'RestingHR': 'Resting Heart Rate',
                    'SleepHours': 'Sleep Duration',
                    'SleepQuality(%)': 'Sleep Quality',
                    'BodyFat(%)': 'Body Fat %',
                    'CaloriesConsumed': 'Calories Consumed',
                    'Protein(g)': 'Protein Intake',
                    'Carbs(g)': 'Carb Intake',
                    'Fat(g)': 'Fat Intake',
                }
                
                # Select variables to correlate
                col1, col2 = st.columns(2)
                
                with col1:
                    var1 = st.selectbox(
                        "Select First Variable", 
                        options=numeric_cols,
                        format_func=lambda x: col_names.get(x, x),
                        index=0
                    )
                
                with col2:
                    remaining_cols = [col for col in numeric_cols if col != var1]
                    var2 = st.selectbox(
                        "Select Second Variable", 
                        options=remaining_cols,
                        format_func=lambda x: col_names.get(x, x),
                        index=0
                    )
                
                # Create scatter plot
                if var1 in filtered_data.columns and var2 in filtered_data.columns:
                    correlation_data = filtered_data[[var1, var2]].dropna()
                    
                    if len(correlation_data) >= 5:
                        fig = px.scatter(
                            correlation_data,
                            x=var1,
                            y=var2,
                            title=f"Correlation: {col_names.get(var1, var1)} vs {col_names.get(var2, var2)}",
                            trendline="ols",
                            labels={
                                var1: col_names.get(var1, var1),
                                var2: col_names.get(var2, var2)
                            }
                        )
                        
                        fig.update_layout(
                            hovermode='closest'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Calculate and display correlation
                        corr = correlation_data[var1].corr(correlation_data[var2])
                        
                        if abs(corr) < 0.2:
                            corr_strength = "very weak"
                        elif abs(corr) < 0.4:
                            corr_strength = "weak"
                        elif abs(corr) < 0.6:
                            corr_strength = "moderate"
                        elif abs(corr) < 0.8:
                            corr_strength = "strong"
                        else:
                            corr_strength = "very strong"
                        
                        st.markdown(f"""
                        ### Correlation Analysis
                        
                        The correlation between {col_names.get(var1, var1)} and {col_names.get(var2, var2)} is **{corr:.2f}**.
                        
                        This indicates a {corr_strength} {'positive' if corr > 0 else 'negative'} relationship.
                        """)
                        
                        # Add interpretation
                        st.subheader("Interpretation")
                        
                        if corr > 0.6:
                            st.success(f"There's a strong positive relationship - as {col_names.get(var1, var1)} increases, {col_names.get(var2, var2)} tends to increase significantly.")
                        elif corr > 0.3:
                            st.info(f"There's a moderate positive relationship - as {col_names.get(var1, var1)} increases, {col_names.get(var2, var2)} tends to increase somewhat.")
                        elif corr > 0:
                            st.info(f"There's a weak positive relationship - as {col_names.get(var1, var1)} increases, {col_names.get(var2, var2)} tends to increase slightly.")
                        elif corr > -0.3:
                            st.info(f"There's a weak negative relationship - as {col_names.get(var1, var1)} increases, {col_names.get(var2, var2)} tends to decrease slightly.")
                        elif corr > -0.6:
                            st.info(f"There's a moderate negative relationship - as {col_names.get(var1, var1)} increases, {col_names.get(var2, var2)} tends to decrease somewhat.")
                        else:
                            st.success(f"There's a strong negative relationship - as {col_names.get(var1, var1)} increases, {col_names.get(var2, var2)} tends to decrease significantly.")
                    else:
                        st.warning("Not enough data points for correlation analysis after removing missing values.")
                else:
                    st.warning("One or both selected variables are not available in the dataset.")
                
                # Correlation heatmap
                st.subheader("Correlation Heatmap")
                
                # Select columns to include in the heatmap
                heatmap_cols = st.multiselect(
                    "Select Variables for Heatmap",
                    options=numeric_cols,
                    default=numeric_cols[:5],
                    format_func=lambda x: col_names.get(x, x)
                )
                
                if heatmap_cols:
                    # Calculate correlation matrix
                    corr_data = filtered_data[heatmap_cols].dropna()
                    
                    if len(corr_data) > 0:
                        corr_matrix = corr_data.corr()
                        
                        # Rename columns and index for better display
                        corr_matrix = corr_matrix.rename(columns=col_names, index=col_names)
                        
                        # Create heatmap
                        fig = px.imshow(
                            corr_matrix,
                            text_auto='.2f',
                            color_continuous_scale='RdBu_r',
                            origin='lower',
                            aspect='auto',
                            title='Correlation Heatmap'
                        )
                        
                        fig.update_layout(
                            height=500,
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Not enough complete data points for correlation heatmap.")
                else:
                    st.info("Please select variables to include in the heatmap.")
            else:
                st.warning("Not enough data for correlation analysis. At least 5 data points needed.")
        
        # Tab 3: Weekly Patterns
        with tab3:
            st.subheader("Weekly Patterns")
            
            # Use the same filtered data as other tabs
            
            if len(filtered_data) >= 7:  # At least a week of data
                # Add weekday column
                filtered_data['Weekday'] = filtered_data['Date'].dt.day_name()
                filtered_data['WeekdayNum'] = filtered_data['Date'].dt.weekday
                
                # Weekday order
                weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                
                # Group by weekday
                weekly_patterns = filtered_data.groupby('Weekday').agg({
                    'Duration(min)': 'mean',
                    'Calories_Burned': 'mean',
                    'Steps': 'mean'
                }).reindex(weekday_order)
                
                # Count active vs. rest days by weekday
                exercise_counts = filtered_data.groupby(['Weekday', filtered_data['Exercise'] != 'Rest']).size().unstack()
                exercise_counts.columns = ['Rest', 'Active']
                exercise_counts = exercise_counts.reindex(weekday_order).fillna(0)
                
                # Calculate percentages
                exercise_counts['Total'] = exercise_counts.sum(axis=1)
                exercise_counts['ActivePct'] = (exercise_counts['Active'] / exercise_counts['Total'] * 100).round(1)
                
                # Select metric to visualize
                pattern_options = ['Duration(min)', 'Calories_Burned', 'Steps', 'ActivityPattern']
                pattern_names = ['Average Duration', 'Average Calories', 'Average Steps', 'Activity vs. Rest Pattern']
                
                pattern_to_show = st.selectbox(
                    "Select Pattern to Visualize", 
                    options=list(zip(pattern_options, pattern_names)),
                    format_func=lambda x: x[1],
                    key="pattern_select"
                )
                
                if pattern_to_show[0] == 'ActivityPattern':
                    # Create stacked bar chart for active vs. rest
                    active_data = exercise_counts.reset_index()
                    
                    fig = px.bar(
                        active_data,
                        x='Weekday',
                        y=['Active', 'Rest'],
                        title='Activity Pattern by Day of Week',
                        labels={'value': 'Count', 'variable': 'Day Type'},
                        color_discrete_map={'Active': '#28a745', 'Rest': '#dc3545'}
                    )
                    
                    # Add percentage labels
                    for i, row in active_data.iterrows():
                        fig.add_annotation(
                            x=row['Weekday'],
                            y=row['Active'] + row['Rest'] / 2,
                            text=f"{row['ActivePct']}%<br>active",
                            showarrow=False,
                            font=dict(color="white", size=12)
                        )
                    
                    fig.update_layout(
                        xaxis=dict(title=''),
                        yaxis=dict(title='Count'),
                        barmode='stack',
                        hovermode='closest',
                        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display most common pattern
                    most_active_day = active_data.loc[active_data['ActivePct'].idxmax()]
                    most_rest_day = active_data.loc[active_data['ActivePct'].idxmin()]
                    
                    st.markdown(f"""
                    ### Weekly Pattern Insights
                    
                    - Your most active day is **{most_active_day['Weekday']}** ({most_active_day['ActivePct']}% active days).
                    - Your most common rest day is **{most_rest_day['Weekday']}** ({100 - most_rest_day['ActivePct']}% rest days).
                    """)
                else:
                    # Create bar chart for selected metric
                    fig = px.bar(
                        weekly_patterns.reset_index(),
                        x='Weekday',
                        y=pattern_to_show[0],
                        title=f'{pattern_to_show[1]} by Day of Week',
                        labels={
                            pattern_to_show[0]: pattern_to_show[1],
                            'Weekday': ''
                        },
                        color=pattern_to_show[0],
                        color_continuous_scale='Viridis'
                    )
                    
                    fig.update_layout(
                        hovermode='closest',
                        coloraxis_showscale=False
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display insights
                    max_day = weekly_patterns[pattern_to_show[0]].idxmax()
                    min_day = weekly_patterns[pattern_to_show[0]].idxmin()
                    average = weekly_patterns[pattern_to_show[0]].mean()
                    
                    st.markdown(f"""
                    ### {pattern_to_show[1]} Insights
                    
                    - Highest on **{max_day}**: {weekly_patterns.loc[max_day, pattern_to_show[0]]:.1f}
                    - Lowest on **{min_day}**: {weekly_patterns.loc[min_day, pattern_to_show[0]]:.1f}
                    - Weekly average: {average:.1f}
                    """)
                
                # Weekly totals chart
                st.subheader("Weekly Volume")
                
                # Group by week
                weekly_totals = filtered_data.groupby(pd.Grouper(key='Date', freq='W-MON')).agg({
                    'Duration(min)': 'sum',
                    'Calories_Burned': 'sum',
                    'Distance(km)': 'sum'
                }).reset_index()
                
                # Rename Date column to Week
                weekly_totals = weekly_totals.rename(columns={'Date': 'Week'})
                weekly_totals['Week'] = weekly_totals['Week'].dt.strftime('%b %d')
                
                # Select metric to visualize
                weekly_options = ['Duration(min)', 'Calories_Burned', 'Distance(km)']
                weekly_names = ['Total Duration (min)', 'Total Calories Burned', 'Total Distance (km)']
                
                weekly_to_show = st.selectbox(
                    "Select Weekly Total to Visualize", 
                    options=list(zip(weekly_options, weekly_names)),
                    format_func=lambda x: x[1],
                    key="weekly_select"
                )
                
                # Create bar chart for weekly totals
                fig = px.bar(
                    weekly_totals,
                    x='Week',
                    y=weekly_to_show[0],
                    title=f'Weekly {weekly_to_show[1]}',
                    labels={
                        weekly_to_show[0]: weekly_to_show[1],
                        'Week': ''
                    },
                    color=weekly_to_show[0],
                    color_continuous_scale='Viridis'
                )
                
                fig.update_layout(
                    hovermode='closest',
                    coloraxis_showscale=False
                )
                
                # Add average line
                avg_value = weekly_totals[weekly_to_show[0]].mean()
                
                fig.add_hline(
                    y=avg_value,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"Average: {avg_value:.1f}",
                    annotation_position="bottom right"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Not enough data to analyze weekly patterns. At least 7 days of data required.")
    
    # Goals & Progress Page
    elif app_mode == "Goals & Progress":
        st.header("Goals & Progress")
        
        # Create tabs for different goal views
        tab1, tab2 = st.tabs(["Current Goals", "Set New Goals"])
        
        # Tab 1: Current Goals
        with tab1:
            st.subheader("Current Goals")
            
            # Display current goals
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Activity Goals")
                
                # Weekly workouts goal
                weekly_workouts_goal = st.session_state.goals['weekly_workouts']
                
                # Calculate current weekly workouts (last 7 days)
                last_7_days = st.session_state.user_data[st.session_state.user_data['Date'] > 
                                                     (datetime.now() - timedelta(days=7))]
                current_weekly_workouts = len(last_7_days[last_7_days['Duration(min)'] > 0])
                
                # Calculate progress percentage
                workout_progress = min(1.0, current_weekly_workouts / weekly_workouts_goal)
                
                st.markdown(f"#### Weekly Workouts: {current_weekly_workouts} / {weekly_workouts_goal}")
                st.progress(workout_progress)
                
                # Weekly calories goal
                weekly_calories_goal = st.session_state.goals['weekly_calories']
                current_weekly_calories = last_7_days['Calories_Burned'].sum()
                calories_progress = min(1.0, current_weekly_calories / weekly_calories_goal)
                
                st.markdown(f"#### Weekly Calories: {int(current_weekly_calories)} / {weekly_calories_goal}")
                st.progress(calories_progress)
                
                # Daily steps goal
                daily_steps_goal = st.session_state.goals['daily_steps']
                current_avg_steps = int(last_7_days['Steps'].mean())
                steps_progress = min(1.0, current_avg_steps / daily_steps_goal)
                
                st.markdown(f"#### Daily Steps: {current_avg_steps} / {daily_steps_goal}")
                st.progress(steps_progress)
            
            with col2:
                st.markdown("### Body Goals")
                
                # Weight goal
                target_weight = st.session_state.goals['target_weight']
                
                if 'Weight(kg)' in st.session_state.user_data.columns:
                    current_weight = st.session_state.user_data['Weight(kg)'].iloc[-1]
                    initial_weight = st.session_state.user_data['Weight(kg)'].iloc[0]
                    
                    # Calculate progress
                    if initial_weight > target_weight:  # Weight loss goal
                        total_to_lose = initial_weight - target_weight
                        lost_so_far = initial_weight - current_weight
                        weight_progress = min(1.0, max(0.0, lost_so_far / total_to_lose if total_to_lose > 0 else 0))
                        
                        st.markdown(f"#### Target Weight: {current_weight:.1f} kg → {target_weight:.1f} kg")
                        st.progress(weight_progress)
                        st.markdown(f"Lost {lost_so_far:.1f} kg of {total_to_lose:.1f} kg goal ({weight_progress*100:.1f}%)")
                    else:  # Weight gain goal
                        total_to_gain = target_weight - initial_weight
                        gained_so_far = current_weight - initial_weight
                        weight_progress = min(1.0, max(0.0, gained_so_far / total_to_gain if total_to_gain > 0 else 0))
                        
                        st.markdown(f"#### Target Weight: {current_weight:.1f} kg → {target_weight:.1f} kg")
                        st.progress(weight_progress)
                        st.markdown(f"Gained {gained_so_far:.1f} kg of {total_to_gain:.1f} kg goal ({weight_progress*100:.1f}%)")
                else:
                    st.markdown(f"#### Target Weight: {target_weight:.1f} kg")
                    st.info("Weight tracking data not available")
                
                # Overall fitness score
                st.markdown("### Overall Fitness Score")
                
                # Calculate a fitness score based on goals' completion
                fitness_score = (
                    (workout_progress * 35) +
                    (calories_progress * 35) +
                    (steps_progress * 30)
                )
                
                st.markdown(f"#### Fitness Score: {fitness_score:.0f}/100")
                
                # Color-coded progress bar
                if fitness_score >= 80:
                    bar_color = "green"
                elif fitness_score >= 50:
                    bar_color = "orange"
                else:
                    bar_color = "red"
                
                st.markdown(
                    f"""
                    <div style="border-radius:20px; height:30px; width:100%; background-color:#ddd">
                        <div style="background-color:{bar_color}; width:{fitness_score}%; height:30px; border-radius:20px; text-align:center; line-height:30px; color:white;">
                            <b>{fitness_score:.0f}</b>
                        </div>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
                
                # Fitness level based on score
                if fitness_score >= 90:
                    st.success("🏆 Excellent! You're exceeding your fitness goals!")
                elif fitness_score >= 75:
                    st.success("💪 Great job! You're on track with your fitness goals.")
                elif fitness_score >= 50:
                    st.warning("👍 Good progress, but there's room for improvement.")
                else:
                    st.error("🏃 You're still working toward your goals. Keep going!")
            
            # Progress over time
            st.subheader("Progress Over Time")
            
            # Group data by week
            weekly_data = st.session_state.user_data.groupby(pd.Grouper(key='Date', freq='W-MON')).agg({
                'Duration(min)': 'sum',
                'Calories_Burned': 'sum',
                'Steps': 'mean',
                'Weight(kg)': 'last' if 'Weight(kg)' in st.session_state.user_data.columns else 'count'
            }).reset_index()
            
            if len(weekly_data) > 1:
                # Create line chart
                progress_metrics = ['Weekly Workouts', 'Weekly Duration', 'Weekly Calories', 'Daily Steps']
                selected_progress = st.selectbox("Select Progress Metric", progress_metrics)
                
                if selected_progress == 'Weekly Workouts':
                    # Calculate workout counts by week
                    workout_counts = st.session_state.user_data[st.session_state.user_data['Duration(min)'] > 0].groupby(
                        pd.Grouper(key='Date', freq='W-MON')
                    ).size().reset_index()
                    workout_counts.columns = ['Date', 'WorkoutCount']
                    
                    fig = px.line(
                        workout_counts,
                        x='Date',
                        y='WorkoutCount',
                        title='Weekly Workout Count',
                        markers=True
                    )
                    
                    # Add goal line
                    fig.add_hline(
                        y=st.session_state.goals['weekly_workouts'],
                        line_dash="dash",
                        line_color="green",
                        annotation_text="Goal"
                    )
                elif selected_progress == 'Weekly Duration':
                    fig = px.line(
                        weekly_data,
                        x='Date',
                        y='Duration(min)',
                        title='Weekly Workout Duration',
                        markers=True
                    )
                elif selected_progress == 'Weekly Calories':
                    fig = px.line(
                        weekly_data,
                        x='Date',
                        y='Calories_Burned',
                        title='Weekly Calories Burned',
                        markers=True
                    )
                    
                    # Add goal line
                    fig.add_hline(
                        y=st.session_state.goals['weekly_calories'],
                        line_dash="dash",
                        line_color="green",
                        annotation_text="Goal"
                    )
                else:  # Daily Steps
                    fig = px.line(
                        weekly_data,
                        x='Date',
                        y='Steps',
                        title='Average Daily Steps',
                        markers=True
                    )
                    
                    # Add goal line
                    fig.add_hline(
                        y=st.session_state.goals['daily_steps'],
                        line_dash="dash",
                        line_color="green",
                        annotation_text="Goal"
                    )
                
                fig.update_layout(
                    xaxis=dict(title='Week'),
                    yaxis=dict(title=''),
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Weight progress chart if available
                if 'Weight(kg)' in st.session_state.user_data.columns:
                    weight_data = st.session_state.user_data[['Date', 'Weight(kg)']].dropna()
                    
                    if len(weight_data) > 1:
                        fig = px.line(
                            weight_data.sort_values('Date'),
                            x='Date',
                            y='Weight(kg)',
                            title='Weight Progress',
                            markers=True
                        )
                        
                        fig.update_layout(
                            xaxis=dict(title='Date'),
                            yaxis=dict(title='Weight (kg)'),
                            hovermode='x unified'
                        )
                        
                        # Add target weight line
                        fig.add_hline(
                            y=st.session_state.goals['target_weight'],
                            line_dash="dash",
                            line_color="green",
                            annotation_text="Target"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Not enough data to show progress over time. Keep tracking your workouts!")
        
        # Tab 2: Set New Goals
        with tab2:
            st.subheader("Set New Goals")
            
            with st.form("goals_form"):
                col1, col2 = st.columns(2)
                
                with col1:
                    weekly_workouts = st.number_input("Weekly Workouts Goal", 
                                                   min_value=1, max_value=14, 
                                                   value=st.session_state.goals['weekly_workouts'])
                    
                    weekly_calories = st.number_input("Weekly Calories Burned Goal", 
                                                   min_value=500, max_value=10000, 
                                                   value=st.session_state.goals['weekly_calories'])
                
                with col2:
                    target_weight = st.number_input("Target Weight (kg)", 
                                                 min_value=40.0, max_value=150.0, 
                                                 value=st.session_state.goals['target_weight'])
                    
                    daily_steps = st.number_input("Daily Steps Goal", 
                                               min_value=1000, max_value=30000, 
                                               value=st.session_state.goals['daily_steps'])
                
                submit_goals = st.form_submit_button("Update Goals")
                
                if submit_goals:
                    st.session_state.goals = {
                        'weekly_workouts': weekly_workouts,
                        'weekly_calories': weekly_calories,
                        'target_weight': target_weight,
                        'daily_steps': daily_steps
                    }
                    st.success("Goals updated successfully!")
            
            # Goal setting tips
            st.subheader("Goal Setting Tips")
            
            st.markdown("""
            ### SMART Goals Framework
            
            Set goals that are:
            
            - **Specific**: Clearly define what you want to accomplish
            - **Measurable**: Include concrete numbers to track progress
            - **Achievable**: Be realistic about what you can accomplish
            - **Relevant**: Ensure goals align with your overall fitness objectives
            - **Time-bound**: Set deadlines to create urgency and focus
            
            ### Recommended Goals Based on Activity Level
            
            **Beginner**:
            - 3-4 workouts per week
            - 1000-1500 weekly calories burned
            - 6,000-8,000 daily steps
            
            **Intermediate**:
            - 4-5 workouts per week
            - 1500-2500 weekly calories burned
            - 8,000-10,000 daily steps
            
            **Advanced**:
            - 5-6 workouts per week
            - 2500+ weekly calories burned
            - 10,000+ daily steps
            
            Remember that consistency is more important than intensity when starting out!
            """)
    
    # Settings Page
    elif app_mode == "Settings":
        st.header("Settings")
        
        # Create tabs for different settings
        tab1, tab2, tab3 = st.tabs(["Profile", "Data Management", "App Settings"])
        
        # Tab 1: Profile Settings
        with tab1:
            st.subheader("User Profile")
            
            with st.form("profile_form"):
                col1, col2 = st.columns(2)
                
                with col1:
                    name = st.text_input("Name", value=st.session_state.user_profile['name'])
                    age = st.number_input("Age", min_value=16, max_value=100, value=st.session_state.user_profile['age'])
                
                with col2:
                    height = st.number_input("Height (cm)", min_value=100, max_value=250, value=st.session_state.user_profile['height'])
                    join_date = st.date_input("Join Date", value=pd.to_datetime(st.session_state.user_profile['join_date']))
                
                submit_profile = st.form_submit_button("Update Profile")
                
                if submit_profile:
                    st.session_state.user_profile = {
                        'name': name,
                        'age': age,
                        'height': height,
                        'join_date': join_date.strftime('%Y-%m-%d')
                    }
                    st.success("Profile updated successfully!")
        
        # Tab 2: Data Management
        with tab2:
            st.subheader("Data Import/Export")
            
            # Export data
            if st.session_state.user_data is not None:
                csv = st.session_state.user_data.to_csv(index=False)
                st.download_button(
                    label="Export All Data",
                    data=csv,
                    file_name="fitness_data_export.csv",
                    mime="text/csv",
                )
            
            # Import data
            st.subheader("Import Data")
            uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
            
            if uploaded_file is not None:
                try:
                    imported_data = pd.read_csv(uploaded_file)
                    
                    # Check if the imported data has at least the required columns
                    required_columns = ['Date', 'Exercise', 'Duration(min)']
                    if all(col in imported_data.columns for col in required_columns):
                        # Convert date strings to datetime objects
                        imported_data['Date'] = pd.to_datetime(imported_data['Date'])
                        
                        if st.button("Confirm Import"):
                            st.session_state.user_data = imported_data
                            st.session_state.user_data.to_csv('fitness_data_comprehensive.csv', index=False)
                            st.success("Data imported successfully!")
                            st.experimental_rerun()
                    else:
                        st.error(f"The CSV file must contain at least the following columns: {', '.join(required_columns)}")
                except Exception as e:
                    st.error(f"Error importing data: {e}")
            
            # Clear data option
            st.subheader("Clear Data")
            st.warning("Danger Zone! This will permanently delete all your fitness data.")
            
            if st.button("Clear All Fitness Data"):
                confirm = st.text_input("Type 'DELETE' to confirm clearing all data")
                if confirm == "DELETE":
                    # Reset data
                    if os.path.exists('fitness_data_comprehensive.csv'):
                        os.remove('fitness_data_comprehensive.csv')
                    
                    st.session_state.user_data = None
                    st.success("All fitness data has been cleared!")
                    st.experimental_rerun()
        
        # Tab 3: App Settings
        with tab3:
            st.subheader("App Settings")
            
            # Theme settings
            st.markdown("### Theme Settings")
            
            st.info("Theme customization is not available in this version.")
            
            # Version info
            st.subheader("About")
            
            st.markdown("""
            ### Personal Fitness Tracker v1.0
            
            Developed using:
            - Python
            - Streamlit
            - Pandas
            - Plotly
            
            This application helps you track and analyze your fitness progress over time.
            """)

# Footer
st.markdown("""
---
Personal Fitness Tracker | Made with ❤️ using Streamlit
""")
