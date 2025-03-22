import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import time
import sqlite3
import datetime
from io import BytesIO
import os

import warnings
warnings.filterwarnings('ignore')

# Initialize database connection
@st.cache_resource
def init_database():
    conn = sqlite3.connect('fitness_tracker.db')
    cursor = conn.cursor()
    
    # Create users table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        age INTEGER,
        weight REAL,
        height REAL,
        gender TEXT,
        goal_steps INTEGER DEFAULT 10000
    )
    ''')
    
    # Create activities table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS activities (
        id INTEGER PRIMARY KEY,
        user_id INTEGER,
        date TEXT,
        activity_type TEXT,
        start_time TEXT,
        end_time TEXT,
        duration INTEGER,
        steps INTEGER,
        calories REAL,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
    ''')
    
    # Create daily_summaries table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS daily_summaries (
        id INTEGER PRIMARY KEY,
        user_id INTEGER,
        date TEXT,
        total_steps INTEGER,
        total_calories REAL,
        active_minutes INTEGER,
        distance REAL,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
    ''')
    
    # Check if we have a user, create one if not
    cursor.execute("SELECT * FROM users LIMIT 1")
    user = cursor.fetchone()
    
    if not user:
        cursor.execute('''
        INSERT INTO users (name, age, weight, height, gender, goal_steps)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', ("Sample User", 30, 70.5, 175.0, "Male", 10000))
        conn.commit()
    
    return conn

# Function to simulate activity data
def simulate_activity_data(conn, user_id, date, activity_level='moderate'):
    cursor = conn.cursor()
    date_str = date.isoformat()
    
    # Clear existing data for this date
    cursor.execute('''
    DELETE FROM activities WHERE user_id = ? AND date = ?
    ''', (user_id, date_str))
    
    cursor.execute('''
    DELETE FROM daily_summaries WHERE user_id = ? AND date = ?
    ''', (user_id, date_str))
    
    conn.commit()
    
    # Define activity levels with steps, durations, etc.
    activity_profiles = {
        'low': {
            'walking_periods': 3,
            'walking_duration': 15,
            'walking_steps_per_min': 90,
            'running_periods': 0,
            'running_duration': 0,
            'running_steps_per_min': 160
        },
        'moderate': {
            'walking_periods': 5,
            'walking_duration': 20,
            'walking_steps_per_min': 100,
            'running_periods': 1,
            'running_duration': 25,
            'running_steps_per_min': 160
        },
        'high': {
            'walking_periods': 6,
            'walking_duration': 25,
            'walking_steps_per_min': 110,
            'running_periods': 2,
            'running_duration': 30,
            'running_steps_per_min': 170
        }
    }
    
    profile = activity_profiles[activity_level]
    
    # Generate walking activities
    for _ in range(profile['walking_periods']):
        # Random start time between 7am and 8pm
        start_hour = np.random.randint(7, 20)
        start_minute = np.random.randint(0, 60)
        
        start_time = datetime.datetime.combine(date, datetime.time(start_hour, start_minute))
        duration = profile['walking_duration']
        end_time = start_time + datetime.timedelta(minutes=duration)
        
        steps = duration * profile['walking_steps_per_min']
        
        # Calculate calories based on user weight
        cursor.execute("SELECT weight FROM users WHERE id = ?", (user_id,))
        weight = cursor.fetchone()[0]
        calories = 3.5 * weight * (duration / 60)  # MET value for walking is approximately 3.5
        
        # Log activity
        cursor.execute('''
        INSERT INTO activities 
        (user_id, date, activity_type, start_time, end_time, duration, steps, calories)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            user_id,
            date_str,
            'walking',
            start_time.isoformat(),
            end_time.isoformat(),
            duration,
            steps,
            calories
        ))
    
    # Generate running activities
    for _ in range(profile['running_periods']):
        # Random start time between 7am and 7pm
        start_hour = np.random.randint(7, 19)
        start_minute = np.random.randint(0, 60)
        
        start_time = datetime.datetime.combine(date, datetime.time(start_hour, start_minute))
        duration = profile['running_duration']
        end_time = start_time + datetime.timedelta(minutes=duration)
        
        steps = duration * profile['running_steps_per_min']
        
        # Calculate calories
        cursor.execute("SELECT weight FROM users WHERE id = ?", (user_id,))
        weight = cursor.fetchone()[0]
        calories = 8.0 * weight * (duration / 60)  # MET value for running is approximately 8.0
        
        # Log activity
        cursor.execute('''
        INSERT INTO activities 
        (user_id, date, activity_type, start_time, end_time, duration, steps, calories)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            user_id,
            date_str,
            'running',
            start_time.isoformat(),
            end_time.isoformat(),
            duration,
            steps,
            calories
        ))
    
    # Calculate daily summary
    cursor.execute('''
    SELECT SUM(steps), SUM(calories), SUM(duration)
    FROM activities
    WHERE user_id = ? AND date = ?
    ''', (user_id, date_str))
    
    result = cursor.fetchone()
    
    total_steps = result[0] or 0
    total_calories = result[1] or 0
    active_minutes = result[2] or 0
    
    # Calculate distance
    cursor.execute("SELECT height FROM users WHERE id = ?", (user_id,))
    height = cursor.fetchone()[0]
    step_length = (height / 100) * 0.43  # average step length in meters
    distance = (total_steps * step_length) / 1000  # convert to km
    
    # Save summary
    cursor.execute('''
    INSERT INTO daily_summaries
    (user_id, date, total_steps, total_calories, active_minutes, distance)
    VALUES (?, ?, ?, ?, ?, ?)
    ''', (
        user_id,
        date_str,
        total_steps,
        total_calories,
        active_minutes,
        distance
    ))
    
    conn.commit()

# Function to get weekly activity data
def get_weekly_activity(conn, user_id):
    cursor = conn.cursor()
    today = datetime.date.today()
    week_ago = today - datetime.timedelta(days=6)  # Last 7 days including today
    
    daily_data = []
    current_date = week_ago
    
    while current_date <= today:
        date_str = current_date.isoformat()
        
        # Check if we have a summary in the database
        cursor.execute('''
        SELECT total_steps, total_calories, active_minutes, distance
        FROM daily_summaries 
        WHERE user_id = ? AND date = ?
        ''', (user_id, date_str))
        
        summary = cursor.fetchone()
        
        if summary:
            daily_data.append({
                'date': date_str,
                'total_steps': summary[0],
                'total_calories': summary[1],
                'active_minutes': summary[2],
                'distance': summary[3]
            })
        else:
            # If no data, maybe simulate some
            simulate_activity_data(conn, user_id, current_date, 'moderate')
            
            cursor.execute('''
            SELECT total_steps, total_calories, active_minutes, distance
            FROM daily_summaries 
            WHERE user_id = ? AND date = ?
            ''', (user_id, date_str))
            
            summary = cursor.fetchone()
            
            if summary:
                daily_data.append({
                    'date': date_str,
                    'total_steps': summary[0],
                    'total_calories': summary[1],
                    'active_minutes': summary[2],
                    'distance': summary[3]
                })
            else:
                # If still no data, add zeros
                daily_data.append({
                    'date': date_str,
                    'total_steps': 0,
                    'total_calories': 0,
                    'active_minutes': 0,
                    'distance': 0
                })
        
        current_date += datetime.timedelta(days=1)
    
    return daily_data

# Function to get daily activity breakdown
def get_daily_activity_breakdown(conn, user_id, date):
    cursor = conn.cursor()
    date_str = date.isoformat() if isinstance(date, datetime.date) else date
    
    # Get activities for the day
    cursor.execute('''
    SELECT activity_type, SUM(duration), SUM(steps), SUM(calories)
    FROM activities
    WHERE user_id = ? AND date = ?
    GROUP BY activity_type
    ''', (user_id, date_str))
    
    activities = cursor.fetchall()
    
    if not activities:
        # Simulate data if none exists
        simulate_activity_data(conn, user_id, date)
        
        cursor.execute('''
        SELECT activity_type, SUM(duration), SUM(steps), SUM(calories)
        FROM activities
        WHERE user_id = ? AND date = ?
        GROUP BY activity_type
        ''', (user_id, date_str))
        
        activities = cursor.fetchall()
    
    result = {}
    
    for activity in activities:
        result[activity[0]] = {
            'duration': activity[1],
            'steps': activity[2],
            'calories': activity[3]
        }
    
    return result

# Function to get user profile
def get_user_profile(conn, user_id=1):
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    user = cursor.fetchone()
    
    if user:
        return {
            'id': user[0],
            'name': user[1],
            'age': user[2],
            'weight': user[3],
            'height': user[4],
            'gender': user[5],
            'goal_steps': user[6]
        }
    return None

# Function to update user profile
def update_user_profile(conn, user_id, **kwargs):
    cursor = conn.cursor()
    
    updates = {}
    for key, value in kwargs.items():
        if value is not None and key in ['name', 'age', 'weight', 'height', 'gender', 'goal_steps']:
            updates[key] = value
    
    if updates:
        set_clause = ", ".join([f"{key} = ?" for key in updates.keys()])
        values = list(updates.values())
        values.append(user_id)
        
        cursor.execute(f"UPDATE users SET {set_clause} WHERE id = ?", values)
        conn.commit()
        return True
    return False

# Function to generate charts for the dashboard
def generate_dashboard_charts(conn, user_id):
    # Get user profile
    user_profile = get_user_profile(conn, user_id)
    
    # Get weekly activity data
    weekly_data = get_weekly_activity(conn, user_id)
    
    # Get today's summary
    today = datetime.date.today()
    today_str = today.isoformat()
    
    cursor = conn.cursor()
    cursor.execute('''
    SELECT total_steps, total_calories, active_minutes, distance
    FROM daily_summaries 
    WHERE user_id = ? AND date = ?
    ''', (user_id, today_str))
    
    today_summary = cursor.fetchone()
    
    if not today_summary:
        # Simulate today if no data
        simulate_activity_data(conn, user_id, today, 'moderate')
        
        cursor.execute('''
        SELECT total_steps, total_calories, active_minutes, distance
        FROM daily_summaries 
        WHERE user_id = ? AND date = ?
        ''', (user_id, today_str))
        
        today_summary = cursor.fetchone()
    
    # Get activity breakdown for today
    activity_breakdown = get_daily_activity_breakdown(conn, user_id, today)
    
    # Create charts
    charts = {}
    
    # Steps progress chart (circular)
    fig_steps, ax = plt.subplots(figsize=(3, 3))
    ax.set_aspect('equal')
    
    total_steps = today_summary[0] if today_summary else 0
    goal_steps = user_profile['goal_steps']
    progress = min(1.0, total_steps / goal_steps)
    
    # Draw background circle
    background = plt.Circle((0.5, 0.5), 0.4, fill=False, linewidth=10, color='#f1f1f1')
    ax.add_artist(background)
    
    # Draw progress arc
    progress_arc = plt.matplotlib.patches.Arc(
        (0.5, 0.5), 0.8, 0.8,
        theta1=90, theta2=90-(progress*360),
        linewidth=10, color='#4CAF50'
    )
    ax.add_patch(progress_arc)
    
    # Add text
    ax.text(0.5, 0.5, f"{total_steps:,}", 
           horizontalalignment='center', verticalalignment='center', 
           fontsize=16, fontweight='bold')
    
    ax.text(0.5, 0.35, "steps", 
           horizontalalignment='center', verticalalignment='center',
           fontsize=12)
    
    ax.text(0.5, 0.8, f"{int(progress*100)}% of goal", 
           horizontalalignment='center', verticalalignment='center',
           fontsize=10)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Save to BytesIO
    buf_steps = BytesIO()
    fig_steps.savefig(buf_steps, format='png', bbox_inches='tight', transparent=True)
    buf_steps.seek(0)
    charts['steps_progress'] = buf_steps
    
    # Weekly step trend
    fig_weekly, ax = plt.subplots(figsize=(10, 4))
    dates = [datetime.date.fromisoformat(day['date']) for day in weekly_data]
    steps = [day['total_steps'] for day in weekly_data]
    
    # Bar colors based on goal achievement
    colors = ['#4CAF50' if s >= goal_steps else '#FFC107' for s in steps]
    
    ax.bar(dates, steps, color=colors, alpha=0.7)
    ax.axhline(y=goal_steps, color='r', linestyle='--', label=f'Goal: {goal_steps:,} steps')
    
    # Format x-axis with day names
    ax.set_xticks(dates)
    ax.set_xticklabels([d.strftime('%a') for d in dates], rotation=0)
    
    # Format y-axis
    ax.set_ylabel('Steps')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f'{int(x):,}'))
    
    ax.set_title('Weekly Step Trend')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    
    buf_weekly = BytesIO()
    fig_weekly.savefig(buf_weekly, format='png', bbox_inches='tight')
    buf_weekly.seek(0)
    charts['weekly_trend'] = buf_weekly
    
    # Activity distribution pie chart
    if activity_breakdown:
        fig_activity, ax = plt.subplots(figsize=(4, 4))
        
        activities = list(activity_breakdown.keys())
        durations = [activity_breakdown[act]['duration'] for act in activities]
        
        # Colors for different activities
        activity_colors = {
            'walking': '#4CAF50',  # green
            'running': '#FF5722',  # orange
            'stationary': '#9E9E9E'  # gray
        }
        
        colors = [activity_colors.get(act, '#2196F3') for act in activities]
        
        ax.pie(durations, labels=[a.capitalize() for a in activities], 
              colors=colors, autopct='%1.1f%%', startangle=90,
              wedgeprops={'edgecolor': 'w', 'linewidth': 1})
        
        ax.set_title('Activity Distribution')
        
        buf_activity = BytesIO()
        fig_activity.savefig(buf_activity, format='png', bbox_inches='tight')
        buf_activity.seek(0)
        charts['activity_distribution'] = buf_activity
    
    # Calories and distance display
    fig_stats, ax = plt.subplots(1, 2, figsize=(8, 3))
    
    # Calories meter
    calories = today_summary[1] if today_summary else 0
    
    # Simplified gauge for calories
    ax[0].add_patch(plt.matplotlib.patches.Rectangle(
        (0.1, 0.3), 0.8, 0.4, color='#f1f1f1', alpha=0.8
    ))
    
    # Normalize calories to a 0-1 scale for display (assuming 1000 cal is max)
    cal_width = min(0.8, calories / 1000 * 0.8)
    ax[0].add_patch(plt.matplotlib.patches.Rectangle(
        (0.1, 0.3), cal_width, 0.4, color='#FF5722'
    ))
    
    ax[0].text(0.5, 0.7, f"{calories:.0f}", 
              horizontalalignment='center', verticalalignment='center',
              fontsize=16, fontweight='bold')
    
    ax[0].text(0.5, 0.2, "Calories Burned", 
              horizontalalignment='center', verticalalignment='center',
              fontsize=10)
    
    ax[0].set_xlim(0, 1)
    ax[0].set_ylim(0, 1)
    ax[0].axis('off')
    
    # Distance meter
    distance = today_summary[3] if today_summary else 0
    
    # Simplified gauge for distance
    ax[1].add_patch(plt.matplotlib.patches.Rectangle(
        (0.1, 0.3), 0.8, 0.4, color='#f1f1f1', alpha=0.8
    ))
    
    # Normalize distance to 0-1 scale (assuming 10km is max)
    dist_width = min(0.8, distance / 10 * 0.8)
    ax[1].add_patch(plt.matplotlib.patches.Rectangle(
        (0.1, 0.3), dist_width, 0.4, color='#2196F3'
    ))
    
    ax[1].text(0.5, 0.7, f"{distance:.2f} km", 
              horizontalalignment='center', verticalalignment='center',
              fontsize=16, fontweight='bold')
    
    ax[1].text(0.5, 0.2, "Distance Covered", 
              horizontalalignment='center', verticalalignment='center',
              fontsize=10)
    
    ax[1].set_xlim(0, 1)
    ax[1].set_ylim(0, 1)
    ax[1].axis('off')
    
    plt.tight_layout()
    
    buf_stats = BytesIO()
    fig_stats.savefig(buf_stats, format='png', bbox_inches='tight')
    buf_stats.seek(0)
    charts['stats'] = buf_stats
    
    # Close all figures to free memory
    plt.close('all')
    
    return charts

# Create sample data files needed for the calorie predictor
def create_sample_data_files():
    # Create calories.csv if it doesn't exist
    if not os.path.exists('calories.csv'):
        calories_df = pd.DataFrame({
            'User_ID': range(1, 101),
            'Calories': np.random.uniform(100, 1000, 100)
        })
        calories_df.to_csv('calories.csv', index=False)
    
    # Create exercise.csv if it doesn't exist
    if not os.path.exists('exercise.csv'):
        exercise_df = pd.DataFrame({
            'User_ID': range(1, 101),
            'Gender': np.random.choice(['male', 'female'], 100),
            'Age': np.random.randint(15, 75, 100),
            'Height': np.random.uniform(150, 200, 100),
            'Weight': np.random.uniform(45, 120, 100),
            'Duration': np.random.randint(5, 60, 100),
            'Heart_Rate': np.random.randint(70, 180, 100),
            'Body_Temp': np.random.uniform(36.5, 39.0, 100)
        })
        exercise_df.to_csv('exercise.csv', index=False)

# Main Streamlit app
def main():
    # Create sample data files needed for calorie predictor
    create_sample_data_files()
    
    # Page config with expanded layout for better visibility
    st.set_page_config(page_title="Personal Fitness Tracker", 
                      page_icon="🏃‍♂️", layout="wide",
                      initial_sidebar_state="expanded")
    
    # Custom CSS for better appearance
    st.markdown("""
    <style>
    .main {
        padding: 1rem 1rem;
    }
    .stProgress > div > div {
        background-color: #4CAF50;
    }
    h1, h2, h3 {
        color: #2196F3;
    }
    .stSidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    .css-1d391kg {
        padding-top: 1rem;
    }
    footer {
        visibility: hidden;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize database
    conn = init_database()
    
    # Page header with logo
    col1, col2 = st.columns([1, 5])
    with col1:
        st.image("https://img.icons8.com/fluency/96/000000/running.png", width=80)
    with col2:
        st.title("Personal Fitness Tracker")
        st.write("Track, analyze, and improve your physical activity with this Python-powered fitness tracker app.")
    
    # Navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Dashboard", "Activity Log", "Calorie Predictor", "Profile Settings"])
    
    # Get user profile
    user_profile = get_user_profile(conn)
    
    # Show user profile summary in sidebar
    st.sidebar.write(f"**User:** {user_profile['name']}")
    st.sidebar.write(f"**Goal:** {user_profile['goal_steps']:,} steps/day")
    
    # App version and info
    st.sidebar.markdown("---")
    st.sidebar.info("v1.0.0 - Built with Streamlit and Python")
    st.sidebar.markdown("---")
    
    if page == "Dashboard":
        st.header("Your Fitness Dashboard")
        st.subheader(f"Welcome, {user_profile['name']}!")
        
        # Generate charts
        charts = generate_dashboard_charts(conn, user_profile['id'])
        
        # Display today's date with nice formatting
        today = datetime.date.today()
        st.markdown(f"**📅 Today**: {today.strftime('%A, %B %d, %Y')}")
        st.markdown("---")
        
        # Create layout with columns
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### Today's Steps")
            st.image(charts['steps_progress'])
        
        with col2:
            st.markdown("### Weekly Progress")
            st.image(charts['weekly_trend'])
        
        # Second row
        col1, col2 = st.columns(2)
        
        with col1:
            if 'activity_distribution' in charts:
                st.markdown("### Activity Breakdown")
                st.image(charts['activity_distribution'])
            else:
                st.markdown("### Activity Breakdown")
                st.info("No activity data available for today. Start moving!")
        
        with col2:
            st.markdown("### Stats Summary")
            st.image(charts['stats'])
            
            # Get daily summary
            cursor = conn.cursor()
            cursor.execute('''
            SELECT total_steps, total_calories, active_minutes, distance
            FROM daily_summaries 
            WHERE user_id = ? AND date = ?
            ''', (user_profile['id'], today.isoformat()))
            
            today_summary = cursor.fetchone()
            
            if today_summary:
                st.markdown("**⏱️ Active Time**: {} minutes".format(today_summary[2]))
                
                # Calculate goal progress
                goal_steps = user_profile['goal_steps']
                progress = min(100, (today_summary[0] / goal_steps) * 100)
                
                # Display progress in percentage
                st.progress(progress/100)
                st.markdown(f"**{progress:.1f}%** of daily step goal completed")
        
        # Motivational message
        st.markdown("---")
        
        # Adaptive motivation based on progress
        if today_summary:
            progress = (today_summary[0] / user_profile['goal_steps']) * 100
            
            if progress < 30:
                st.info("💪 Get moving! You still have plenty of time to reach your goal today.")
            elif progress < 70:
                st.success("👍 Good progress! Keep going to reach your daily goal.")
            else:
                st.balloons()
                st.success("🎉 Excellent work! You're well on your way to achieving your goal.")
        
        # Quick tips section
        st.markdown("---")
        st.markdown("### 💡 Quick Tips")
        
        tips_col1, tips_col2 = st.columns(2)
        
        with tips_col1:
            st.markdown("**Boost Your Steps**")
            st.markdown("• Take the stairs instead of elevator")
            st.markdown("• Park farther away from entrances")
            st.markdown("• Set hourly reminders to walk around")
        
        with tips_col2:
            st.markdown("**Maximize Your Tracking**")
            st.markdown("• Keep your phone with you during activities")
            st.markdown("• Log manual workouts in the Activity Log")
            st.markdown("• Update your profile for accurate calorie estimates")
    
    elif page == "Activity Log":
        st.header("Activity Log")
        st.markdown("Record and review your physical activities")
        
        # Date selection with improved styling
        col1, col2 = st.columns([2, 1])
        with col1:
            selected_date = st.date_input("Select date:", datetime.date.today())
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)  # Add spacing
            if st.button("📊 Generate Sample Data", help="Create sample activity data for this date"):
                with st.spinner("Generating activity data..."):
                    activity_level = st.selectbox("Activity level:", ["low", "moderate", "high"])
                    simulate_activity_data(conn, user_profile['id'], selected_date, activity_level)
                    st.success("Sample data generated!")
                    st.experimental_rerun()
        
        # Get activities for the selected date
        cursor = conn.cursor()
        cursor.execute('''
        SELECT activity_type, start_time, end_time, duration, steps, calories
        FROM activities
        WHERE user_id = ? AND date = ?
        ORDER BY start_time
        ''', (user_profile['id'], selected_date.isoformat()))
        
        activities = cursor.fetchall()
        
        if not activities:
            st.warning(f"📝 No activities recorded for {selected_date.strftime('%A, %B %d')}.")
            
            # Add a nice empty state illustration
            st.markdown("""
            <div style="text-align: center; padding: 20px;">
                <img src="https://img.icons8.com/bubbles/200/000000/calendar.png" width="150">
                <p>Generate sample data or add activities manually</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Add activity form
            with st.expander("➕ Add Activity Manually"):
                with st.form("add_activity_form"):
                    activity_type = st.selectbox("Activity type:", ["walking", "running", "stationary"])
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        # Convert to string for time_input
                        start_time = st.time_input("Start time:", datetime.time(8, 0))
                    with col2:
                        duration = st.number_input("Duration (minutes):", 1, 180, 30)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        steps = st.number_input("Steps (approx):", 0, 10000, 
                                              100 * duration if activity_type == "walking" else 
                                              160 * duration if activity_type == "running" else 0)
                    with col2:
                        # Calculate default calories based on activity type and user weight
                        met = 3.5 if activity_type == "walking" else 8.0 if activity_type == "running" else 1.0
                        default_calories = met * user_profile['weight'] * (duration / 60)
                        calories = st.number_input("Calories burned:", 0.0, 2000.0, float(default_calories), 5.0)
                    
                    submitted = st.form_submit_button("Add Activity")
                    
                    if submitted:
                        # Calculate end time
                        start_datetime = datetime.datetime.combine(selected_date, start_time)
                        end_datetime = start_datetime + datetime.timedelta(minutes=duration)
                        
                        # Insert activity
                        cursor.execute('''
                        INSERT INTO activities 
                        (user_id, date, activity_type, start_time, end_time, duration, steps, calories)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            user_profile['id'],
                            selected_date.isoformat(),
                            activity_type,
                            start_datetime.isoformat(),
                            end_datetime.isoformat(),
                            duration,
                            steps,
                            calories
                        ))
                        
                        # Recalculate daily summary
                        cursor.execute('''
                        DELETE FROM daily_summaries WHERE user_id = ? AND date = ?
                        ''', (user_profile['id'], selected_date.isoformat()))
                        
                        cursor.execute('''
                        SELECT SUM(steps), SUM(calories), SUM(duration)
                        FROM activities
                        WHERE user_id = ? AND date = ?
                        ''', (user_profile['id'], selected_date.isoformat()))
                        
                        result = cursor.fetchone()
                        
                        total_steps = result[0] or 0
                        total_calories = result[1] or 0
                        active_minutes = result[2] or 0
                        
                        # Calculate distance
                        step_length = (user_profile['height'] / 100) * 0.43
                        distance = (total_steps * step_length) / 1000
                        
                        cursor.execute('''
                        INSERT INTO daily_summaries
                        (user_id, date, total_steps, total_calories, active_minutes, distance)
                        VALUES (?, ?, ?, ?, ?, ?)
                        ''', (
                            user_profile['id'],
                            selected_date.isoformat(),
                            total_steps,
                            total_calories,
                            active_minutes,
                            distance
                        ))
                        
                        conn.commit()
                        st.success("Activity added successfully!")
                        st.experimental_rerun()
        else:
            # Display activities in a table with improved formatting
            activities_df = pd.DataFrame(activities, 
                                       columns=["Activity", "Start Time", "End Time", 
                                               "Duration (min)", "Steps", "Calories"])
            
            # Format datetime columns
            activities_df["Start Time"] = pd.to_datetime(activities_df["Start Time"]).dt.strftime("%H:%M")
            activities_df["End Time"] = pd.to_datetime(activities_df["End Time"]).dt.strftime("%H:%M")
            
            # Capitalize activity names
            activities_df["Activity"] = activities_df["Activity"].str.capitalize()
            
            # Format numbers
            activities_df["Steps"] = activities_df["Steps"].map("{:,}".format)
            activities_df["Calories"] = activities_df["Calories"].round(1)
            
            # Style the dataframe
            st.dataframe(activities_df, height=300)
            
            # Display add activity expander
            with st.expander("➕ Add Another Activity"):
                with st.form("add_more_activity_form"):
                    activity_type = st.selectbox("Activity type:", ["walking", "running", "stationary"], key="more_act_type")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        start_time = st.time_input("Start time:", datetime.time(8, 0), key="more_start_time")
                    with col2:
                        duration = st.number_input("Duration (minutes):", 1, 180, 30, key="more_duration")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        steps = st.number_input("Steps (approx):", 0, 10000, 
                                              100 * duration if activity_type == "walking" else 
                                              160 * duration if activity_type == "running" else 0,
                                              key="more_steps")
                    with col2:
                        # Calculate default calories based on activity type and user weight
                        met = 3.5 if activity_type == "walking" else 8.0 if activity_type == "running" else 1.0
                        default_calories = met * user_profile['weight'] * (duration / 60)
                        calories = st.number_input("Calories burned:", 0.0, 2000.0, float(default_calories), 5.0,
                                                 key="more_calories")
                    
                    submitted = st.form_submit_button("Add Activity")
                    
                    if submitted:
                        # Calculate end time
                        start_datetime = datetime.datetime.combine(selected_date, start_time)
                        end_datetime = start_datetime + datetime.timedelta(minutes=duration)
                        
                        # Insert activity (same as above)
                        cursor.execute('''
                        INSERT INTO activities 
                        (user_id, date, activity_type, start_time, end_time, duration, steps, calories)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            user_profile['id'],
                            selected_date.isoformat(),
                            activity_type,
                            start_datetime.isoformat(),
                            end_datetime.isoformat(),
                            duration,
                            steps,
                            calories
                        ))
                        
                        # Recalculate daily summary (same as above)
                        cursor.execute('''
                        DELETE FROM daily_summaries WHERE user_id = ? AND date = ?
                        ''', (user_profile['id'], selected_date.isoformat()))
                        
                        cursor.execute('''
                        SELECT SUM(steps), SUM(calories), SUM(duration)
                        FROM activities
                        WHERE user_id = ? AND date = ?
                        ''', (user_profile['id'], selected_date.isoformat()))
                        
                        result = cursor.fetchone()
                        
                        total_steps = result[0] or 0
                        total_calories = result[1] or 0
                        active_minutes = result[2] or 0
                        
                        # Calculate distance
                        step_length = (user_profile['height'] / 100) * 0.43
                        distance = (total_steps * step_length) / 1000
                        
                        cursor.execute('''
                        INSERT INTO daily_summaries
                        (user_id, date, total_steps, total_calories, active_minutes, distance)
                        VALUES (?, ?, ?, ?, ?, ?)
                        ''', (
                            user_profile['id'],
                            selected_date.isoformat(),
                            total_steps,
                            total_calories,
                            active_minutes,
                            distance
                        ))
                        
                        conn.commit()
                        st.success("Activity added successfully!")
                        st.experimental_rerun()
            
            # Get daily summary
            cursor.execute('''
            SELECT total_steps, total_calories, active_minutes, distance
            FROM daily_summaries
            WHERE user_id = ? AND date = ?
            ''', (user_profile['id'], selected_date.isoformat()))
            
            summary = cursor.fetchone()
            
            if summary:
                st.markdown("### Daily Summary")
                
                # Create colorful metric cards
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(f"""
                    <div style="padding:10px;border-radius:5px;background-color:#e8f5e9;">
                        <h4 style="margin:0;color:#2e7d32;">Total Steps</h4>
                        <p style="font-size:24px;font-weight:bold;margin:0;">{summary[0]:,}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div style="padding:10px;border-radius:5px;background-color:#fff3e0;">
                        <h4 style="margin:0;color:#e64a19;">Calories Burned</h4>
                        <p style="font-size:24px;font-weight:bold;margin:0;">{summary[1]:.1f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div style="padding:10px;border-radius:5px;background-color:#e1f5fe;">
                        <h4 style="margin:0;color:#0288d1;">Active Minutes</h4>
                        <p style="font-size:24px;font-weight:bold;margin:0;">{summary[2]}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    st.markdown(f"""
                    <div style="padding:10px;border-radius:5px;background-color:#f3e5f5;">
                        <h4 style="margin:0;color:#7b1fa2;">Distance (km)</h4>
                        <p style="font-size:24px;font-weight:bold;margin:0;">{summary[3]:.2f}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Generate visualization
            st.markdown("### Activity Visualization")
            
            # Create timeline with better styling
            fig, ax = plt.subplots(figsize=(10, 4))
            
            activity_colors = {
                'walking': '#4CAF50',  # green
                'running': '#FF5722',  # orange
                'stationary': '#9E9E9E'  # gray
            }
            
            for activity in activities:
                act_type, start, end = activity[0], activity[1], activity[2]
                start_dt = datetime.datetime.fromisoformat(start)
                end_dt = datetime.datetime.fromisoformat(end)
                
                ax.barh(0, (end_dt - start_dt).total_seconds() / 60, 
                       left=(start_dt - datetime.datetime.combine(selected_date, datetime.time.min)).total_seconds() / 60,
                       color=activity_colors.get(act_type, '#2196F3'),
                       alpha=0.7, height=0.5)
            
            # Add activity labels
            handles = [plt.matplotlib.patches.Patch(color=color, label=act.capitalize()) 
                      for act, color in activity_colors.items() 
                      if act in [a[0] for a in activities]]
            
            ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.15),
                     ncol=3)
            
            # Format axes
            ax.set_yticks([])
            
            # Set x-axis to hours in the day
            ax.set_xlim(0, 24 * 60)  # minutes in a day
            ax.set_xticks(np.arange(0, 24*60+1, 60))
            ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 25)])
            
            ax.set_xlabel("Time of Day")
            ax.set_title(f"Activity Timeline - {selected_date.strftime('%A, %B %d')}")
            
            ax.grid(axis='x', linestyle='--', alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Option to delete activities
            if st.button("🗑️ Delete All Activities for This Date"):
                confirmation = st.checkbox("I confirm I want to delete all activities for this date", key="delete_confirm")
                if confirmation:
                    cursor.execute('''
                    DELETE FROM activities WHERE user_id = ? AND date = ?
                    ''', (user_profile['id'], selected_date.isoformat()))
                    
                    cursor.execute('''
                    DELETE FROM daily_summaries WHERE user_id = ? AND date = ?
                    ''', (user_profile['id'], selected_date.isoformat()))
                    
                    conn.commit()
                    st.success("All activities for this date have been deleted.")
                    st.experimental_rerun()
    
    elif page == "Calorie Predictor":
        st.header("Calorie Burn Predictor")
        st.markdown("Estimate calories burned based on your activity parameters")
        
        # User input parameters (from sidebar)
        st.sidebar.header("Input Parameters")
        
        age = st.sidebar.slider("Age:", 10, 100, user_profile['age'])
        
        # Calculate BMI from profile
        height_m = user_profile['height'] / 100
        bmi = user_profile['weight'] / (height_m ** 2)
        bmi = round(bmi, 1)  # Round BMI to 1 decimal place
        bmi = st.sidebar.slider("BMI:", 15.0, 40.0, float(bmi), 0.1)
        
        duration = st.sidebar.slider("Duration (min):", 0, 60, 30)
        heart_rate = st.sidebar.slider("Heart Rate (bpm):", 60, 200, 120)
        body_temp = st.sidebar.slider("Body Temperature (°C):", 36.0, 40.0, 37.2, 0.1)
        gender = st.sidebar.radio("Gender:", ["Male", "Female"], 0 if user_profile['gender'] == "Male" else 1)
        
        # Use column names matching the model
        data_model = {
            "Age": age,
            "BMI": bmi,
            "Duration": duration,
            "Heart_Rate": heart_rate,
            "Body_Temp": body_temp,
            "Gender_male": 1 if gender == "Male" else 0
        }
        
        # Create DataFrame from user input
        df = pd.DataFrame(data_model, index=[0])
        
        # Display user parameters in a clean card format
        st.markdown("### Your Parameters")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div style="padding:15px;border-radius:5px;background-color:#f5f5f5;margin-bottom:10px;">
                <p style="margin:0;"><strong>Age:</strong> {age} years</p>
            </div>
            <div style="padding:15px;border-radius:5px;background-color:#f5f5f5;margin-bottom:10px;">
                <p style="margin:0;"><strong>BMI:</strong> {bmi}</p>
            </div>
            <div style="padding:15px;border-radius:5px;background-color:#f5f5f5;margin-bottom:10px;">
                <p style="margin:0;"><strong>Duration:</strong> {duration} minutes</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style="padding:15px;border-radius:5px;background-color:#f5f5f5;margin-bottom:10px;">
                <p style="margin:0;"><strong>Heart Rate:</strong> {heart_rate} bpm</p>
            </div>
            <div style="padding:15px;border-radius:5px;background-color:#f5f5f5;margin-bottom:10px;">
                <p style="margin:0;"><strong>Body Temperature:</strong> {body_temp} °C</p>
            </div>
            <div style="padding:15px;border-radius:5px;background-color:#f5f5f5;margin-bottom:10px;">
                <p style="margin:0;"><strong>Gender:</strong> {gender}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Create synthetic training data
        n_samples = 1000
        np.random.seed(42)
        
        # Generate synthetic features
        X_synth = pd.DataFrame({
            "Age": np.random.randint(15, 75, n_samples),
            "BMI": np.random.uniform(18, 35, n_samples),
            "Duration": np.random.randint(5, 60, n_samples),
            "Heart_Rate": np.random.randint(70, 180, n_samples),
            "Body_Temp": np.random.uniform(36.5, 39.0, n_samples),
            "Gender_male": np.random.randint(0, 2, n_samples)
        })
        
        # Generate target with some relationships to features
        y_synth = (
            0.05 * X_synth["Age"] +
            1.5 * X_synth["BMI"] +
            3.5 * X_synth["Duration"] +
            0.8 * X_synth["Heart_Rate"] +
            15 * (X_synth["Body_Temp"] - 37) +
            10 * X_synth["Gender_male"] +
            np.random.normal(0, 15, n_samples)
        )
        
        # Ensure no negative calories
        y_synth = np.maximum(0, y_synth)
        
        # Split data for training
        X_train, X_test, y_train, y_test = train_test_split(X_synth, y_synth, test_size=0.2, random_state=42)
        
        # Train a Random Forest model with visual progress
        st.markdown("### Model Training")
        
        with st.spinner("Training machine learning model..."):
            progress_bar = st.progress(0)
            
            # Simulate model training with progress
            for i in range(100):
                progress_bar.progress(i + 1)
                time.sleep(0.01)
            
            model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
            model.fit(X_train, y_train)
        
        st.success("Model trained successfully!")
        
        # Make prediction with animation
        st.markdown("### Prediction Result")
        
        prediction = model.predict(df)
        
        # Create an animated prediction display
        placeholder = st.empty()
        for i in range(1, 11):
            partial_result = prediction[0] * i / 10
            placeholder.markdown(f"""
            <div style="text-align:center;padding:30px;border-radius:10px;background-color:#e8f5e9;">
                <h2 style="margin:0;color:#2e7d32;">Estimated Calories Burned</h2>
                <p style="font-size:48px;font-weight:bold;margin:10px 0;">{partial_result:.1f}</p>
                <p>kilocalories</p>
            </div>
            """, unsafe_allow_html=True)
            time.sleep(0.05)
        
        # Final prediction display
        placeholder.markdown(f"""
        <div style="text-align:center;padding:30px;border-radius:10px;background-color:#e8f5e9;">
            <h2 style="margin:0;color:#2e7d32;">Estimated Calories Burned</h2>
            <p style="font-size:48px;font-weight:bold;margin:10px 0;">{prediction[0]:.2f}</p>
            <p>kilocalories</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Feature importance visualization
        st.markdown("### What Factors Matter Most?")
        
        # Calculate and display feature importance with improved styling
        feature_imp = pd.DataFrame({'Feature': X_train.columns, 'Importance': model.feature_importances_})
        feature_imp = feature_imp.sort_values('Importance', ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 4))
        
        # Use a more attractive color palette
        bar_colors = ['#1976D2', '#2196F3', '#64B5F6', '#90CAF9', '#BBDEFB', '#E3F2FD']
        
        # Create the horizontal bar chart
        sns.barplot(x='Importance', y='Feature', data=feature_imp, palette=bar_colors)
        
        # Customize the chart
        ax.set_title('Factors Influencing Calorie Burn', fontsize=16)
        ax.set_xlabel('Relative Importance', fontsize=12)
        ax.set_ylabel('')  # Remove y-axis label
        
        # Add value labels to the bars
        for i, v in enumerate(feature_imp['Importance']):
            ax.text(v + 0.01, i, f"{v:.3f}", va='center')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Add explanation of the factors
        st.markdown("""
        ### Understanding The Factors
        
        The chart above shows how different factors influence calorie burn during physical activity:
        
        - **Duration**: Typically the most significant factor - the longer you exercise, the more calories you burn
        - **Heart Rate**: Higher heart rates generally indicate more intense activity, resulting in greater calorie burn
        - **BMI**: Your body mass index affects how efficiently you burn calories
        - **Body Temperature**: Elevated body temperature often correlates with increased metabolic activity
        - **Age**: Metabolic rates naturally change with age, affecting calorie expenditure
        - **Gender**: Biological differences can impact calorie burn rates
        """)
        
        # Similar activities section with better styling
        st.markdown("### Similar Activities")
        st.markdown("Activities with similar calorie burn estimates:")
        
        # Create synthetic similar activities with more realistic data
        similar_range = [max(0, prediction[0] - 50), prediction[0] + 50]
        
        # Activity types with realistic MET values
        activities_with_met = {
            "Walking (moderate pace)": 3.5,
            "Walking (brisk pace)": 5.0,
            "Jogging": 7.0,
            "Running": 9.0,
            "Cycling (casual)": 4.0,
            "Cycling (moderate)": 6.0,
            "Swimming (leisure)": 6.0,
            "Dancing": 4.5,
            "Yoga": 2.5,
            "High-intensity interval training": 8.5
        }
        
        similar_acts = []
        
        # Generate 5 similar activities based on the prediction
        for i in range(5):
            # Select a random activity type
            act_type = np.random.choice(list(activities_with_met.keys()))
            met_value = activities_with_met[act_type]
            
            # Calculate a duration that would produce similar calories
            target_calories = np.random.uniform(similar_range[0], similar_range[1])
            
            # calories = MET * weight * hours
            # duration (min) = (calories / (MET * weight)) * 60
            duration_min = max(5, min(120, (target_calories / (met_value * user_profile['weight'])) * 60))
            duration_min = round(duration_min)
            
            # Recalculate actual calories with the rounded duration
            actual_calories = met_value * user_profile['weight'] * (duration_min / 60)
            
            similar_acts.append({
                "Activity": act_type,
                "Duration (min)": duration_min,
                "Intensity": "Low" if met_value < 4 else "Medium" if met_value < 7 else "High",
                "Calories": actual_calories
            })
        
        similar_df = pd.DataFrame(similar_acts)
        similar_df["Calories"] = similar_df["Calories"].round(1)
        
        # Display the similar activities with improved styling
        st.markdown("""
        <style>
        .dataframe-container {
            border-radius: 5px;
            overflow: hidden;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        </style>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
        st.dataframe(similar_df, height=220)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Tips section with actionable insights
        st.markdown("### 💡 Fitness Tips")
        
        # Tips based on user input data
        tips = []
        
        if heart_rate > 160:
            tips.append("Your selected heart rate is quite high. For sustainable cardio, consider keeping your heart rate between 120-150 bpm for most workouts.")
        
        if duration < 20:
            tips.append("Try to aim for at least 20-30 minutes of continuous activity for cardiovascular benefits.")
        
        if bmi > 25:
            tips.append("Combining cardio with strength training could help improve body composition and increase your resting metabolic rate.")
        
        if age > 50:
            tips.append("As we age, recovery becomes more important. Make sure to include rest days in your fitness routine.")
        
        if not tips:
            # Default tips if none of the specific conditions apply
            tips = [
                "Stay hydrated during your workout sessions for optimal performance.",
                "Tracking your heart rate can help ensure you're working at the right intensity.",
                "For sustainable fitness progress, consistency matters more than intensity."
            ]
        
        # Display tips in an attractive format
        for tip in tips:
            st.info(f"**Tip**: {tip}")
    
    elif page == "Profile Settings":
        st.header("User Profile Settings")
        st.markdown("Customize your profile information and app preferences")
        
        # Create tabs for different settings
        tab1, tab2, tab3 = st.tabs(["Profile Info", "Data Management", "App Settings"])
        
        with tab1:  # Profile Info
            # Display current profile in a clean card
            st.markdown("### Current Profile")
            
            # Create a card-style profile display
            st.markdown(f"""
            <div style="padding:20px;border-radius:10px;background-color:#f8f9fa;margin-bottom:20px;box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                <h3 style="margin-top:0;">{user_profile['name']}</h3>
                <p><strong>Age:</strong> {user_profile['age']} years</p>
                <p><strong>Weight:</strong> {user_profile['weight']} kg</p>
                <p><strong>Height:</strong> {user_profile['height']} cm</p>
                <p><strong>Gender:</strong> {user_profile['gender']}</p>
                <p><strong>Daily Step Goal:</strong> {user_profile['goal_steps']:,}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Form for updating profile with improved styling
            st.markdown("### Update Profile")
            
            with st.form("profile_form"):
                name = st.text_input("Name:", user_profile['name'])
                
                col1, col2 = st.columns(2)
                with col1:
                    age = st.number_input("Age:", 10, 100, user_profile['age'])
                    height = st.number_input("Height (cm):", 100.0, 220.0, user_profile['height'], 0.5)
                    gender = st.radio("Gender:", ["Male", "Female"], 0 if user_profile['gender'] == "Male" else 1)
                
                with col2:
                    weight = st.number_input("Weight (kg):", 30.0, 150.0, user_profile['weight'], 0.5)
                    goal_steps = st.number_input("Daily Step Goal:", 1000, 30000, user_profile['goal_steps'], 1000)
                
                submitted = st.form_submit_button("Update Profile")
                
                if submitted:
                    # Update profile
                    result = update_user_profile(
                        conn, 
                        user_profile['id'],
                        name=name,
                        age=age,
                        weight=weight,
                        height=height,
                        gender=gender,
                        goal_steps=goal_steps
                    )
                    
                    if result:
                        st.success("Profile updated successfully!")
                        # Force page refresh to show updated profile
                        st.experimental_rerun()
            
            # Calculate and display BMI
            if st.checkbox("Calculate BMI"):
                height_m = user_profile['height'] / 100
                bmi = user_profile['weight'] / (height_m ** 2)
                bmi = round(bmi, 1)
                
                # BMI categories
                if bmi < 18.5:
                    bmi_category = "Underweight"
                    category_color = "#FFC107"  # amber
                elif bmi < 25:
                    bmi_category = "Normal weight"
                    category_color = "#4CAF50"  # green
                elif bmi < 30:
                    bmi_category = "Overweight"
                    category_color = "#FF9800"  # orange
                else:
                    bmi_category = "Obese"
                    category_color = "#F44336"  # red
                
                # Display BMI in a nice card
                st.markdown(f"""
                <div style="padding:20px;border-radius:10px;background-color:{category_color}20;margin:20px 0;border-left:5px solid {category_color};">
                    <h3 style="margin:0;color:{category_color};">Your BMI: {bmi}</h3>
                    <p style="font-size:18px;margin:5px 0 0 0;">Category: <strong>{bmi_category}</strong></p>
                </div>
                """, unsafe_allow_html=True)
                
                # BMI chart
                fig, ax = plt.subplots(figsize=(10, 3))
                
                # BMI categories with colors
                categories = [
                    {"name": "Underweight", "min": 15, "max": 18.5, "color": "#FFC107"},
                    {"name": "Normal", "min": 18.5, "max": 25, "color": "#4CAF50"},
                    {"name": "Overweight", "min": 25, "max": 30, "color": "#FF9800"},
                    {"name": "Obese", "min": 30, "max": 40, "color": "#F44336"}
                ]
                
                # Draw BMI ranges
                for cat in categories:
                    ax.add_patch(plt.matplotlib.patches.Rectangle(
                        (cat["min"], 0), cat["max"] - cat["min"], 1,
                        color=cat["color"], alpha=0.7,
                        label=cat["name"]
                    ))
                    ax.text((cat["min"] + cat["max"]) / 2, 0.5, cat["name"],
                          horizontalalignment='center', verticalalignment='center',
                          fontweight='bold', color='white')
                
                # Add user's BMI marker
                ax.axvline(x=bmi, color='blue', linestyle='--', linewidth=2)
                ax.text(bmi, 1.1, f"Your BMI: {bmi}", 
                      horizontalalignment='center', color='blue', fontweight='bold')
                
                ax.set_xlim(15, 40)
                ax.set_ylim(0, 1.2)
                ax.set_xlabel("BMI")
                ax.set_yticks([])
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Add BMI information
                st.markdown("""
                **About BMI:**
                
                Body Mass Index (BMI) is a simple calculation using a person's weight and height. While useful as a general guideline, it has limitations as it doesn't account for factors like muscle mass, bone density, or overall body composition.
                
                **BMI Categories:**
                - **Below 18.5**: Underweight
                - **18.5 to 24.9**: Normal weight
                - **25.0 to 29.9**: Overweight
                - **30.0 and above**: Obese
                """)
        
        with tab2:  # Data Management
            st.markdown("### Data Management")
            
            # Generate sample data section
            st.markdown("#### Generate Sample Data")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Generate Sample Data for Today"):
                    with st.spinner("Generating sample data..."):
                        activity_level = st.selectbox(
                            "Activity level:", 
                            ["low", "moderate", "high"],
                            index=1
                        )
                        simulate_activity_data(conn, user_profile['id'], datetime.date.today(), activity_level)
                        st.success("Sample data generated for today!")
            
            with col2:
                if st.button("Generate Sample Data for Past Week"):
                    with st.spinner("Generating sample data for the past week..."):
                        progress_bar = st.progress(0)
                        today = datetime.date.today()
                        
                        for i in range(7):
                            # Update progress bar
                            progress_bar.progress((i+1)/7)
                            
                            date = today - datetime.timedelta(days=i)
                            # Randomly select activity level
                            activity_level = np.random.choice(['low', 'moderate', 'high'], 
                                                            p=[0.2, 0.5, 0.3])
                            simulate_activity_data(conn, user_profile['id'], date, activity_level)
                        
                        st.success("Sample data generated for the past week!")
            
            # Export data section
            st.markdown("#### Export Data")
            
            # Get date range for export
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Start date:", datetime.date.today() - datetime.timedelta(days=7))
            with col2:
                end_date = st.date_input("End date:", datetime.date.today())
            
            # Check if we have data to export
            cursor = conn.cursor()
            cursor.execute('''
            SELECT COUNT(*) FROM activities
            WHERE user_id = ? AND date >= ? AND date <= ?
            ''', (user_profile['id'], start_date.isoformat(), end_date.isoformat()))
            
            count = cursor.fetchone()[0]
            
            if count > 0:
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("Export as CSV"):
                        # Get activity data for export
                        cursor.execute('''
                        SELECT date, activity_type, start_time, end_time, duration, steps, calories
                        FROM activities
                        WHERE user_id = ? AND date >= ? AND date <= ?
                        ORDER BY date, start_time
                        ''', (user_profile['id'], start_date.isoformat(), end_date.isoformat()))
                        
                        activities = cursor.fetchall()
                        
                        # Convert to DataFrame
                        activities_df = pd.DataFrame(activities, 
                                                  columns=['date', 'activity_type', 'start_time', 
                                                         'end_time', 'duration', 'steps', 'calories'])
                        
                        # Convert to CSV
                        csv = activities_df.to_csv(index=False)
                        
                        # Create download button
                        st.download_button(
                            label="Download CSV",
                            data=csv,
                            file_name=f"fitness_data_{start_date}_to_{end_date}.csv",
                            mime="text/csv"
                        )
                
                with col2:
                    if st.button("Export as JSON"):
                        # Get activity data for export
                        cursor.execute('''
                        SELECT date, activity_type, start_time, end_time, duration, steps, calories
                        FROM activities
                        WHERE user_id = ? AND date >= ? AND date <= ?
                        ORDER BY date, start_time
                        ''', (user_profile['id'], start_date.isoformat(), end_date.isoformat()))
                        
                        activities = cursor.fetchall()
                        
                        # Convert to list of dictionaries
                        activities_list = []
                        for activity in activities:
                            activities_list.append({
                                'date': activity[0],
                                'activity_type': activity[1],
                                'start_time': activity[2],
                                'end_time': activity[3],
                                'duration': activity[4],
                                'steps': activity[5],
                                'calories': activity[6]
                            })
                        
                        # Convert to JSON string
                        import json
                        json_str = json.dumps(activities_list, indent=2)
                        
                        # Create download button
                        st.download_button(
                            label="Download JSON",
                            data=json_str,
                            file_name=f"fitness_data_{start_date}_to_{end_date}.json",
                            mime="application/json"
                        )
            else:
                st.info("No data available for export in the selected date range.")
            
            # Clear data section
            st.markdown("#### Clear Data")
            
            if st.button("🗑️ Clear All Activity Data"):
                confirmation = st.checkbox("I understand this will delete all my activity records", key="confirm_delete")
                
                if confirmation:
                    cursor = conn.cursor()
                    cursor.execute("DELETE FROM activities WHERE user_id = ?", (user_profile['id'],))
                    cursor.execute("DELETE FROM daily_summaries WHERE user_id = ?", (user_profile['id'],))
                    conn.commit()
                    st.success("All activity data has been cleared.")
        
        with tab3:  # App Settings
            st.markdown("### App Settings")
            
            # Theme selection (simulated)
            st.markdown("#### Theme")
            theme = st.selectbox("Select theme:", ["Light", "Dark", "Blue", "Green"])
            st.info(f"{theme} theme selected. Theme changes are simulated in this demo.")
            
            # Notification settings (simulated)
            st.markdown("#### Notifications")
            
            notify_goal = st.checkbox("Notify when daily goal is reached", value=True)
            notify_inactivity = st.checkbox("Notify after periods of inactivity", value=True)
            
            if notify_inactivity:
                inactivity_period = st.slider("Inactivity reminder after (minutes):", 30, 120, 60, 15)
                st.info(f"You'll be reminded to move after {inactivity_period} minutes of inactivity.")
            
            # Units settings (simulated)
            st.markdown("#### Units")
            
            units_system = st.radio("Measurement system:", ["Metric (kg, cm)", "Imperial (lb, in)"])
            
            if units_system == "Imperial (lb, in)":
                st.info("Note: Imperial units are simulated in this demo. Weight and height will still be stored in metric units.")
            
            # Save settings button (simulated)
            if st.button("Save Settings"):
                st.success("Settings saved successfully!")
                
                # Settings details (simulated)
                st.markdown(f"""
                **Current Settings:**
                - Theme: {theme}
                - Goal Notifications: {"On" if notify_goal else "Off"}
                - Inactivity Reminders: {"On" if notify_inactivity else "Off"}
                  {f"  - Reminder after: {inactivity_period} minutes" if notify_inactivity else ""}
                - Units: {units_system}
                """)
    
    # Close database connection when app is done
    conn.close()

if __name__ == "__main__":
    main()
