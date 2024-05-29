#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  5 19:36:55 2024

@author: felipezenteno
"""
import streamlit as st
import pandas as pd
import time
import plotly.express as px
import plotly.graph_objects as go
import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import lxml
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots



nba_teams = {
    "Atlanta Hawks": "ATL",
    "Boston Celtics": "BOS",
    "Brooklyn Nets": "BRK",
    "Charlotte Hornets": "CHO",
    "Chicago Bulls": "CHI",
    "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL",
    "Denver Nuggets": "DEN",
    "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW",
    "Houston Rockets": "HOU",
    "Indiana Pacers": "IND",
    "LA Clippers": "LAC",
    "Los Angeles Lakers": "LAL",
    "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA",
    "Milwaukee Bucks": "MIL",
    "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP",
    "New York Knicks": "NYK",
    "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL",
    "Philadelphia 76ers": "PHI",
    "Phoenix Suns": "PHO",
    "Portland Trail Blazers": "POR",
    "Sacramento Kings": "SAC",
    "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR",
    "Utah Jazz": "UTA",
    "Washington Wizards": "WAS"
}

nba_stats = {
    'G': 'Games Played',
    'GS': 'Games Started',
    'MP': 'Minutes Per Game',
    'FG': 'Field Goals Made',
    'FGA': 'Field Goals Attempted',
    'FG%': 'Field Goal Percentage',
    '3P': 'Three-Point Field Goals Made',
    '3PA': 'Three-Point Field Goals Attempted',
    '3P%': 'Three-Point Field Goal Percentage',
    '2P': 'Two-Point Field Goals Made',
    '2PA': 'Two-Point Field Goals Attempted',
    '2P%': 'Two-Point Field Goal Percentage',
    'eFG%': 'Effective Field Goal Percentage',
    'FT': 'Free Throws Made',
    'FTA': 'Free Throws Attempted',
    'FT%': 'Free Throw Percentage',
    'ORB': 'Offensive Rebounds',
    'DRB': 'Defensive Rebounds',
    'TRB': 'Total Rebounds',
    'AST': 'Assists',
    'STL': 'Steals',
    'BLK': 'Blocks',
    'TOV': 'Turnovers',
    'PF': 'Personal Fouls',
    'PTS': 'Points'
}
data_shot = pd.read_csv('NBA_Shots_24.csv')

# Functions
@st.cache_data
def load_data_new(year):
    player_df = pd.DataFrame()
    for i in range(10):
        url = "https://www.basketball-reference.com/leagues/NBA_" + str(year-i) + "_per_game.html"
        html = pd.read_html(url, header=0)
        df = html[0]
        raw = df.drop(df[df.Age == 'Age'].index)
        raw = raw.fillna(0)
        playerstats = raw.drop(['Rk'], axis=1)
        playerstats['Year'] = year-i
        except_columns = ['Player', 'Pos', 'Tm']
        columns_to_convert = [col for col in playerstats.columns if col not in except_columns]
        except_columns_int  = ['Year','Age', 'G', 'GS']
        columns_to_convert_float = [col for col in columns_to_convert if col not in except_columns_int]
        # DF - Format
        playerstats[columns_to_convert] = round(playerstats[columns_to_convert].astype(float), 1)
        playerstats[except_columns_int] = playerstats[except_columns_int].astype(int)
        playerstats[columns_to_convert_float] = playerstats[columns_to_convert_float].applymap('{:.1f}'.format)
        playerstats.drop_duplicates(subset=['Player'], keep='first', inplace=True)
        playerstats.reset_index(drop=True, inplace=True)
        # player_df = player_df.append(playerstats, ignore_index=False)
        player_df = pd.concat([player_df ,playerstats], ignore_index=True)

        time.sleep(1)

    columns = list(player_df.columns)
    columns.remove('Year')
    player_df = player_df[['Year'] + columns]
    return player_df


def plot_sorted(df, selected_player, column, title_plot):
    df_sorted = df.sort_values(by=column)
    # Create scatter plot
    fig = go.Figure()
    for index, row in df_sorted.iterrows():
        if row['Player'] == selected_player:
            fig.add_trace(go.Scatter(x=[row[column]], y=[row['Player']], mode='markers', marker=dict(color='red'), showlegend=True))
        else:
            fig.add_trace(go.Scatter(x=[row[column]], y=[row['Player']], mode='markers', marker=dict(color='blue'), showlegend=False))
    # Customize layout
    fig.update_layout(
        title=title_plot,
        xaxis=dict(title=column),
        yaxis=dict(title='Players'),
        hovermode='closest')
    st.plotly_chart(fig)
    
    
def draw_court(ax=None, color='black', lw=2):
        if ax is None:
            ax = plt.gca()

        # Create the basketball hoop
        hoop = patches.Circle((0, 0), radius=7.5, linewidth=lw, color=color, fill=False)

        # Create backboard
        backboard = plt.Line2D([-30, 30], [-7.5, -7.5], linewidth=lw, color=color)

        # The paint
        outer_box = patches.Rectangle((-80, -47.5), 160, 190, linewidth=lw, color=color, fill=False)
        inner_box = patches.Rectangle((-60, -47.5), 120, 190, linewidth=lw, color=color, fill=False)

        # Free throw top arc
        top_free_throw = patches.Arc((0, 142.5), 120, 120, theta1=0, theta2=180, linewidth=lw, color=color)
        # Free throw bottom arc
        bottom_free_throw = patches.Arc((0, 142.5), 120, 120, theta1=180, theta2=0, linewidth=lw, color=color, linestyle='dashed')

        # Restricted area
        restricted = patches.Arc((0, 0), 80, 80, theta1=0, theta2=180, linewidth=lw, color=color)

        # Three point line
        corner_three_a = plt.Line2D([-220, -220], [-47.5, 92.5], linewidth=lw, color=color)
        corner_three_b = plt.Line2D([220, 220], [-47.5, 92.5], linewidth=lw, color=color)
        three_arc = patches.Arc((0, 0), 475, 475, theta1=22, theta2=158, linewidth=lw, color=color)

        # Center Court
        center_outer_arc = patches.Arc((0, 422.5), 120, 120, theta1=180, theta2=0, linewidth=lw, color=color)
        center_inner_arc = patches.Arc((0, 422.5), 40, 40, theta1=180, theta2=0, linewidth=lw, color=color)

        court_elements = [hoop, outer_box, inner_box, top_free_throw, bottom_free_throw,
                          restricted, three_arc, center_outer_arc, center_inner_arc]

        for element in court_elements:
            ax.add_patch(element)

        # Add lines separately
        ax.add_line(backboard)
        ax.add_line(corner_three_a)
        ax.add_line(corner_three_b)

        return ax




def get_MVP_prediction(historical_data):
# DataFrame containing the last 11 MVP winners
    mvp_winners_data = {
        "Year": [2023, 2022, 2021, 2020, 2019, 2018, 2017, 2016, 2015, 2014, 2013],
        "Player": ["Giannis Antetokounmpo", "Nikola Jokić", "Nikola Jokić", "Giannis Antetokounmpo", 
                   "Giannis Antetokounmpo", "James Harden", "Russell Westbrook", "Stephen Curry", 
                   "Stephen Curry", "Kevin Durant", "LeBron James"],
        "MVP_Winner": [1]*11  # Adding a column with value 1 for all entries
    }
    mvp_winners_df = pd.DataFrame(mvp_winners_data)
    
    # Merge the existing DataFrame with the MVP winners DataFrame
    merged_df = pd.merge(historical_data, mvp_winners_df, on=["Year", "Player"], how="left")
    
    # Fill NaN values in the "MVP_Winner" column with 0
    merged_df["MVP_Winner"].fillna(0, inplace=True)
    
    # Convert the "MVP_Winner" column to integer type
    merged_df["MVP_Winner"] = merged_df["MVP_Winner"].astype(int)
    
    #merged_df.rename(columns={"MVP_Winner_y": "MVP_Winner"}, inplace=True)
    
    historical_data = merged_df
    historical_data = historical_data.fillna(0)
    
    # Select relevant features
    features = ["G", "GS", "MP", "FG%", "3P%", "2P%", "eFG%", "FT%", "ORB", "DRB", "TRB", "AST", "STL", "BLK", "TOV", "PF", "PTS", "TS%"]
    
    # Define target variable
    target = "MVP_Winner"  # Assuming this column indicates whether the player won MVP (1) or not (0)
    
    # Split data into features and target variable
    X = historical_data[features]
    y = historical_data[target]
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train Random Forest classifier
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    
    # Predict MVP winners
    predictions = clf.predict(X_test)
    
    # Evaluate model accuracy
    accuracy = accuracy_score(y_test, predictions)
    #print("Model Accuracy:", accuracy)
    
    # Predict the next MVP using current season statistics
    
    current_season_stats = historical_data[(historical_data.Year == 2024)]
    current_season_stats['MVP_Winner'] = 0
    current_season_stats = current_season_stats.fillna(0)
    
    next_mvp_prediction = clf.predict(current_season_stats[features])
    
    # Get feature importances
    feature_importances = clf.feature_importances_
    
    # Create a DataFrame to display feature importances
    feature_importance_df = pd.DataFrame({"Feature": features, "Importance": feature_importances})
    
    # Sort features by importance
    feature_importance_df = feature_importance_df.sort_values(by="Importance", ascending=False)
    
    for stat in feature_importance_df.Feature:
        x = current_season_stats[stat].astype(float)*float(feature_importance_df[feature_importance_df.Feature == stat].Importance)
        current_season_stats['MVP_Winner'] = x + current_season_stats['MVP_Winner']
    
    results = current_season_stats[['Player', 'MVP_Winner']]
    results = results.sort_values(by='MVP_Winner', ascending=False)
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    results.reset_index(drop=True, inplace=True)
    feature_importance_df.reset_index(drop=True, inplace=True)
    feature_importance_df.rename(columns={"Feature": "Stat","Importance": "Importance_%"}, inplace=True)
    feature_importance_df['Importance_%'] = feature_importance_df['Importance_%']*100
    return results, feature_importance_df

#Plot MVP
def create_pie_MVP(Stats_df):
    players_df, stats_df = get_MVP_prediction(Stats_df)

# Create subplots: use 'domain' type for Pie subplot
    fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])
    fig.add_trace(go.Pie(players_df),
                  1, 1)
    fig.add_trace(go.Pie(stats_df),
                  1, 2)
    
    # Use `hole` to create a donut-like pie chart
    fig.update_traces(hole=.4, hoverinfo="label+percent+name")
    
    fig.update_layout(
        title_text="MVP Prediction",
        # Add annotations in the center of the donut pies.
        annotations=[dict(text='Players', x=0.18, y=0.5, font_size=20, showarrow=False),
                     dict(text='Stats', x=0.82, y=0.5, font_size=20, showarrow=False)])
    fig.show()

# Main
def main():
    st.set_page_config(layout="wide")
    st.title('NBA Stats App')
    st.markdown(""" Welcome to the NBA stats app! Explore team and player statistics effortlessly.\n\n Powered by Python, Streamlit, Pandas, and visualization tools like Matplotlib and Plotly.\n\n Our data is sourced from Basketball-reference.com.""")
    selected_year = datetime.datetime.now().year
    playerstats = load_data_new(selected_year)
    #               Calculate new Stats
    playerstats['eFG%'] = (playerstats['FG'].astype(float) + 0.5*playerstats['3P'].astype(float))/playerstats['FGA'].astype(float)
    playerstats['TS%'] = 0.5*playerstats['PTS'].astype(float)/(playerstats['FGA'].astype(float)+(0.475*playerstats['FTA'].astype(float)))

    # Sidebar - Team selection
    st.sidebar.header('Select a team')
    selected_team_dic = st.sidebar.selectbox('Team', list(nba_teams.keys()))
    selected_team = nba_teams[selected_team_dic]
    # Filtering data
    df_selected_team = playerstats[(playerstats.Tm == (selected_team))]
    teamstats_year = playerstats[(playerstats.Year == selected_year) & (playerstats.Tm == selected_team)]
    #                       PLAYER SELECTER
    st.sidebar.header('Select a Player')
    players_list = df_selected_team[df_selected_team['Year'] == selected_year].Player.unique()
    selected_player = st.sidebar.selectbox('Player', players_list)
    df_selected_player = playerstats[playerstats['Player'] == selected_player]
    st.sidebar.header('Compare Player stats')
    compare_selected = st.sidebar.radio(f'Compare **{selected_player}** with:', ['Team', 'Position', 'NBA'])
    if compare_selected == 'NBA':
        df_selected_compare = playerstats[(playerstats.Year == selected_year)]
        compare_title = 'Comparing with the NBA'

    if compare_selected == 'Team':
        df_selected_compare = teamstats_year
        compare_title = f'Comparing with the {selected_team_dic}'

    if compare_selected == 'Position':
        df_selected_compare = playerstats[(playerstats.Pos == df_selected_player.iloc[0]['Pos']) & (playerstats.Year == selected_year)]
        compare_title = f'Comparing with every {df_selected_player.Pos.iloc[0]} in the NBA'

    #df_selected_compare['eFG%'] = (df_selected_compare['FG'].astype(float) + 0.5*df_selected_compare['3P'].astype(float))/df_selected_compare['FGA'].astype(float)
    #df_selected_compare['TS%'] = 0.5*df_selected_compare['PTS'].astype(float)/(df_selected_compare['FGA'].astype(float)+(0.475*df_selected_compare['FTA'].astype(float)))

    #               Stats Abreviation Meaning
    st.sidebar.header('Stats Abreviation Meaning')
    selected_stat_dic = st.sidebar.selectbox('Stat', list(nba_stats.keys()))
    selected_stat = nba_stats[selected_stat_dic]
    st.sidebar.write(selected_stat)
    #                           TEAM STATS
    teamstats_year = teamstats_year.sort_values(by='Year', ascending=True)
    teamstats_year['Year'] = teamstats_year['Year'].astype(str)
    with st.expander(f' **{selected_team_dic}** Stats'):
        st.markdown(f"""
             A DataFrame with the stats of the {selected_team_dic}
        """)
        with st.spinner('Loading team stats'):
            teamstats_year.reset_index(drop=True, inplace=True)
            st.dataframe(teamstats_year)
            st.write('Data Dimension: ' + str(teamstats_year.shape[0]) + ' rows and ' + str(teamstats_year.shape[1]) + ' columns.')

    df_selected_player = df_selected_player.sort_values(by='Year', ascending=False)
    df_selected_player['Year'] = df_selected_player['Year'].astype(str)
    stats_float = ['PTS', 'FGA', 'FG', '3P', 'FTA']
    for stat in stats_float:
            df_selected_player[stat] = df_selected_player[stat].astype(float)
    with st.expander(f' **{selected_player}** Stats'):
        st.markdown(f"""
             A DataFrame with the stats of the {selected_player} from the last {df_selected_player.shape[0]} years
        """)
        with st.spinner(f'Loading {selected_player} stats'):
            data = data_shot[data_shot.PLAYER_NAME == selected_player]
            teamstats_year.reset_index(drop=True, inplace=True)
            st.dataframe(df_selected_player)
            st.write('Data Dimension: ' + str(df_selected_player.shape[0]) + ' rows and ' + str(df_selected_player.shape[1]) + ' columns.')
            stats_to_plot = st.multiselect("Stats to Plot",
                                           df_selected_player.columns[5:],
                                           ["G", "GS", "PTS", 'FGA'])
            for i in stats_to_plot:
                df_selected_player[i] = df_selected_player[i].astype(float)
            # Plot Chart
            st.plotly_chart(px.line(df_selected_player,
                                    x="Year",
                                    y=stats_to_plot), use_container_width=True)
            # Compare Plot
            col1, col2, col3 = st.columns(3)
            with col1:
                selected_stat = st.selectbox('Stat to Compare', df_selected_player.columns[5:])
            df_selected_player[selected_stat] = df_selected_player[selected_stat].astype(float)
            st.write(plot_sorted(df_selected_compare, selected_player, selected_stat, f'{selected_stat} - {compare_title} in 2024'))
            st.header('Shot Chart')
            fig, ax = plt.subplots(figsize=(8, 7))
            ax.set_facecolor('black')
            # Draw the court
            draw_court(ax, color='white')  # Court elements in white
            # Create hexbin plot
            hb = ax.hexbin(data['LOC_X'], data['LOC_Y'], gridsize=50, cmap='coolwarm', mincnt=1)
            # Add color bar
            cb = fig.colorbar(hb, ax=ax)
            cb.set_label('Shot Frequency')
            # Remove axis labels and ticks
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            # Labels and title
            ax.set_title(f'{selected_player} Shot Chart')
            # Set the limits to half court
            ax.set_xlim(-250, 250)
            ax.set_ylim(422.5, -47.5)
            # Display the plot in Streamlit
            st.pyplot(fig)
            
            

    df_selected_player = df_selected_player.sort_values(by='Year', ascending=True)
    df_selected_player['Year'] = df_selected_player['Year'].astype(str)

    st.header('Scoring Efficiency Metrics')                                            
    with st.expander('True Shooting Percentage - TS%'):
        st.markdown("""
             The true shooting percentage is the shooting percentage adjusted for three-pointers and free throws, and measures a player's efficiency at shooting the ball.
        """)
        formula = r'\text{TS\%} = \frac{0.5 \times \text{PTS}}{\text{FGA} + 0.475 \times \text{FTA}}'

        st.latex(formula)
        with st.spinner('Loading scatter plot'):
            st.write(plot_sorted(df_selected_compare, selected_player, 'TS%', f'True Shooting Percentage - {compare_title} in 2024'))

    with st.expander('Effective Field Goal Percentage - eFG%'):
        st.markdown("""
             The Effective Field Goal Percentage is a statistic that adjusts field goal percentage to account for the fact that three-point field goals count for three points, while all other field goals only count for two points.
        """)
        formula = r'\text{eFG\%} = \frac{\text{FG} + (0.5 \times \text{3P})}{\text{FGA}}'

        st.latex(formula)
        with st.spinner('Loading scatter plot'):
            st.write(plot_sorted(df_selected_compare, selected_player, 'eFG%', f'Effective Field Goal Percentage - {compare_title} in 2024'))


    st.header('Most Valuable Player Prediction')
    with st.expander('**MVP** Prediction'):
        st.markdown(""" This MVP predictor tool utilizes a Random Forest classifier to forecast the MVP winner. It merges MVP winners' data, meticulously handles missing values, and strategically selects relevant features, assigning higher importance to certain statistics in the model. After splitting and training the data, it evaluates the model's accuracy. Lastly, it leverages current season statistics to predict the next MVP.
        """)

        with st.spinner('Loading MVP Prediction'):
            col1, col2, col3 = st.columns([20,1,20])
            st.write(create_pie_MVP(playerstats))
            with col1:
                st.dataframe(get_MVP_prediction(playerstats)[0])
            with col3:
                st.dataframe(get_MVP_prediction(playerstats)[1])
                    
    st.markdown("##")
    st.markdown("##")
    st.write("© Copyright 2024 Felipe Zenteno  All rights reserved.")

if __name__ == "__main__":
    main()
