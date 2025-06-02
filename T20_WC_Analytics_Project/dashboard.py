import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

df = pd.read_csv('data/t20_wc_matches.csv')
df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]

st.sidebar.title("🏏 T20 World Cup Dashboard")
option = st.sidebar.selectbox("Select Analysis", [
    "Top Run Scorers",
    "Top Economical Bowlers",
    "Toss vs Match Win",
    "Venue Average Runs",
    "Predict Match Outcome"
])

st.title("T20 World Cup Data Analytics")

if option == "Top Run Scorers":
    st.subheader("Top 10 Run Scorers")
    top_batsmen = df.groupby('batsman')["batsman_runs"].sum().reset_index()
    top_batsmen = top_batsmen.sort_values(by="batsman_runs", ascending=False).head(10)
    fig, ax = plt.subplots()
    sns.barplot(x="batsman_runs", y="batsman", data=top_batsmen, ax=ax)
    st.pyplot(fig)

elif option == "Top Economical Bowlers":
    st.subheader("Top 10 Bowlers (Least Runs Conceded)")
    top_bowlers = df.groupby('bowler')["total_runs"].sum().reset_index()
    top_bowlers = top_bowlers.sort_values(by="total_runs").head(10)
    fig, ax = plt.subplots()
    sns.barplot(x="total_runs", y="bowler", data=top_bowlers, ax=ax)
    st.pyplot(fig)

elif option == "Toss vs Match Win":
    st.subheader("Toss Winner = Match Winner Analysis")
    toss_wins = df[df["toss_winner"] == df["match_winner"]]
    ratio = len(toss_wins) / len(df)
    st.metric("Toss-Match Win Correlation", f"{ratio:.2%}")
    st.write(f"Out of {len(df)} matches, {len(toss_wins)} had the same toss and match winner.")

elif option == "Venue Average Runs":
    st.subheader("Average Total Runs by Venue")
    venue_scores = df.groupby("venue")["total_runs"].mean().sort_values(ascending=False)
    st.bar_chart(venue_scores)

elif option == "Predict Match Outcome":
    st.subheader("Predict Match Outcome Based on Toss")
    model = joblib.load("predictor_model.pkl")
    team_map = joblib.load("team_map.pkl")
    venue_list = joblib.load("venue_list.pkl")

    selected_team = st.selectbox("Toss Winner Team", list(team_map.keys()))
    selected_venue = st.selectbox("Venue", venue_list)

    team_code = team_map[selected_team]
    venue_code = venue_list.index(selected_venue)

    if st.button("Predict"):
        result = model.predict([[team_code, venue_code]])
        if result[0] == 1:
            st.success("✅ Toss winner is likely to win the match!")
        else:
            st.warning("❌ Toss winner may not win the match.")
