import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

df = pd.read_csv('data/t20_wc_matches.csv')
df.dropna(inplace=True)
df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]

# Create visuals folder if not exists
os.makedirs("visuals", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# Top run scorers
top_batsmen = df.groupby('batsman')["batsman_runs"].sum().reset_index().sort_values(by="batsman_runs", ascending=False).head(10)
plt.figure(figsize=(10, 6))
sns.barplot(x="batsman_runs", y="batsman", data=top_batsmen)
plt.title("Top 10 Run Scorers")
plt.savefig("visuals/top_run_scorers.png")
plt.close()

# Top bowlers (least runs conceded)
top_bowlers = df.groupby('bowler')["total_runs"].sum().reset_index().sort_values(by="total_runs").head(10)
plt.figure(figsize=(10, 6))
sns.barplot(x="total_runs", y="bowler", data=top_bowlers)
plt.title("Top 10 Economical Bowlers")
plt.savefig("visuals/top_bowlers.png")
plt.close()

# Toss win correlation
toss_wins = df[df["toss_winner"] == df["match_winner"]]
correlation = len(toss_wins) / len(df) * 100

# Venue average scores
venue_avg = df.groupby("venue")["total_runs"].mean().sort_values(ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x=venue_avg.values, y=venue_avg.index)
plt.title("Venue-wise Average Runs")
plt.xlabel("Average Runs")
plt.savefig("visuals/venue_avg_scores.png")
plt.close()

# Write report
with open("reports/summary.txt", "w") as f:
    f.write(f"Total Matches Analyzed: {len(df)}\n")
    f.write(f"Top Run Scorer: {top_batsmen.iloc[0]['batsman']} ({top_batsmen.iloc[0]['batsman_runs']} runs)\n")
    f.write(f"Best Bowler (Least Runs): {top_bowlers.iloc[0]['bowler']} ({top_bowlers.iloc[0]['total_runs']} runs)\n")
    f.write(f"Toss-Win Correlation: {correlation:.2f}%\n")
