import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

gdf = pd.read_csv('RahulGrades.csv')
tdf = pd.read_csv('TextToxicity.csv')[['timestamp', 'Preds']]

gdf['timestamp'] = [datetime(gdf.loc[i, 'Year'], gdf.loc[i, 'Month'], gdf.loc[i, 'Day']) for i in range(len(gdf))]
gdf = gdf.sort_values('timestamp')[['timestamp', 'Grade (Decimal)']].reset_index(drop=True)

tdf['timestamp'] = [datetime.strptime(tdf.loc[i, 'timestamp'], '%Y-%m-%d %H:%M:%S').date() for i in range(len(tdf))]
end = gdf.iloc[-1]['timestamp'].date()
tdf = tdf[(tdf.timestamp <= end)]
tdf = tdf.groupby('timestamp')['Preds'].mean().reset_index()

scaler = MinMaxScaler()
tdf['Preds'] = scaler.fit_transform(pd.DataFrame(tdf['Preds']))

plt.plot(gdf['timestamp'], gdf['Grade (Decimal)'])
plt.plot(tdf['timestamp'], tdf['Preds'])
plt.legend(['Chemistry Grades', 'Text Toxicity'])
plt.title("Rahul's Chemistry Grades and Text Message Toxicity vs. Time")
plt.xlabel('Date')
plt.ylabel('Normalized Score (0-1)')

