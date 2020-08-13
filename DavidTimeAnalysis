import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

gdf = pd.read_csv('RahulGrades.csv')
tdf = pd.read_csv('TextToxicity.csv')[['timestamp', 'Preds']]

gdf['timestamp'] = [datetime(gdf.loc[i, 'Year'], gdf.loc[i, 'Month'], gdf.loc[i, 'Day'], 8, 39, 0) for i in range(len(gdf))]
gdf = gdf.sort_values('timestamp')[['timestamp', 'Grade (Decimal)']].reset_index(drop=True)

tdf['timestamp'] = [datetime.strptime(tdf.loc[i, 'timestamp'], '%Y-%m-%d %H:%M:%S') for i in range(len(tdf))]
end = gdf.iloc[-1]['timestamp']
tdf = tdf[(tdf.timestamp <= end)]

plt.plot(gdf['timestamp'], gdf['Grade (Decimal)'])
plt.plot(tdf['timestamp'], tdf['Preds']/5)
plt.legend(['Grades', 'Text Toxicity'])
