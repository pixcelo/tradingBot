# %%
import pandas as pd
import numpy as np
import plotly.express as px
# %%
df = pd.read_pickle('../fetch/df_ohlcv.pkl')
df
# %%
df['sma_200'] = df['op'].rolling(200).mean()
df.reset_index(inplace=True)
df
# %%
fig = px.line(df, x='timestamp', y=['op', 'sma_200'], title='BTC')
fig.show()
# %%
# 移動平均線のクロス
def sma_cross(price, prev_price, sma):
    return price > sma and prev_price < sma
    
df['prev_open'] = df['op'].shift(1)
df.dropna(inplace=True)

# np.vectorizeは、Pythonの関数を配列に適用できるように変換する関数
df['sma_cross'] = np.vectorize(sma_cross)(df['op'], df['prev_open'], df['sma_200'])

sma_crosses = df[df.sma_cross == True]
sma_crosses

# %%
df.info()
# %%
# run backtest
class Position:
    def __init__(self, open_time, open_price, volume) -> None:
        self.open_time = open_time
        self.open_price = open_price
        self.volume = volume
        self.status = 'open'

        self.close_time = None
        self.close_price = None
        self.profit = None
    
    def as_dict(self):
        if self.close_price is not None:
            self.profit = (self.close_price - self.open_price) * self.volume / self.open_price

        return {'open_time': self.open_time,
                'open_price': self.open_price,
                'volume': self.volume,
                'close_time': self.close_time,
                'close_price': self.close_price,
                'profit': self.profit,
                'status': self.status
                }

positions = []

for i, data in df.iterrows():
    # entry strategy
    if data['sma_cross'] is True:
        # 1ユニットを購入するとする
        positions.append(Position(data['timestamp'], data['op'], 1))

    # exit strategy
    if data['op'] < data['sma_200']:
        for pos in positions:
            if pos.status == 'open':
                pos.close_time = data['timestamp']
                pos.close_price = data['op']
                pos.status = 'closed'

pos_df = pd.DataFrame([p.as_dict() for p in positions])
pos_df['cumulative_profit'] = pos_df['profit'].cumsum() # cumulative 累計
pos_df
# %%
fig_pos = px.line(pos_df, x='close_time', y='cumulative_profit')
fig_pos.show()
# %%
