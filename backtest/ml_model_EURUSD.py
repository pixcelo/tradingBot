# %%
from backtesting.test import EURUSD, SMA

data = EURUSD.copy()
data

# 教師あり機械学習(supervised machine learning)では、入力された特徴ベクトル（独立変数）を既知の出力値（従属変数）に写像する関数を学習しようとする
# f : X → y

# モデル関数が十分であれば、新たに獲得した入力特徴ベクトルから将来の出力値をある程度確実に予測することができる
# この例では、いくつかの価格由来の特徴量と一般的なテクニカル指標を、2日後の価格時点に対応させる
# モデル設計行列を構築する X を以下に示す

# %%
def BBANDS(data, n_lookback, n_std):
    """Bollinger bands indicator"""
    hlc3 = (data.High + data.Low + data.Close) / 3
    mean, std = hlc3.rolling(n_lookback).mean(), hlc3.rolling(n_lookback).std()
    upper = mean + n_std*std
    lower = mean - n_std*std
    return upper, lower


close = data.Close.values
sma10 = SMA(data.Close, 10)
sma20 = SMA(data.Close, 20)
sma50 = SMA(data.Close, 50)
sma100 = SMA(data.Close, 100)
upper, lower = BBANDS(data, 20, 2)

# Design matrix / independent features:

# Price-derived features
data['X_SMA10'] = (close - sma10) / close
data['X_SMA20'] = (close - sma20) / close
data['X_SMA50'] = (close - sma50) / close
data['X_SMA100'] = (close - sma100) / close

data['X_DELTA_SMA10'] = (sma10 - sma20) / close
data['X_DELTA_SMA20'] = (sma20 - sma50) / close
data['X_DELTA_SMA50'] = (sma50 - sma100) / close

# Indicator features
data['X_MOM'] = data.Close.pct_change(periods=2)
data['X_BB_upper'] = (upper - close) / close
data['X_BB_lower'] = (lower - close) / close
data['X_BB_width'] = (upper - lower) / close
data['X_Sentiment'] = ~data.index.to_series().between('2017-09-27', '2017-12-14')

# Some datetime features for good measure
data['X_day'] = data.index.dayofweek
data['X_hour'] = data.index.hour

data = data.dropna().astype(float)
# %%

# 指標はすべて過去の値に対してのみ作用するので、デザイン行列を事前に計算することは安全である。あるいは、モデルを学習する前に毎回行列を再構築する
# 作り物のセンチメント特徴は、現実には、ニュースソース、Twitterのセンチメント、Stocktwitsなどを解析することで同様の特徴を得ることができる
# これは、入力データがあらゆる種類の追加説明カラムを含むことができることを示すだけである

# 従属変数は2日後の価格（リターン）とし、値を単純化する
# 1 が正の値（かつ有意）であるとき。- 1 負の場合は-1、0 は2日後のリターンがほぼゼロである場合である
# X と従属変数であるクラス変数 y をNumPyの配列として返す関数を書く

# %%
import numpy as np

def get_X(data):
    """Return model design matrix X"""
    return data.filter(like='X').values

def get_y(data):
    """Return dependent variable y"""
    y = data.Close.pct_change(48).shift(-48)  # Returns after roughly two days
    y[y.between(-.004, .004)] = 0             # Devalue returns smaller than 0.4%
    y[y > 0] = 1
    y[y < 0] = -1
    return y

def get_clean_Xy(df):
    """Return (X, y) cleaned of NaN values"""
    X = get_X(df)
    y = get_y(df).values
    isnan = np.isnan(y)
    X = X[~isnan]
    y = y[~isnan]
    return X, y

# scikit-learnのシンプルなk-nearest neighbors（kNN）アルゴリズムを使って、どのようにモデル化されるかを見る
# オーバーフィッティングを避ける（あるいは少なくとも実証する）ために、常にデータを訓練セットとテストセットに分割する
# 特に、モデルの性能を構築されたデータと同じデータで検証しないようにする

# %%
import pandas as pd
import pandas.plotting._matplotlib
from sklearn.neighbors import KNeighborsClassifier # K−近傍法
from sklearn.model_selection import train_test_split

X, y = get_clean_Xy(data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5, random_state=0)

clf = KNeighborsClassifier(7)  # Model the output based on 7 "nearest" examples
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

_ = pd.DataFrame({'y_true': y_test, 'y_pred': y_pred}).plot(figsize=(15, 2), alpha=.7)
print('Classification accuracy: ', np.mean(y_test == y_pred))
# %%

# 予測がプラス（2日後の価格が上昇すると予測される）のときはいつでも、利用可能な資本の20%で資産を20:1のレバレッジで購入し、
# 予測がマイナスのときは同じ条件で売却し、妥当な損切りと利益確定のレベルを設定するという単純な戦略をバックテストしてみる
# data.dfアクセッサが着実に使用されていることに注目する

# %%
from backtesting import Backtest, Strategy

N_TRAIN = 400


class MLTrainOnceStrategy(Strategy):
    price_delta = .004  # 0.4%

    def init(self):        
        # Init our model, a kNN classifier
        self.clf = KNeighborsClassifier(7)

        # Train the classifier in advance on the first N_TRAIN examples
        df = self.data.df.iloc[:N_TRAIN]
        X, y = get_clean_Xy(df)
        self.clf.fit(X, y)

        # Plot y for inspection
        self.I(get_y, self.data.df, name='y_true')

        # Prepare empty, all-NaN forecast indicator
        self.forecasts = self.I(lambda: np.repeat(np.nan, len(self.data)), name='forecast')

    def next(self):
        # Skip the training, in-sample data
        if len(self.data) < N_TRAIN:
            return

        # Proceed only with out-of-sample data. Prepare some variables
        high, low, close = self.data.High, self.data.Low, self.data.Close
        current_time = self.data.index[-1]

        # Forecast the next movement
        X = get_X(self.data.df.iloc[-1:])
        forecast = self.clf.predict(X)[0]

        # Update the plotted "forecast" indicator
        self.forecasts[-1] = forecast

        # 予想が上向きで、まだロングポジションを保有していない場合
        # 利用可能な口座資本の20%分のロングオーダーを出す、ショートの場合はその逆
        # また, 目標とする利食いと損切りの価格を, 現在の終値から1つの価格差に設定
        # 現在の終値から1つ離れた価格となるように設定
        upper, lower = close[-1] * (1 + np.r_[1, -1]*self.price_delta)

        if forecast == 1 and not self.position.is_long:
            self.buy(size=.2, tp=upper, sl=lower)
        elif forecast == -1 and not self.position.is_short:
            self.sell(size=.2, tp=lower, sl=upper)

        # さらに、2日以上開いた取引には積極的な損切りを設定すること
        # 積極的なストップロスを設定
        for trade in self.trades:
            if current_time - trade.entry_time > pd.Timedelta('2 days'):
                if trade.is_long:
                    trade.sl = max(trade.sl, low)
                else:
                    trade.sl = min(trade.sl, high)


bt = Backtest(data, MLTrainOnceStrategy, commission=.0002, margin=.05)
bt.run()
# %%
bt.plot()

# k-foldやleave-one-out cross-validationのようなwalk-forward最適化
# %%
class MLWalkForwardStrategy(MLTrainOnceStrategy):
    def next(self):
        # Skip the cold start period with too few values available
        if len(self.data) < N_TRAIN:
            return

        # 20回反復するごとにのみモデルを再トレーニングする。
        # 20 << N_TRAIN なので、「最近の学習例」という点ではあまり損をしない
        # "最近の学習例 "という点ではあまり失われないが、速度は大幅に向上する
        if len(self.data) % 20:
            return super().next()

        # Retrain on last N_TRAIN values
        df = self.data.df[-N_TRAIN:]
        X, y = get_clean_Xy(df)
        self.clf.fit(X, y)

        # モデルのフィッティングが完了
        # MLTrainOnceStrategy と同じように進める
        super().next()


bt = Backtest(data, MLWalkForwardStrategy, commission=.0002, margin=.05)
bt.run()
# %%
bt.plot()
# %%
"""
どうやら、過去のN_TRAINデータポイントでローリング方式で繰り返し再トレーニングすると、我々の基本モデルは汎化が悪くなり、性能もそれほど良くはないようです。

これは、backtesting.pyフレームワークで機械学習の予測モデルを使用する一つの方法を示す、単純で工夫された、皮肉な例でした。
現実には、FXの短期自動売買で安定した利益を得るためには、はるかに優れた特徴空間、優れたモデル（cf. deep learning）、優れた資金管理戦略が必要です。
より適切なデータサイエンスは、熱心な読者のための練習問題です。

即座に思いつく最適化のヒントをいくつか挙げてみましょう。

データは王様です。データは王様です。デザインマトリックスの機能が、単なるランダムなノイズではなく、選択したターゲット変数（複数可）を可能な限りモデル化し、
相関することを確認してください。
単一のターゲット変数をモデル化するのではなく 
yをモデル化する代わりに、多数のターゲット/クラス変数をモデル化し、おそらく上記の「48時間リターン」よりもうまく設計します。

予測価格、出来高、「離陸」するまでの時間、SL/TP レベル、最適なポジションサイズ...など、すべてをモデル化します。

取引に入る前に、必要な確信度を高め、余分な領域の専門知識と裁量の制限を課すことで、誤検出を減らすことができます。

https://kernc.github.io/backtesting.py/doc/examples/Trading%20with%20Machine%20Learning.html
"""