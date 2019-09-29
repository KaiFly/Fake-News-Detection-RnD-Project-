# This use for predict_news.py
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
def sen_feature(data):
    vs = analyzer.polarity_scores(data)
    return vs['compound']