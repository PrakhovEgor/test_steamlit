import pickle


def model_prediction(df):
    with open('data/xgb_grid.pkl', 'rb') as f:
        model = pickle.load(f)
        return model.predict(df)