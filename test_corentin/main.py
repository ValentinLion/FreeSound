from processing import *
import pandas as pd


from tqdm import tqdm, tqdm_pandas

tqdm.pandas()

SAMPLE_RATE = 44100

importMFCCFromFile = True

path_train_audio = "../FreeSoundData/audio_train/"
path_test_audio = "../FreeSoundData/audio_test/"
path_csvfiles = "."

train = pd.read_csv(path_csvfiles+'/../FreeSoundData/train.csv')
submission = pd.read_csv(path_csvfiles+'/../FreeSoundData/sample_submission.csv')

if importMFCCFromFile:
    train_data = importCSV(path_csvfiles,"/test_corentin/train_mfcc.csv")
    test_data = importCSV(path_csvfiles,"/test_corentin/test_mfcc.csv")
else:
    train_data, test_data = apply_mfcc(train, path_train_audio,path_test_audio,SAMPLE_RATE)
    toCSV(train_data,"./test_corentin","/train_mfcc.csv")
    toCSV(test_data,"./test_corentin","/test_mfcc.csv")

X,y,i2c = transform_data(train_data)

preds = randomForestPredictions(X,y,test_data,i2c)
#preds = XGBPredictions(X,y,test_data,i2c)
#preds = svmPredictions(X,y,test_data,i2c)
#preds = neighborsPredictions(2,X,y,test_data,i2c)


predictionsToCSV(preds,path_test_audio,"./test_corentin","/submissions.csv")

"""
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from pprint import pprint
if  __name__ == "__main__":
    rf = RandomForestRegressor(random_state = 42)
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'bootstrap': bootstrap}
    rf = RandomForestRegressor()
    rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 1, cv = 3, verbose=0, random_state=42, n_jobs = -1)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=10, shuffle = True)
    rf_random.fit(X_train, y_train)

    pprint(rf_random.best_estimator_)

"""
