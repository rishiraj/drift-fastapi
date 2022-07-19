from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from category_encoders import OrdinalEncoder
from lightgbm import LGBMRegressor
from eurybia import SmartDrift
from sklearn.model_selection import train_test_split

from eurybia.data.data_loader import data_loading
house_df, house_dict = data_loading('house_prices')

# Let us consider that the column "YrSold" corresponds to the reference date. 
#In 2006, a model was trained using data. And in 2007, we want to detect data drift on new data in production to predict
#house price
house_df_learning = house_df.loc[house_df['YrSold'] == 2006]
house_df_2007 = house_df.loc[house_df['YrSold'] == 2007]

y_df_learning=house_df_learning['SalePrice'].to_frame()
X_df_learning=house_df_learning[house_df_learning.columns.difference(['SalePrice','YrSold'])]

y_df_2007=house_df_2007['SalePrice'].to_frame()
X_df_2007=house_df_2007[house_df_2007.columns.difference(['SalePrice','YrSold'])]

from category_encoders import OrdinalEncoder

categorical_features = [col for col in X_df_learning.columns if X_df_learning[col].dtype == 'object']

encoder = OrdinalEncoder(
    cols=categorical_features,
    handle_unknown='ignore',
    return_df=True).fit(X_df_learning)

X_df_learning_encoded=encoder.transform(X_df_learning)

Xtrain, Xtest, ytrain, ytest = train_test_split(X_df_learning_encoded, y_df_learning, train_size=0.6, random_state=1)

regressor = LGBMRegressor(n_estimators=200).fit(Xtrain,ytrain)

from eurybia import SmartDrift

SD = SmartDrift(df_current=X_df_2007,
                df_baseline=X_df_learning,
                deployed_model=regressor, # Optional: put in perspective result with importance on deployed model
                encoding=encoder # Optional: if deployed_model and encoder to use this model
               )

SD.compile()

SD.generate_report(    
    output_file='/home/rishiraj/drift/static/index.html',    
    title_story="Data drift",
    title_description="""House price Data drift 2007""", # Optional: add a subtitle to describe report
    )

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
