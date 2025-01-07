from house_price_prediction_ann.utils import *
import scipy.stats as stats
from scipy.special import boxcox1p
from scipy.stats import normaltest
from scipy.optimize import minimize
from sklearn.preprocessing import RobustScaler
from sklearn.neighbors import LocalOutlierFactor

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

def hpp_data_prep():

    ## Dataset Reading
    df_train = load_train()
    df_test = load_test()
    df = concat_df_on_y_axis(df_train, df_test)
    df.shape


    df_copy = df.copy()
    check_df(df_copy)


    df_copy.drop(['Id'], axis = 1, inplace = True)
    cat_cols_eda, num_cols_eda, cat_but_car_eda = grab_col_names(df)


    nans = df.isna().sum().sort_values(ascending=False)
    nans = nans[nans > 0]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.grid()
    ax.bar(nans.index, nans.values, zorder=2, color="#3f72af")
    ax.set_ylabel("No. of missing values", labelpad=10)
    ax.set_xlim(-0.6, len(nans) - 0.4)
    ax.xaxis.set_tick_params(rotation=90)
    plt.show()

    '''
    for feature in cat_cols_eda:
        cat_summary(df_train, feature)
        df_train.groupby(feature)['SalePrice'].mean().plot.bar()
        plt.title(feature + ' vs Sale Price')
        plt.show()


    for col in num_cols_eda:
        num_summary(df_train, col, True)
    '''

    y1 = df_train['SalePrice']
    plt.figure(2); plt.title('Normal')
    sns.distplot(y1, kde=False, fit=stats.norm)
    plt.figure(3); plt.title('Log Normal')
    sns.distplot(y1, kde=False, fit=stats.lognorm)


    #Log Transform
    #y = np.log(df_copy["SalePrice"])

    sns.set(font_scale=1.1)
    corr_train = df_train[num_cols_eda].corr()
    mask = np.triu(corr_train.corr())
    plt.figure(figsize=(20, 20))
    sns.heatmap(corr_train, annot=True, fmt='.2f', cmap='coolwarm', square=True, mask=mask, linewidth=1, cbar=True)
    plt.show()


    ## Grabbing Columns (NUM - CAT)
    cols_with_na_meaning = ['Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 
                                'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature', 'MasVnrType']


    for col in cols_with_na_meaning:
        df_copy[col].fillna("None",inplace=True)


    encoding = {
        'None': 0,
        'Po': 1, 'No': 1, 'Unf': 1, 'Sal': 1, 'MnWw': 1,
        'Fa': 2, 'Mn': 2, 'LwQ': 2, 'Sev': 2, 'RFn': 2, 'GdWo': 2,
        'TA': 3, 'Av': 3, 'Rec': 3, 'Maj2': 3, 'Fin': 3, 'MnPrv': 3,
        'Gd': 4, 'BLQ': 4, 'Maj1': 4, 'GdPrv': 4,
        'Ex': 5, 'ALQ': 5, 'Mod': 5,
        'GLQ': 6, 'Min2': 6,
        'Min1': 7,
        'Typ': 8,
    }

    # Kodlamayı uygulayacağımız sütunlar listesi
    columns_to_encode = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 
                        'KitchenQual', 'Functional', 'FireplaceQu', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence']

    # Kodlamayı uygulama
    for column in columns_to_encode:
        df_copy[column] = df_copy[column].map(encoding)


    cat_cols, num_cols, cat_but_car = grab_col_names(df_copy)


    known_num_cols = ['BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'Fireplaces', 'GarageCars', 'OverallCond', 'YrSold']


    for col in known_num_cols:
        if num_cols.__contains__(col) == False:
            num_cols.append(col)
            cat_cols.remove(col)

    for col in columns_to_encode:
        if num_cols.__contains__(col) == False:
            num_cols.append(col)
            cat_cols.remove(col)


    #check_df(df_copy)


    num_cols_without_target = [col for col in num_cols if col not in 'SalePrice']


    df_copy[cat_cols] = df_copy[cat_cols].applymap(lambda x: x.replace(' ', '') if isinstance(x, str) else x)


    ## Suppressing Outliers of Numeric Columns
    print(len(num_cols))

    print(len(cat_cols))

    print('*------------------------*')

    for col in num_cols_without_target:
        print(col, check_outlier(df_copy, col))

    for col in num_cols_without_target:
        replace_with_thresholds(df_copy, col)

    print('------------------------')

    for col in num_cols_without_target:
        print(col, check_outlier(df_copy, col))

    print('*------------------------*')


    ## Dealing with Missing Values and Encoding
    missing_df, missing_columns = missing_values_table(df_copy, True)


    df_copy = quick_missing_imp_groupped(df_copy, cat_cols = cat_cols, num_cols = num_cols_without_target, missing_columns_df = missing_df)


    #rare_analyser(df_copy, cat_cols)


    df_copy = rare_encoder(df_copy, 0.01)


    df_copy["NEW_1st*GrLiv"] = df_copy["1stFlrSF"] * df_copy["GrLivArea"]

    df_copy["NEW_Garage*GrLiv"] = (df_copy["GarageArea"] * df_copy["GrLivArea"])

    df_copy["TotalQual"] = df_copy[["OverallQual", "OverallCond", "ExterQual", "ExterCond", "BsmtCond", "BsmtFinType1",
                        "BsmtFinType2", "HeatingQC", "KitchenQual", "Functional", "FireplaceQu", "GarageQual", "GarageCond", "Fence", 'BsmtQual', 'BsmtExposure', 
                        'GarageFinish','PoolQC']].sum(axis = 1) # 42


    # Total Floor
    df_copy["NEW_TotalFlrSF"] = df_copy["1stFlrSF"] + df_copy["2ndFlrSF"] # 32

    # Total Finished Basement Area
    df_copy["NEW_TotalBsmtFin"] = df_copy.BsmtFinSF1 + df_copy.BsmtFinSF2 # 56

    # Porch Area
    df_copy["NEW_PorchArea"] = df_copy.OpenPorchSF + df_copy.EnclosedPorch + df_copy.ScreenPorch + df_copy["3SsnPorch"] + df_copy.WoodDeckSF # 93

    # Total House Area
    df_copy["NEW_TotalHouseArea"] = df_copy.NEW_TotalFlrSF + df_copy.TotalBsmtSF # 156

    df_copy["NEW_TotalSqFeet"] = df_copy.GrLivArea + df_copy.TotalBsmtSF # 35


    # Lot Ratio
    df_copy["NEW_LotRatio"] = df_copy.GrLivArea / df_copy.LotArea # 64

    df_copy["NEW_RatioArea"] = df_copy.NEW_TotalHouseArea / df_copy.LotArea # 57

    df_copy["NEW_GarageLotRatio"] = df_copy.GarageArea / df_copy.LotArea # 69

    # MasVnrArea
    df_copy["NEW_MasVnrRatio"] = df_copy.MasVnrArea / df_copy.NEW_TotalHouseArea # 36

    # Dif Area
    df_copy["NEW_DifArea"] = (df_copy.LotArea - df_copy["1stFlrSF"] - df_copy.GarageArea - df_copy.NEW_PorchArea - df_copy.WoodDeckSF) # 73


    df_copy["NEW_OverallGrade"] = df_copy["OverallQual"] * df_copy["OverallCond"] # 61


    df_copy["NEW_Restoration"] = np.where(df_copy["YearRemodAdd"] < df_copy["YearBuilt"], 0, df_copy["YearRemodAdd"] - df_copy["YearBuilt"])

    df_copy["NEW_HouseAge"] = np.where(df_copy["YrSold"] < df_copy["YearBuilt"], 0, df_copy["YrSold"] - df_copy["YearBuilt"])

    df_copy["NEW_RestorationAge"] = np.where(df_copy["YrSold"] < df_copy["YearRemodAdd"], 0, df_copy["YrSold"] - df_copy["YearRemodAdd"])

    df_copy["NEW_GarageAge"] = np.abs(df_copy.GarageYrBlt - df_copy.YearBuilt) # 17

    df_copy["NEW_GarageRestorationAge"] = np.abs(df_copy.GarageYrBlt - df_copy.YearRemodAdd) # 30

    df_copy["NEW_GarageSold"] = np.where(df_copy["YrSold"] < df_copy["GarageYrBlt"], 0, df_copy["YrSold"] - df_copy["GarageYrBlt"])


    df_copy["NEW_TotalBaths"] = df_copy["FullBath"] + df_copy["BsmtFullBath"] + 0.5*(df_copy["HalfBath"]+df_copy["BsmtHalfBath"])


    df_copy['NEW_HasPool'] = df_copy['PoolArea'].apply(lambda x: 1 if x > 0 else 0)


    df_copy['NEW_Has2ndFloor'] = df_copy['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)


    df_copy['NEW_HasGarage'] = df_copy['GarageCars'].apply(lambda x: 1 if x > 0 else 0)


    df_copy['NEW_HasBsmt'] = df_copy['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)


    df_copy['NEW_HasFireplace'] = df_copy['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)


    df_copy['NEW_HasPorch'] = df_copy['NEW_PorchArea'].apply(lambda x: 1 if x > 0 else 0)

    new_num_features = ['NEW_1st*GrLiv', 'NEW_Garage*GrLiv', 'TotalQual', 'NEW_TotalFlrSF', 'NEW_TotalBsmtFin', 'NEW_PorchArea', 'NEW_TotalHouseArea', 'NEW_TotalSqFeet',
                    'NEW_LotRatio', 'NEW_RatioArea', 'NEW_GarageLotRatio', 'NEW_MasVnrRatio', 'NEW_DifArea', 'NEW_OverallGrade', 'NEW_Restoration', 'NEW_HouseAge', 
                    'NEW_RestorationAge',  'NEW_GarageAge',  'NEW_GarageRestorationAge',  'NEW_GarageSold', 'NEW_TotalBaths']

    new_cat_features = ['NEW_HasPool', 'NEW_Has2ndFloor', 'NEW_HasGarage',
                    'NEW_HasBsmt', 'NEW_HasFireplace', 'NEW_HasPorch']


    num_cols.extend(new_num_features)
    cat_cols.extend(new_cat_features)


    ## Data Transform & Feature Scaling
    ### Data Transformation
    #By looking at the data we can say that "MSSubClass" and "YrSold" are Catagorical Variables, so we transform them into dtype : object
    df_copy[["MSSubClass", "YrSold"]] = df_copy[["MSSubClass", "YrSold"]].astype("category") #converting into catagorical value


    num_to_cat = ["MSSubClass", "YrSold"]


    for col in num_to_cat:
        if cat_cols.__contains__(col) == False:
            cat_cols.append(col)
            num_cols.remove(col)


    #"MoSold" is a Cyclic Value. We handle this type of data by mapping each cyclical variable onto a circle such that the lowest value for that variable appears right next to the largest value. We compute the x- and y- component of that point using sine and cosin trigonometric functions.
    df_copy["MoSoldsin"] = np.sin(2 * np.pi * df_copy["MoSold"] / 12) #Sine Function
    df_copy["MoSoldcos"] = np.cos(2 * np.pi * df_copy["MoSold"] / 12) #Cosine Function
    df_copy = df_copy.drop("MoSold", axis=1)


    num_cols.append('MoSoldsin')
    num_cols.append('MoSoldcos')
    num_cols.remove('MoSold')


    num_cols_without_target = [col for col in num_cols if col not in 'SalePrice']

    #scaler = RobustScaler()

    #df_copy[num_cols_without_target] = pd.DataFrame(scaler.fit_transform(df_copy[num_cols_without_target]))

    # Define a function to find the optimal lambda for a single column
    def find_best_lambda(column):
        # Objective function: negative p-value for normality
        def objective(lmbda, data):
            transformed_data = boxcox1p(data, lmbda)
            _, p_value = normaltest(transformed_data)
            return -p_value
        
        # Minimize the objective function
        result = minimize(objective, x0=0.0, args=(column,), bounds=[(-2, 2)])
        return result.x[0]
    
    transformed_columns = {}
    lambdas = {}

    for col in num_cols_without_target:
        # Ensure non-negative values (shift if necessary)
        min_value = df_copy[col].min()
        if min_value <= 0:
            df_copy[col] += abs(min_value) + 1  # Shift data to be strictly positive

        # Skip nearly constant columns
        if df_copy[col].std() < 1e-6:  # Threshold for low variance
            print(f"Skipping {col} due to low variance")
            continue

        # Find the best lambda for the column
        best_lambda = find_best_lambda(df_copy[col])
        lambdas[col] = best_lambda  # Store the optimal lambda

        # Apply the Box-Cox1p transformation
        transformed_columns[col] = boxcox1p(df_copy[col], best_lambda)

    # Update the DataFrame with transformed columns
    for col in num_cols_without_target:
        if col in transformed_columns:  # Only update if it was transformed
            df_copy[col] = transformed_columns[col]


    ## Encoding
    df_copy = one_hot_encoder(df_copy, cat_cols)
    df_copy = one_hot_encoder(df_copy, cat_but_car)


    ## Final
    #check_df(df_copy)


    split_index = 1460

    # İlk parça: 0'dan split_index'e kadar
    preprocessed_train = df_copy.iloc[:split_index]

    # İkinci parça: split_index'ten sona kadar
    preprocessed_test = df_copy.iloc[split_index:]


    #check_df(preprocessed_train)

    preprocessed_test = preprocessed_test.drop(["SalePrice"], axis=1)

    #LOCAL OUTLIER FACTOR
    clf = LocalOutlierFactor(n_neighbors=20)
    clf.fit_predict(preprocessed_train)

    df_scores = clf.negative_outlier_factor_
    print(df_scores[0:5])
    # df_scores = -df_scores
    print(np.sort(df_scores)[0:5])

    scores = pd.DataFrame(np.sort(df_scores))
    scores.plot(stacked=True, xlim=[0, 10], style='.-')
    plt.show()
    th = np.sort(df_scores)[9]

    print(preprocessed_train[df_scores < th])

    print(preprocessed_train[df_scores < th].shape)


    print(preprocessed_train.describe([0.01, 0.05, 0.75, 0.90, 0.99]).T)

    print(preprocessed_train[df_scores < th].index)


    preprocessed_train[df_scores < th].drop(axis=0, labels=preprocessed_train[df_scores < th].index)


    y = np.log(preprocessed_train["SalePrice"])
    X = preprocessed_train.drop(["SalePrice"], axis=1)

    rf_model = RandomForestRegressor(random_state=46, n_jobs=-1).fit(X, y)

    plot_importance(rf_model, X, num=20)

    ## TO CSV
    preprocessed_train.to_csv('../house_price_prediction_ann/house_price_prediction_ann/data/train_preprocessed.csv', index = False)
    preprocessed_test.to_csv('../house_price_prediction_ann/house_price_prediction_ann/data/test_preprocessed.csv', index = False)

    return X, y, preprocessed_test



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.backend as K

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


def main():
    X, y, test_prep = hpp_data_prep()

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=46)

    model = Sequential()
    model.add(Dense(128, activation='relu', input_dim=x_train.shape[1]))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss=root_mean_squared_error)

    early_stopping = EarlyStopping(monitor='val_loss', patience=25, verbose=1, mode='min', restore_best_weights=True)

    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=300, batch_size=250, callbacks=[early_stopping])
    #model.fit(X, y, epochs=300, batch_size=250, callbacks=[early_stopping])

    model_loss = pd.DataFrame(model.history.history)

    model_loss.plot()

    plt.show()

    predictions = model.predict(test_prep).flatten()  # 2D -> 1D'ye dönüştürülüyor
    dictionary = {"Id": test_prep.index + 1461, "SalePrice": predictions}
    dfSubmission = pd.DataFrame(dictionary)

    #dfSubmission['SalePrice'] = pd.DataFrame(scaler.inverse_transform(dfSubmission['SalePrice']))
    dfSubmission['SalePrice'] = np.exp(dfSubmission['SalePrice'])
    
    dfSubmission.to_csv("../house_price_prediction_ann/house_price_prediction_ann/data/housePricePredictions.csv", index=False)

    model.save("house_price_prediction_ann_model.h5")

    return model


if __name__ == "__main__":
    print("Process Started...")
    main()