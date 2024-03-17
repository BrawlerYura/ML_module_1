import pickle

from catboost import CatBoostClassifier
import pandas as pd
import numpy as np

class BrawlerClassifierModel:
    def kfold(self, X, y, k=5):
        X_trains = []
        y_trains = []
        X_vals = []
        y_vals = []

        for i in range(k):
            low = int(len(y) * i / k)
            high = int(len(y) * (i + 1) / k)
            y_vals.append(y[low:high])
            X_vals.append(X[low:high, :])

        for i in range(k):
            y_temp = []
            X_temp = []
            for j in range(k):
                if j != i:
                    y_temp.append(y_vals[j])
                    X_temp.append(X_vals[j])

            y_trains.append(np.concatenate(y_temp))
            X_trains.append(np.vstack(X_temp))
        result = []
        for i in range(k):
            result.append(((X_trains[i], y_trains[i]), (X_vals[i], y_vals[i])))
        return result

    def prepare_df(self, df):
        def replace_mode(table, column):
            val = table[column].mode()[0]
            table[column] = table[column].fillna(val)
            return table

        df.drop('Name', axis=1, inplace=True)

        df['Age_group'] = np.nan
        df.loc[(df['Age'] > 0) & (df['Age'] < 7), 'Age_group'] = 1
        df.loc[(df['Age'] >= 7) & (df['Age'] < 14), 'Age_group'] = 2
        df.loc[(df['Age'] >= 14) & (df['Age'] < 18), 'Age_group'] = 3
        df.loc[(df['Age'] >= 18) & (df['Age'] <= 25), 'Age_group'] = 4
        df.loc[(df['Age'] > 25) & (df['Age'] <= 30), 'Age_group'] = 5
        df.loc[(df['Age'] > 30) & (df['Age'] <= 50), 'Age_group'] = 6
        df.loc[df['Age'] > 50, 'Age_group'] = 7
        df.loc[df['Age'].isna(), 'Age_group'] = -1
        df.loc[df['Age'] == 0, 'Age_group'] = 0
        print(df['Age_group'].isna().sum(axis=0))

        df['Expenditure'] = df[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].sum(axis=1)
        df['No_spending'] = (df['Expenditure'] == 0).astype(int)

        df['Group'] = df['PassengerId'].apply(lambda x: x.split('_')[0]).astype(int)
        df['Group_size'] = df['Group'].map(lambda x: df['Group'].value_counts()[x])

        df['Cabin'] = df['Cabin'].fillna('Z/9999/Z')
        df['Cabin_deck'] = df['Cabin'].apply(lambda x: x.split('/')[0])
        df['Cabin_number'] = df['Cabin'].apply(lambda x: x.split('/')[1]).astype(int)
        df['Cabin_side'] = df['Cabin'].apply(lambda x: x.split('/')[2])
        df.loc[df['Cabin_deck'] == 'Z', 'Cabin_deck'] = np.nan
        df.loc[df['Cabin_number'] == 9999, 'Cabin_number'] = np.nan
        df.loc[df['Cabin_side'] == 'Z', 'Cabin_side'] = np.nan

        df.drop('Cabin', axis=1, inplace=True)

        df['Cabin_number_group'] = np.nan
        df.loc[df['Cabin_number'] < 400, 'Cabin_number_group'] = 0
        df.loc[(df['Cabin_number'] >= 400) & (df['Cabin_number'] < 900), 'Cabin_number_group'] = 1
        df.loc[(df['Cabin_number'] >= 900) & (df['Cabin_number'] <= 1200), 'Cabin_number_group'] = 2
        df.loc[(df['Cabin_number'] > 1200) & (df['Cabin_number'] <= 1500), 'Cabin_number_group'] = 3
        df.loc[(df['Cabin_number'] > 1500) & (df['Cabin_number'] < 9999), 'Cabin_number_group'] = 4
        df.loc[df['Cabin_number'] >= 9999, 'Cabin_number_group'] = 5

        GHP_gb = df.groupby(['Group', 'HomePlanet'])['HomePlanet'].size().unstack().fillna(0)

        GHP_index = df[df['HomePlanet'].isna()][
            (df[df['HomePlanet'].isna()]['Group']).isin(GHP_gb.index)].index

        df.loc[GHP_index, 'HomePlanet'] = df.iloc[GHP_index, :]['Group'].map(lambda x: GHP_gb.idxmax(axis=1)[x])

        CH_gb = df.groupby(['Cabin_deck', 'HomePlanet'])['HomePlanet'].size().unstack().fillna(0)

        GHP_index = df[df['HomePlanet'].isna()][
            (df[df['HomePlanet'].isna()]['Cabin_deck']).isin(CH_gb.index)].index

        df.loc[GHP_index, 'HomePlanet'] = df.iloc[GHP_index, :]['Cabin_deck'].map(
            lambda x: CH_gb.idxmax(axis=1)[x])

        df = replace_mode(df, 'HomePlanet')

        df.loc[df['CryoSleep'] == True, ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']] = 0
        df.loc[df['CryoSleep'].isna() & df['No_spending'] == True, 'CryoSleep'] = 0
        df.loc[df['CryoSleep'].isna(), 'CryoSleep'] = False

        GD_gb = df.groupby(['Group', 'Destination'])['Destination'].size().unstack().fillna(0)

        GHP_index = df[df['Destination'].isna()][
            (df[df['Destination'].isna()]['Group']).isin(GD_gb.index)].index

        df.loc[GHP_index, 'Destination'] = df.iloc[GHP_index, :]['Group'].map(lambda x: GD_gb.idxmax(axis=1)[x])

        GD_gb = df.groupby(['Cabin_deck', 'Destination'])['Destination'].size().unstack().fillna(0)

        GHP_index = df[df['Destination'].isna()][
            (df[df['Destination'].isna()]['Cabin_deck']).isin(GD_gb.index)].index

        df.loc[GHP_index, 'Destination'] = df.iloc[GHP_index, :]['Cabin_deck'].map(
            lambda x: GD_gb.idxmax(axis=1)[x])

        df = replace_mode(df, 'Destination')

        GCD_gb = df[df['Group_size'] > 1].groupby(['Group', 'Cabin_deck'])['Cabin_deck'].size().unstack().fillna(0)
        GCN_gb = df[df['Group_size'] > 1].groupby(['Group', 'Cabin_number'])['Cabin_number'].size().unstack().fillna(0)
        GCS_gb = df[df['Group_size'] > 1].groupby(['Group', 'Cabin_side'])['Cabin_side'].size().unstack().fillna(0)

        GCS_index = df[df['Cabin_side'].isna()][(df[df['Cabin_side'].isna()]['Group']).isin(GCS_gb.index)].index
        df.loc[GCS_index, 'Cabin_side'] = df.iloc[GCS_index, :]['Group'].map(lambda x: GCS_gb.idxmax(axis=1)[x])

        df['Cabin_side'] = df['Cabin_side'].fillna('Z')

        GCN_index = df[df['Cabin_number'].isna()][(df[df['Cabin_number'].isna()]['Group']).isin(GCN_gb.index)].index
        df.loc[GCN_index, 'Cabin_number'] = df.iloc[GCN_index, :]['Group'].map(lambda x: GCN_gb.idxmax(axis=1)[x])

        GCD_index = df[df['Cabin_deck'].isna()][(df[df['Cabin_deck'].isna()]['Group']).isin(GCD_gb.index)].index
        df.loc[GCD_index, 'Cabin_deck'] = df.iloc[GCD_index, :]['Group'].map(lambda x: GCD_gb.idxmax(axis=1)[x])

        na_rows_CD = df.loc[df['Cabin_deck'].isna() & df['Group_size'] == 1, 'Cabin_deck'].index
        df.loc[df['Cabin_deck'].isna(), 'Cabin_deck'] = \
        df[df['Group_size'] == 1].groupby(['HomePlanet', 'Destination', ])['Cabin_deck'].transform(
            lambda x: x.fillna(pd.Series.mode(x)[0]))[na_rows_CD]

        df['Cabin_deck'] = df['Cabin_deck'].fillna('F')

        df['Cabin_number'] = df['Cabin_number'].fillna(9999)

        df['Cabin_number_group'] = np.nan
        df.loc[df['Cabin_number'] < 400, 'Cabin_number_group'] = 0
        df.loc[(df['Cabin_number'] >= 400) & (df['Cabin_number'] < 900), 'Cabin_number_group'] = 1
        df.loc[(df['Cabin_number'] >= 900) & (df['Cabin_number'] <= 1200), 'Cabin_number_group'] = 2
        df.loc[(df['Cabin_number'] > 1200) & (df['Cabin_number'] <= 1500), 'Cabin_number_group'] = 3
        df.loc[(df['Cabin_number'] > 1500) & (df['Cabin_number'] < 9999), 'Cabin_number_group'] = 4
        df.loc[df['Cabin_number'] >= 9999, 'Cabin_number_group'] = 5

        df.groupby(['HomePlanet', 'No_spending', 'Group_size', 'Cabin_deck', 'Cabin_side'])[
            'Age'].mean().unstack().fillna(
            0)

        na_rows_A = df.loc[df['Age'].isna(), 'Age'].index
        df.loc[df['Age'].isna(), 'Age'] = \
        df.groupby(['HomePlanet', 'No_spending', 'Group_size', 'Cabin_deck', 'Cabin_side'])['Age'].transform(
            lambda x: x.fillna(x.mean()))[na_rows_A]

        df.loc[df['Age'] < 14, 'Age_group'] = 0
        df.loc[(df['Age'] >= 14) & (df['Age'] < 18), 'Age_group'] = 1
        df.loc[(df['Age'] >= 18) & (df['Age'] <= 25), 'Age_group'] = 2
        df.loc[(df['Age'] > 25) & (df['Age'] <= 30), 'Age_group'] = 3
        df.loc[(df['Age'] > 30) & (df['Age'] <= 50), 'Age_group'] = 4
        df.loc[df['Age'] > 50, 'Age_group'] = 5

        df.groupby(['HomePlanet', 'Group_size', 'Age_group'])['Expenditure'].mean().unstack().fillna(0)

        for col in ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']:
            na_rows = df.loc[df[col].isna(), col].index
            df.loc[df[col].isna(), col] = \
            df.groupby(['HomePlanet', 'Group_size', 'Age_group'])[col].transform(lambda x: x.fillna(x.mean()))[na_rows]

        mode_fill = df.groupby(['HomePlanet', 'Group_size', 'Age_group', 'Destination', 'Expenditure'])[
            'VIP'].transform(lambda x: x.mode().iat[0] if not x.mode().empty else False)
        df['VIP'] = df['VIP'].fillna(mode_fill)

        df["VIP"] = df["VIP"].astype(int)
        df["CryoSleep"] = df["CryoSleep"].astype(int)
        df.drop('PassengerId', axis=1, inplace=True)

        df = replace_mode(df, 'HomePlanet')
        df['HomePlanet'] = df['HomePlanet'].replace({'Earth': 0, 'Europa': 1, 'Mars': 2})
        df['Destination'] = df['Destination'].replace({'TRAPPIST-1e': 0, 'PSO J318.5-22': 1, '55 Cancri e': 2})
        df['Cabin_deck'] = df['Cabin_deck'].replace({'B': 1, 'F': 5, 'A': 0, 'G': 6, 'E': 4, 'D': 3, 'C': 2, 'T': 7})
        df['Cabin_side'] = df['Cabin_side'].replace({'P': 0, 'S': 1, 'Z': -1})

        df.drop('Group', axis=1, inplace=True)

        return df

    def __init__(self):
        with open('../data/model/model.pkl', 'rb') as model_pkl:
            self.model = pickle.load(model_pkl)


    def train(self, dataset):
        df_train = pd.read_csv(dataset)

        df_train_prepared = self.prepare_df(df_train)

        df_train_prepared["Transported"] = df_train_prepared["Transported"].astype(int)

        target = 'Transported'

        y = df_train_prepared[target].values
        X = df_train_prepared.drop(columns=target).values

        # def objective(trial):
        #    params = {
        #       'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        #       'depth': trial.suggest_int('depth', 5, 10),
        #       'iterations': trial.suggest_int('iterations', 100, 1200),
        #       # 'cat_features': [0, 1, 2, 3, 5, 11]
        #    }
        #
        #    k_fold = 13
        #
        #    model = CatBoostClassifier(**params)
        #
        #    accs = []
        #    for ((X_train, y_train), (X_val, y_val)) in kfold(X, y, k_fold):
        #       model.fit(X_train, y_train)
        #       pred = model.predict(X_val)
        #       accs.append(np.mean(pred == y_val))
        #    acc_mean = np.mean(accs)
        #
        #    return acc_mean
        #
        #
        # study = optuna.create_study(direction='maximize')
        # study.optimize(objective, n_trials=100)
        #
        # best_params = study.best_params
        # print("Best params:", best_params)

        model_cb = CatBoostClassifier(learning_rate=0.07981609439133353, depth=5, iterations=392)

        accs = []
        for ((X_train, y_train), (X_val, y_val)) in self.kfold(X, y, 13):
            model_cb.fit(X_train, y_train)
            pred = model_cb.predict(X_val)
            accs.append(np.mean(pred == y_val))
        acc_mean = np.mean(accs)

        print(acc_mean)

        with open('../data/model/model.pkl', 'wb') as model_pkl:
            pickle.dump(model_cb, model_pkl)

        return "Training completed successfully"



    def predict(self, dataset):
        df_test = pd.read_csv(dataset)
        ids = df_test['PassengerId']

        df_test_prepared = self.prepare_df(df_test)

        predict = self.model.predict(df_test_prepared)
        predict = np.where(predict == 1, True, False)

        predictions_df = pd.DataFrame({'PassengerId': ids, 'Transported': predict})

        predictions_df.to_csv('../data/results.csv', index=False)

        return "Predictiong completed successfully"
