import pandas as pd
df = pd.read_csv('data_raw.csv')
all_features = df.columns
names = [feat for feat in all_features if "net_name" in feat]
useless = ["info_gew", "info_resul", "interviewtime", "id", "date"]
droplist = names+useless
practice_list = ["legum", "conc", "add", "lact", "breed", "covman",
                 "comp", "drag", "cov", "plow", "solar", "biog", "ecodr"]
for feat in all_features:
    if any(x in feat for x in practice_list):
        droplist.append(feat)
df = df.drop(columns=droplist)
non_numeric = list(df.select_dtypes(include=['O']).columns)
for col in non_numeric:
    codes, uniques = pd.factorize(df[col])
    df[col] = codes
df.to_csv('data_processed.csv')
