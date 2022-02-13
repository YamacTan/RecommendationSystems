#######################
# Yamac TAN - Data Science Bootcamp - Week 4 - Project 1
#######################

# %%
import pandas as pd
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.expand_frame_repr', False)
from mlxtend.frequent_patterns import apriori, association_rules

# %%
###############################################
# Görev 1 - Adım 1
###############################################

df_ = pd.read_excel("Odevler/HAFTA_04/PROJE_1/online_retail_II.xlsx", sheet_name="Year 2010-2011")
df = df_.copy()
df.head()
df.shape
# %%
###############################################
# Görev 1 - Adım 2
###############################################

df = df[~df["StockCode"].str.contains("POST", na=False)]

# %%
###############################################
# Görev 1 - Adım 3
###############################################
df.dropna(inplace=True)

# %%
###############################################
# Görev 1 - Adım 4
###############################################

df = df[~df["Invoice"].str.contains("C", na=False)]

# %%
###############################################
# Görev 1 - Adım 5
###############################################

df = df[df["Price"] > 0]

# %%
###############################################
# Görev 1 - Adım 5
###############################################
df.dtypes
df[["Price", "Quantity"]].describe().T

# Betimsel istatistik gözleminde aykırı değerler olduğundan baskılanması gerekmektedir.

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

replace_with_thresholds(df, "Price")
replace_with_thresholds(df, "Quantity")

# %%
###############################################
# Görev 2 - Adım 1
###############################################

df_ger = df[df["Country"] == "Germany"]  # Veri setini pivot etmeden önce Almanya müşterilerine indirgedik.
df_ger.head()

def create_invoice_product_df(df, id=False):

    if id:
        return df.groupby(["Invoice", "StockCode"])["Quantity"].sum().unstack().fillna(0) \
            .applymap(lambda x: 1 if x > 0 else 0)
    else:
        return df.groupby(["Invoice", "Description"])["Quantity"].sum().unstack().fillna(0) \
        .applymap(lambda x: 1 if x > 0 else 0)

pivot_ger = create_invoice_product_df(df_ger)

# %%
###############################################
# Görev 2 - Adım 2
###############################################

def create_rules(df, country="Germany", id = False):
    df_county = df[df["Country"] == country]
    pivot = create_invoice_product_df(df_county, id)
    frequent_items = apriori(pivot, min_support=0.01, use_colnames=True)
    rules = association_rules(frequent_items,
                              metric="support",
                              min_threshold=0.01)
    return rules

rules_ger = create_rules(df)

# %%
###############################################
# Görev 3 - Adım 1
###############################################

def check_id(dataframe, stock_code):
    product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()
    print(product_name)

rules_ger_with_ids = create_rules(df,id = True)

#check_id kullanımını görebilmek için Antecedents sütununundan bazı ürünlerin isimlerine bakılabilir.
check_id(df,16237)
check_id(df,22326)
check_id(df,20674)

# %%
###############################################
# Görev 3 - Adım 2
###############################################

def arl_recommender(rules_df, product_id, rec_count=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list = []
    for i, product in enumerate(sorted_rules["antecedents"]):
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])

    return recommendation_list[0:rec_count]

#df_ger dataframe'indeki kullanıcıların aldığı farklı ürünlerden 3 adet örnek ürün seçilsin.
# 16237, 20674, 22423

sample_products = [16237, 20674, 22423]
recommended_products =[]
for productid in sample_products:
    recommended_products.append(arl_recommender(rules_ger_with_ids, productid)[0])

recommended_products

# %%
###############################################
# Görev 3 - Adım 3
###############################################

recommended_names = [check_id(df,x) for x in recommended_products]