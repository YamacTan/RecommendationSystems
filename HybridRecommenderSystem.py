#######################
# Yamac TAN - Data Science Bootcamp - Week 4 - Project 2
#######################

# %%
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)

# %%
###############################################
# Görev 1 - User Based Recommendation
###############################################

movie = pd.read_csv('Notes/HAFTA_04 RECOMMENDER SYSTEMS/recommender_systems/datasets/movie_lens_dataset/movie.csv')
rating = pd.read_csv('Notes/HAFTA_04 RECOMMENDER SYSTEMS/recommender_systems/datasets/movie_lens_dataset/rating.csv')
df = rating.merge(movie, how="left", on="movieId")

# Rare movies bilgisine ulaşmak için filmlere yapılan yorum sayılarını bulmamız gerekmektedir.
comments = pd.DataFrame(df["title"].value_counts())
rare_movies = comments[comments["title"] < 1000].index
common_movies = df[~df["title"].isin(rare_movies)]

user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")


def prep_user_movie_data():
    movie = pd.read_csv('Notes/HAFTA_04 RECOMMENDER SYSTEMS/recommender_systems/datasets/movie_lens_dataset/movie.csv')
    rating = pd.read_csv(
        'Notes/HAFTA_04 RECOMMENDER SYSTEMS/recommender_systems/datasets/movie_lens_dataset/rating.csv')
    df = rating.merge(movie, how="left", on="movieId")
    comments = pd.DataFrame(df["title"].value_counts())
    rare_movies = comments[comments["title"] < 1000].index
    common_movies = df[~df["title"].isin(rare_movies)]
    user_movie_df = common_movies.pivot_table(values="rating", index=["userId"], columns=["title"])
    return user_movie_df


# %%
###############################################
# Görev 2
###############################################

# Execute sırasında değerin değişmemesi adına random_state belirtilmiştir.
random_user = int(pd.Series(user_movie_df.index).sample(1, random_state=60).values)
random_user_df = user_movie_df[user_movie_df.index == random_user]
movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist()

# %%
###############################################
# Görev 3
###############################################

movies_watched_df = user_movie_df[movies_watched]
user_movie_count = movies_watched_df.T.notnull().sum().reset_index()
user_movie_count.columns = ["userId", "movie_count"]
users_same_movies = user_movie_count[user_movie_count["movie_count"] >= (len(movies_watched) * 60 / 100)][
    "userId"].tolist()

# %%
###############################################
# Görev 4
###############################################


movies_watched_df_filtered = pd.concat(
    [movies_watched_df[movies_watched_df.index.isin(users_same_movies)], random_user_df[movies_watched]])

corr_df = movies_watched_df_filtered.T.corr().unstack().sort_values().drop_duplicates()
corr_df = pd.DataFrame(corr_df, columns=["corr"])
corr_df.index.names = ['user1', 'user2']
corr_df = corr_df.reset_index()

top_users = corr_df[(corr_df["user1"] == random_user) & (corr_df["corr"] >= 0.65)][["user2", "corr"]].reset_index(drop=True)

#Merge işlemi yapabilmek adına sütun adını merge veri setindekine uygun hale getirmeliyiz.
top_users.rename(columns={"user2": "userId"}, inplace=True)
final_df = top_users.merge(rating[["userId", "movieId", "rating"]], how='inner')

# %%
###############################################
# Görev 5
###############################################

final_df["weighted_rating"] = final_df["corr"] * final_df["rating"]

recommendation_df = final_df.groupby('movieId').agg({"weighted_rating": "mean"})

recommendation_df = recommendation_df.reset_index()
recommended_movies = recommendation_df[recommendation_df["weighted_rating"] > 3.5].sort_values("weighted_rating",
                                                                                            ascending=False)
recommended_movies = recommended_movies.merge(movie[["movieId", "title"]])

recommended_movies["title"].head(5)


# %%
###############################################
#Item Based Recommendation
###############################################

movie = pd.read_csv('Notes/HAFTA_04 RECOMMENDER SYSTEMS/recommender_systems/datasets/movie_lens_dataset/movie.csv')
rating = pd.read_csv('Notes/HAFTA_04 RECOMMENDER SYSTEMS/recommender_systems/datasets/movie_lens_dataset/rating.csv')
df = rating.merge(movie, how="left", on="movieId")

current_df = df[(df["userId"] == random_user) & (df["rating"] == 5.0)].sort_values(by = "timestamp", ascending = False)

most_current_id = int(current_df.iloc[0:1]["movieId"].values[0])
movie_name = movie[movie["movieId"] == most_current_id]["title"]
new_user_movie_df = user_movie_df[movie_name]

recommendation = user_movie_df.corrwith(new_user_movie_df).sort_values(ascending=False)

recommendation[1:6]
