"""
Loads data sets from files and processes them in a shared format.
Implemented support for the following explicit feedback data sets:
Amazon2018 - 29 data sets (https://nijianmo.github.io/amazon/index.html)
Anime (https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database)
Beer - 2 data sets (https://cseweb.ucsd.edu/~jmcauley/datasets.html#multi_aspect)
BookCrossing (http://www2.informatik.uni-freiburg.de/~cziegler/BX/)
CiaoDVD (https://guoguibing.github.io/librec/datasets.html)
ClothingFit - 2 data sets (https://www.kaggle.com/datasets/rmisra/clothing-fit-dataset-for-size-recommendation)
Douban - 3 data sets (https://github.com/DeepGraphLearning/RecommenderSystems)
DoubanShort (https://www.kaggle.com/datasets/utmhikari/doubanmovieshortcomments)
EachMovie (http://www.gatsby.ucl.ac.uk/~chuwei/data/EachMovie/eachmovie.html)
Epinions - 2 data sets (http://www.trustlet.org/epinions.html)
Filmtrust (https://guoguibing.github.io/librec/datasets.html)
FoodComRecipes (https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions)
Goodreads (https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/home)
GoogleLocal (https://cseweb.ucsd.edu/~jmcauley/datasets.html#google_local)
Jester - 2 data sets (https://eigentaste.berkeley.edu/dataset/)
LearningFromSets (https://grouplens.org/datasets/learning-from-sets-of-items-2019/)
Libimseti (http://konect.cc/networks/libimseti/)
LibraryThing (https://cseweb.ucsd.edu/~jmcauley/datasets.html#social_data)
MarketBias - 2 data sets (https://github.com/MengtingWan/marketBias)
MovieLens - 7 data sets (https://grouplens.org/datasets/movielens/)
MovieTweetings (https://github.com/sidooms/MovieTweetings)
Netflix (https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data)
Rekko (https://boosters.pro/championship/rekko_challenge/data)
Wikilens (http://konect.cc/networks/wikilens-ratings/)
Yahoo - 4 data sets (https://webscope.sandbox.yahoo.com/)
Yelp (https://www.yelp.com/dataset)
"""

import os
from pathlib import Path
import gzip
from typing import List, Tuple
from datetime import datetime
import time
import json
import pandas as pd
import numpy as np


file_path = Path(os.path.dirname(os.path.abspath(__file__)))

# list containing all supported data sets
all_selectors = [
    ("Amazon2018", "all-beauty"),
    ("Amazon2018", "appliances"),
    ("Amazon2018", "arts-crafts-and-sewing"),
    ("Amazon2018", "automotive"),
    ("Amazon2018", "books"),
    ("Amazon2018", "cds-and-vinyl"),
    ("Amazon2018", "cell-phones-and-accessories"),
    ("Amazon2018", "clothing-shoes-and-jewelry"),
    ("Amazon2018", "digital-music"),
    ("Amazon2018", "electronics"),
    ("Amazon2018", "fashion"),
    ("Amazon2018", "gift-cards"),
    ("Amazon2018", "grocery-and-gourmet-food"),
    ("Amazon2018", "home-and-kitchen"),
    ("Amazon2018", "industrial-and-scientific"),
    ("Amazon2018", "kindle-store"),
    ("Amazon2018", "luxury-beauty"),
    ("Amazon2018", "magazine-subscriptions"),
    ("Amazon2018", "movies-and-tv"),
    ("Amazon2018", "musical-instruments"),
    ("Amazon2018", "office-products"),
    ("Amazon2018", "patio-lawn-and-garden"),
    ("Amazon2018", "pet-supplies"),
    ("Amazon2018", "prime-pantry"),
    ("Amazon2018", "sports-and-outdoors"),
    ("Amazon2018", "software"),
    ("Amazon2018", "tools-and-home-improvement"),
    ("Amazon2018", "toys-and-games"),
    ("Amazon2018", "video-games"),
    ("Anime", "anime"),
    ("Beer", "beeradvocate"),
    ("Beer", "ratebeer"),
    ("BookCrossing", "bookcrossing"),
    ("CiaoDVD", "ciaodvd"),
    ("ClothingFit", "modcloth"),
    ("ClothingFit", "renttherunway"),
    ("Douban", "book"),
    ("Douban", "movie"),
    ("Douban", "music"),
    ("DoubanShort", "douban-short"),
    ("EachMovie", "eachmovie"),
    ("Epinions", "epinions"),
    ("Epinions", "epinions-extended"),
    ("Filmtrust", "filmtrust"),
    ("FoodComRecipes", "foodcom-recipes"),
    ("Goodreads", "goodreads"),
    ("GoogleLocal", "googlelocal"),
    ("Jester", "jester3"),
    ("Jester", "jester4"),
    ("LearningFromSets", "learningfromsets"),
    ("Libimseti", "libimseti"),
    ("LibraryThing", "librarything"),
    ("MarketBias", "amazon"),
    ("MarketBias", "modcloth"),
    ("MovieLens", "1m"),
    ("MovieLens", "10m"),
    ("MovieLens", "20m"),
    ("MovieLens", "25m"),
    ("MovieLens", "100k"),
    ("MovieLens", "latest"),
    ("MovieLens", "latest-small"),
    ("MovieTweetings", "movietweetings"),
    ("Netflix", "netflixprize"),
    ("Rekko", "rekko"),
    ("WikiLens", "wikilens"),
    ("Yahoo", "movies"),
    ("Yahoo", "music1"),
    ("Yahoo", "music2"),
    ("Yahoo", "music3"),
    ("Yelp", "yelp")
]


def map_ids(data: pd.DataFrame, cols: List[str]):
    """
    maps the incoming columns to ascending integers
    """
    for col in cols:
        unique_ids = {key: value for value, key in enumerate(data[col].unique())}
        data[col].update(data[col].map(unique_ids))


def normalize_names(data: pd.DataFrame, cols: List[str]):
    """
    renames and sets the data types of columns
    """
    if len(cols) == 4:
        data.rename(columns={cols[0]: 'user', cols[1]: 'item', cols[2]: 'rating', cols[3]: 'timestamp'}, inplace=True)
        data['user'] = data['user'].astype(np.int64)
        data['item'] = data['item'].astype(np.int64)
        data['rating'] = data['rating'].astype(np.float64)
        data['timestamp'] = data['timestamp'].astype(np.int64)
    elif len(cols) == 3:
        data.rename(columns={cols[0]: 'user', cols[1]: 'item', cols[2]: 'rating'}, inplace=True)
        data['user'] = data['user'].astype(np.int64)
        data['item'] = data['item'].astype(np.int64)
        data['rating'] = data['rating'].astype(np.float64)
    else:
        print("The data needs to have three or four columns.")


def load_and_process_data_set(data_set_category: str, data_set: str,
                              base_folder: Path = Path(file_path / "data_sets/"),
                              return_timestamp: bool = True,
                              load_only_stats: bool = False,
                              process_all: bool = False):
    """
    either loads the processed data from a file or loads the original data and processes it.
    """

    if process_all:
        print("Trying to process all available data sets.")
        for selector in all_selectors:
            load_and_process_data_set(selector[0], selector[1], base_folder, return_timestamp, load_only_stats, False)
        print("Processed all data sets.")
        return None, None

    if os.path.exists(f"{base_folder}/.processed/{data_set_category}/{data_set}"):
        print(f"Loading {data_set_category} {data_set} from processed file.")
        with open(f"{base_folder}/.processed/{data_set_category}/{data_set}/stats.json", 'r') as json_file:
            stats = json.load(json_file)
        if load_only_stats:
            return None, stats
        else:
            relevant_data = pd.read_hdf(f"{base_folder}/.processed/{data_set_category}/{data_set}/processed.h5")
            if isinstance(relevant_data, pd.DataFrame):
                if return_timestamp:
                    return relevant_data, stats
                else:
                    return relevant_data[["user", "item", "rating"]], stats
            else:
                print(f"The loaded data {data_set_category} {data_set} is not a DataFrame. Processing again.")

    start = time.perf_counter()
    print(f"Processing {data_set_category} {data_set}.")

    relevant_data = None
    if data_set_category == "Amazon2018":
        data_folder = Path(base_folder / f"Amazon2018/{data_set}")
        rating_file = data_folder / Path(str(os.listdir(data_folder)[0]))

        # read file
        relevant_data = pd.read_json(rating_file, lines=True,
                                     dtype={
                                         'reviewerID': str,
                                         'asin': str,
                                         'overall': np.float64,
                                         'unixReviewTime': np.float64
                                     })[['reviewerID', 'asin', 'overall', 'unixReviewTime']]

        # re-map user and item ids
        map_ids(relevant_data, ["reviewerID", "asin"])

        # set default names and data types
        normalize_names(relevant_data, ["reviewerID", "asin", "overall", "unixReviewTime"])
    elif data_set_category == "Anime":
        data_folder = Path(base_folder / f"Anime/{data_set}")
        rating_file = data_folder / "rating.csv"

        # read file
        relevant_data = pd.read_csv(rating_file, header=0,
                                    names=['user_id', 'anime_id', 'rating'],
                                    dtype={
                                        'user': np.float64,
                                        'item': np.float64,
                                        'rating': np.float64
                                    })

        # remove non-ratings.
        relevant_data = relevant_data[relevant_data["rating"] != -1]

        # re-map user and item ids
        map_ids(relevant_data, ["user_id", "anime_id"])

        # set default names and data types
        normalize_names(relevant_data, ["user_id", "anime_id", "rating"])
    elif data_set_category == "Beer":
        data_folder = Path(base_folder / f"Beer/{data_set}")
        rating_file = data_folder / Path(str(os.listdir(data_folder)[0]))

        # read file
        final_dict = {'review/profileName': [], 'beer/beerId': [], 'review/overall': [], 'review/time': []}
        with open(rating_file) as file:
            for line in file:
                dic = eval(line)
                if all(k in dic for k in ("review/profileName", "beer/beerId", "review/overall", "review/time")):
                    final_dict['review/profileName'].append(dic['review/profileName'])
                    final_dict['beer/beerId'].append(dic['beer/beerId'])
                    final_dict['review/overall'].append(dic['review/overall'])
                    final_dict['review/time'].append(dic['review/time'])
        relevant_data = pd.DataFrame.from_dict(final_dict)

        # re-map user and item ids
        map_ids(relevant_data, ["review/profileName", "beer/beerId"])

        # fix ratings
        relevant_data["review/overall"] = relevant_data["review/overall"].apply(lambda x: x.split('/')[0])

        # set default names and data types
        normalize_names(relevant_data, ["review/profileName", "beer/beerId", "review/overall", "review/time"])
    elif data_set_category == "BookCrossing":
        data_folder = Path(base_folder / f"BookCrossing/{data_set}")
        rating_file = data_folder / "BX-Book-Ratings.csv"

        # read file
        relevant_data = pd.read_csv(rating_file, header=0, sep=';', encoding="unicode_escape",
                                    dtype={
                                        'User-ID': np.int64,
                                        'ISBN': str,
                                        'Book-Rating': np.float64
                                    })

        # remove non-ratings.
        relevant_data = relevant_data[relevant_data["Book-Rating"] != 0]

        # re-map user and item ids
        map_ids(relevant_data, ["User-ID", "ISBN"])

        # set default names and data types
        normalize_names(relevant_data, ["User-ID", "ISBN", "Book-Rating"])
    elif data_set_category == "CiaoDVD":
        data_folder = Path(base_folder / f"CiaoDVD/{data_set}")
        rating_file = data_folder / "movie-ratings.txt"

        # read file
        relevant_data = pd.read_csv(rating_file, header=None, sep=',',
                                    names=['userId', 'movieId', 'movie-categoryId', 'reviewId', 'movieRating',
                                           'reviewDate'],
                                    usecols=['userId', 'movieId', 'movieRating', 'reviewDate'],
                                    dtype={
                                        'userId': np.int64,
                                        'movieId': np.int64,
                                        'movieRating': np.float64,
                                        'reviewDate': str
                                    })

        # re-map user and item ids
        map_ids(relevant_data, ["userId", "movieId"])

        # convert date
        relevant_data["reviewDate"] = relevant_data["reviewDate"].apply(
            lambda x: time.mktime(datetime.strptime(x, "%Y-%m-%d").timetuple()))

        # set default names and data types
        normalize_names(relevant_data, ["userId", "movieId", "movieRating", "reviewDate"])
    elif data_set_category == "ClothingFit":
        data_folder = Path(base_folder / f"ClothingFit/{data_set}")
        rating_file = data_folder / Path(str(os.listdir(data_folder)[0]))

        if data_set == "modcloth":
            # read file
            relevant_data = pd.read_json(rating_file, lines=True,
                                         dtype={
                                             'user_id': np.int64,
                                             'item_id': np.int64,
                                             'quality': np.float64
                                         })[["user_id", "item_id", "quality"]]

            # fix nan ratings
            relevant_data = relevant_data[~relevant_data["quality"].isna()]

            # re-map user and item ids
            map_ids(relevant_data, ["user_id", "item_id"])

            # set default names and data types
            normalize_names(relevant_data, ["user_id", "item_id", "quality"])
        elif data_set == "renttherunway":
            # read file
            relevant_data = pd.read_json(rating_file, lines=True,
                                         dtype={
                                             'user_id': np.int64,
                                             'item_id': np.int64,
                                             'rating': np.float64,
                                             'review_date': str
                                         })[["user_id", "item_id", "rating", "review_date"]]

            # fix nan ratings
            relevant_data = relevant_data[~relevant_data["rating"].isna()]

            # re-map user and item ids
            map_ids(relevant_data, ["user_id", "item_id"])

            # convert date
            relevant_data["review_date"] = relevant_data["review_date"].apply(
                lambda x: time.mktime(datetime.strptime(x, "%B %d, %Y").timetuple()))

            # set default names and data types
            normalize_names(relevant_data, ["user_id", "item_id", "rating", "review_date"])
    elif data_set_category == "Douban":
        data_folder = Path(base_folder / f"Douban/{data_set}")
        rating_file = data_folder / Path(str(os.listdir(data_folder)[0]))

        # read file
        relevant_data = pd.read_csv(rating_file, header=0, sep='\t',
                                    dtype={
                                        'UserId': np.int64,
                                        'ItemId': np.int64,
                                        'Rating': np.int64,
                                        'Timestamp': np.float64
                                    })

        # remove non-ratings.
        relevant_data = relevant_data[relevant_data["Rating"] != -1]

        # re-map user and item ids
        map_ids(relevant_data, ["UserId", "ItemId"])

        # set default names and data types
        normalize_names(relevant_data, ["UserId", "ItemId", "Rating", "Timestamp"])
    elif data_set_category == "DoubanShort":
        data_folder = Path(base_folder / f"DoubanShort/{data_set}")
        rating_file = data_folder / Path(str(os.listdir(data_folder)[0]))

        # read file
        relevant_data = \
            pd.read_csv(rating_file, header=0, usecols=['Username', 'Movie_Name_EN', 'Star', 'Date'], dtype={
                'Username': str,
                'Movie_Name_EN': str,
                'Star': np.float64,
                'Date': str
            })[['Username', 'Movie_Name_EN', 'Star', 'Date']]

        # re-map user and item ids
        map_ids(relevant_data, ["Username", "Movie_Name_EN"])

        # convert date
        relevant_data["Date"] = relevant_data["Date"].apply(
            lambda x: time.mktime(datetime.strptime(x, "%Y-%m-%d").timetuple()))

        # set default names and data types
        normalize_names(relevant_data, ["Username", "Movie_Name_EN", "Star", "Date"])
    elif data_set_category == "EachMovie":
        data_folder = Path(base_folder / f"EachMovie/{data_set}")
        rating_file = data_folder / Path(str(os.listdir(data_folder)[0]))

        # read file
        relevant_data = pd.read_csv(rating_file, delim_whitespace=True, header=None,
                                    names=['user', 'item', 'rating'],
                                    dtype={
                                        'user': np.int64,
                                        'item': np.int64,
                                        'rating': np.float64
                                    })

        # re-map user and item ids
        map_ids(relevant_data, ["user", "item"])

        # adjust ratings
        relevant_data["rating"] -= 1

        # set default names and data types
        normalize_names(relevant_data, ["user", "item", "rating"])
    elif data_set_category == "Epinions":
        data_folder = Path(base_folder / f"Epinions/{data_set}")
        rating_file = data_folder / Path(str(os.listdir(data_folder)[0]))

        if data_set == "epinions":
            # read file
            relevant_data = pd.read_csv(rating_file, header=None, sep=' ',
                                        names=['user', 'item', 'rating'],
                                        dtype={
                                            'user': np.int64,
                                            'item': np.int64,
                                            'rating': np.float64
                                        })

            # re-map user and item ids
            map_ids(relevant_data, ["user", "item"])

            # set default names and data types
            normalize_names(relevant_data, ["user", "item", "rating"])
        elif data_set == "epinions-extended":
            # read file
            relevant_data = pd.read_csv(rating_file, header=None, sep='\t',
                                        names=['OBJECT_ID', 'MEMBER_ID', 'RATING', 'STATUS', 'CREATION',
                                               'LAST_MODIFIED', 'TYPE', 'VERTICAL_ID'],
                                        usecols=['MEMBER_ID', 'OBJECT_ID', 'RATING', 'CREATION',
                                                 'LAST_MODIFIED'],
                                        dtype={
                                            'OBJECT_ID': np.int64,
                                            'MEMBER_ID': np.int64,
                                            'RATING': np.float64,
                                            'CREATION': str,
                                            'LAST_MODIFIED': str
                                        })[['MEMBER_ID', 'OBJECT_ID', 'RATING', 'CREATION',
                                            'LAST_MODIFIED']]

            # treat rating 6 as 5 per documentation
            relevant_data.loc[relevant_data["RATING"] == 6, ["RATING"]] = 5

            # fix missing date
            relevant_data["LAST_MODIFIED"] = np.where(relevant_data["LAST_MODIFIED"].isna(),
                                                      relevant_data['CREATION'],
                                                      relevant_data["LAST_MODIFIED"])
            relevant_data.drop(columns=["CREATION"], inplace=True)

            # re-map user and item ids
            map_ids(relevant_data, ["MEMBER_ID", "OBJECT_ID"])

            # convert date
            relevant_data["LAST_MODIFIED"] = relevant_data["LAST_MODIFIED"].apply(
                lambda x: time.mktime(datetime.strptime(x, "%Y/%m/%d").timetuple()))

            # set default names and data types
            normalize_names(relevant_data, ["MEMBER_ID", "OBJECT_ID", "RATING", "LAST_MODIFIED"])
    elif data_set_category == "Filmtrust":
        data_folder = Path(base_folder / f"Filmtrust/{data_set}")
        rating_file = data_folder / "ratings.txt"

        # read file
        relevant_data = pd.read_csv(rating_file, header=None, sep=' ',
                                    names=['user', 'item', 'rating'],
                                    dtype={
                                        'user': np.int64,
                                        'item': np.int64,
                                        'rating': np.float64
                                    })

        # re-map user and item ids
        map_ids(relevant_data, ["user", "item"])

        # set default names and data types
        normalize_names(relevant_data, ["user", "item", "rating"])
    elif data_set_category == "FoodComRecipes":
        data_folder = Path(base_folder / f"FoodComRecipes/{data_set}")
        rating_file = data_folder / Path(str(os.listdir(data_folder)[0]))

        # read file
        relevant_data = pd.read_csv(rating_file, header=0, sep=',',
                                    usecols=['user_id', 'recipe_id', 'date', 'rating'],
                                    dtype={
                                        'user_id': np.int64,
                                        'recipe_id': np.int64,
                                        'date': str,
                                        'rating': np.float64
                                    })

        # re-map user and item ids
        map_ids(relevant_data, ["user_id", "recipe_id"])

        # convert date
        relevant_data["date"] = relevant_data["date"].apply(
            lambda x: time.mktime(datetime.strptime(x, "%Y-%m-%d").timetuple()))

        # set default names and data types
        normalize_names(relevant_data, ["user_id", "recipe_id", "rating", "date"])
    elif data_set_category == "Goodreads":
        data_folder = Path(base_folder / f"Goodreads/{data_set}")
        rating_file = data_folder / Path(str(os.listdir(data_folder)[0]))

        # read file
        relevant_data = pd.read_csv(rating_file, header=0, sep=',',
                                    usecols=['user_id', 'book_id', 'rating'],
                                    dtype={
                                        'user_id': np.int64,
                                        'book_id': np.int64,
                                        'rating': np.float64
                                    })

        # re-map user and item ids
        map_ids(relevant_data, ["user_id", "book_id"])

        # set default names and data types
        normalize_names(relevant_data, ["user_id", "book_id", "rating"])
    elif data_set_category == "GoogleLocal":
        data_folder = Path(base_folder / f"GoogleLocal/{data_set}")
        rating_file = data_folder / Path(str(os.listdir(data_folder)[0]))

        # read file
        final_dict = {'gPlusUserId': [], 'gPlusPlaceId': [], 'rating': [], 'unixReviewTime': []}
        with gzip.GzipFile(rating_file, "r") as f:
            for line in f:
                dic = eval(line)
                final_dict['gPlusUserId'].append(dic['gPlusUserId'])
                final_dict['gPlusPlaceId'].append(dic['gPlusPlaceId'])
                final_dict['rating'].append(dic['rating'])
                final_dict['unixReviewTime'].append(dic['unixReviewTime'])
        relevant_data = pd.DataFrame.from_dict(final_dict)

        # re-map user and item ids
        map_ids(relevant_data, ["gPlusUserId", "gPlusPlaceId"])

        # remove nan
        relevant_data["unixReviewTime"] = relevant_data["unixReviewTime"].replace(np.nan, -1)

        # set default names and data types
        normalize_names(relevant_data, ["gPlusUserId", "gPlusPlaceId", "rating", "unixReviewTime"])
    elif data_set_category == "Jester":
        data_folder = Path(base_folder / f"Jester/{data_set}")
        rating_file = data_folder / Path(str(os.listdir(data_folder)[0]))

        # read file
        relevant_data = pd.read_excel(rating_file, sheet_name=0, header=None).loc[:, 1:]
        relevant_data["user"] = [i for i in range(len(relevant_data))]
        relevant_data = relevant_data.melt(id_vars="user", var_name="item", value_name="rating")
        relevant_data = relevant_data[relevant_data["rating"] != 99]
        relevant_data.dropna(subset=["rating"], inplace=True)
        relevant_data = relevant_data.loc[(-10 <= relevant_data["rating"]) & (relevant_data["rating"] <= 10)]

        # re-map user and item ids
        map_ids(relevant_data, ["user", "item"])

        # set default names and data types
        normalize_names(relevant_data, ["user", "item", "rating"])
    elif data_set_category == "LearningFromSets":
        data_folder = Path(base_folder / f"LearningFromSets/{data_set}")
        rating_file = data_folder / "item_ratings.csv"

        # read file
        relevant_data = pd.read_csv(rating_file, header=0,
                                    usecols=['userId', 'movieId', 'rating', 'timestamp'],
                                    dtype={
                                        'userId': str,
                                        'movieId': np.int64,
                                        'rating': np.float64,
                                        'timestamp': str
                                    })

        # re-map user and item ids
        map_ids(relevant_data, ["userId", "movieId"])

        # convert date
        relevant_data["timestamp"] = relevant_data["timestamp"].apply(
            lambda x: time.mktime(datetime.strptime(x, "%Y-%m-%d %H:%M:%S").timetuple()))

        # set default names and data types
        normalize_names(relevant_data, ["userId", "movieId", "rating", "timestamp"])
    elif data_set_category == "Libimseti":
        data_folder = Path(base_folder / f"Libimseti/{data_set}")
        rating_file = data_folder / Path(str(os.listdir(data_folder)[0]))

        # read file
        relevant_data = pd.read_csv(rating_file, header=0, sep='\t',
                                    names=['user', 'item', 'rating'],
                                    dtype={
                                        'user': np.int64,
                                        'item': np.int64,
                                        'rating': np.int64
                                    })

        # re-map user and item ids
        map_ids(relevant_data, ["user", "item"])

        # set default names and data types
        normalize_names(relevant_data, ["user", "item", "rating"])
    elif data_set_category == "LibraryThing":
        data_folder = Path(base_folder / f"LibraryThing/{data_set}")
        rating_file = data_folder / "reviews.json"

        # read file
        final_dict = {'user': [], 'work': [], 'stars': [], 'unixtime': []}
        with open(rating_file) as f:
            for line in f:
                dic = eval(line)
                if all(k in dic for k in ("user", "work", "stars", "unixtime")):
                    final_dict['user'].append(dic['user'])
                    final_dict['work'].append(dic['work'])
                    final_dict['stars'].append(dic['stars'])
                    final_dict['unixtime'].append(dic['unixtime'])
        relevant_data = pd.DataFrame.from_dict(final_dict)

        # remove nan
        relevant_data["unixtime"] = relevant_data["unixtime"].replace(np.nan, -1)

        # re-map user and item ids
        map_ids(relevant_data, ["user", "work"])

        # set default names and data types
        normalize_names(relevant_data, ["user", "work", "stars", "unixtime"])
    elif data_set_category == "MarketBias":
        data_folder = Path(base_folder / f"MarketBias/{data_set}")
        rating_file = data_folder / Path(str(os.listdir(data_folder)[0]))

        if data_set == "amazon":
            # read file
            relevant_data = pd.read_csv(rating_file, header=0, sep=',',
                                        usecols=['user_id', 'item_id', 'rating', 'timestamp'],
                                        dtype={
                                            'user_id': np.int64,
                                            'item_id': np.int64,
                                            'rating': np.float64,
                                            'timestamp': str
                                        })[['user_id', 'item_id', 'rating', 'timestamp']]

            # re-map user and item ids
            map_ids(relevant_data, ["user_id", "item_id"])

            # convert date
            relevant_data["timestamp"] = relevant_data["timestamp"].apply(
                lambda x: time.mktime(datetime.strptime(x, "%Y-%m-%d").timetuple()))

            # set default names and data types
            normalize_names(relevant_data, ["user_id", "item_id", "rating", "timestamp"])
        elif data_set == "modcloth":
            # read file
            relevant_data = pd.read_csv(rating_file, header=0, sep=',',
                                        usecols=['user_id', 'item_id', 'rating', 'timestamp'],
                                        dtype={
                                            'user_id': str,
                                            'item_id': np.int64,
                                            'rating': np.float64,
                                            'timestamp': str
                                        })[['user_id', 'item_id', 'rating', 'timestamp']]

            # re-map user and item ids
            map_ids(relevant_data, ["user_id", "item_id"])

            # convert date
            def s_d(x):
                d = x.split('+')[0]
                if '.' not in d:
                    d += ".000000"
                return d[:-3]

            relevant_data["timestamp"] = relevant_data["timestamp"].apply(lambda x: s_d(x))
            relevant_data["timestamp"] = relevant_data["timestamp"].apply(
                lambda x: time.mktime(datetime.strptime(x, "%Y-%m-%d %H:%M:%S.%f").timetuple()))

            # set default names and data types
            normalize_names(relevant_data, ["user_id", "item_id", "rating", "timestamp"])
    elif data_set_category == "MovieLens":
        data_folder = Path(base_folder / f"MovieLens/{data_set}")

        if data_set == "1m" or data_set == "10m":
            rating_file = data_folder / "ratings.dat"

            # read file
            relevant_data = pd.read_csv(rating_file, header=None, sep='::', engine="python",
                                        names=['user', 'item', 'rating', 'timestamp'],
                                        dtype={
                                            'user': np.int64,
                                            'item': np.int64,
                                            'rating': np.float64,
                                            'timestamp': np.int64
                                        })

            # re-map user and item ids
            map_ids(relevant_data, ["user", "item"])

            # set default names and data types
            normalize_names(relevant_data, ["user", "item", "rating", "timestamp"])
        elif data_set == "20m" or data_set == "25m" or data_set == "latest" or data_set == "latest-small":
            rating_file = data_folder / "ratings.csv"

            # read file
            relevant_data = pd.read_csv(rating_file, header=0, sep=',', engine="python",
                                        names=['userId', 'movieId', 'rating', 'timestamp'],
                                        dtype={
                                            'userId': np.int64,
                                            'movieId': np.int64,
                                            'rating': np.float64,
                                            'timestamp': np.int64
                                        })

            # re-map user and item ids
            map_ids(relevant_data, ["userId", "movieId"])

            # set default names and data types
            normalize_names(relevant_data, ["userId", "movieId", "rating", "timestamp"])
        elif data_set == "100k":
            rating_file = data_folder / "u.data"

            # read file
            relevant_data = pd.read_csv(rating_file, sep='\t', header=None,
                                        names=['user', 'item', 'rating', 'timestamp'],
                                        dtype={
                                            'user': np.int64,
                                            'item': np.int64,
                                            'rating': np.float64,
                                            'timestamp': np.int64})

            # re-map user and item ids
            map_ids(relevant_data, ["user", "item"])

            # set default names and data types
            normalize_names(relevant_data, ["user", "item", "rating", "timestamp"])
    elif data_set_category == "MovieTweetings":
        data_folder = Path(base_folder / f"MovieTweetings/{data_set}")
        rating_file = data_folder / Path(str(os.listdir(data_folder)[0]))

        # read file
        relevant_data = pd.read_csv(rating_file, header=None, sep='::', engine="python",
                                    names=['user', 'item', 'rating', 'timestamp'],
                                    dtype={
                                        'user': np.int64,
                                        'item': np.int64,
                                        'rating': np.float64,
                                        'timestamp': np.int64})

        # re-map user and item ids
        map_ids(relevant_data, ["user", "item"])

        # set default names and data types
        normalize_names(relevant_data, ["user", "item", "rating", "timestamp"])
    elif data_set_category == "Netflix":
        data_folder = Path(base_folder / f"Netflix/{data_set}")
        rating_files = [data_folder / "combined_data_1.txt",
                        data_folder / "combined_data_2.txt",
                        data_folder / "combined_data_3.txt",
                        data_folder / "combined_data_4.txt"]

        # read file
        full_data = []
        current_item = -1
        for rating_file in rating_files:
            with open(rating_file) as file:
                for num, line in enumerate(file):
                    line = line.strip()
                    if line.endswith(':'):
                        current_item = line[:-1]
                    else:
                        user, rating, timestamp = line.split(',')
                        full_data.append({"user": user, "item": current_item, "rating": rating, "timestamp": timestamp})
        relevant_data = pd.DataFrame(full_data)

        # reset index to avoid duplicates
        relevant_data.reset_index(drop=True, inplace=True)

        # re-map user and item ids
        map_ids(relevant_data, ["user", "item"])

        # convert date
        relevant_data["timestamp"] = relevant_data["timestamp"].apply(
            lambda x: time.mktime(datetime.strptime(x, "%Y-%m-%d").timetuple()))

        # set default names and data types
        normalize_names(relevant_data, ["user", "item", "rating", "timestamp"])
    elif data_set_category == "Rekko":
        data_folder = Path(base_folder / f"Rekko/{data_set}")
        rating_file = data_folder / "ratings.csv"

        # read file
        relevant_data = pd.read_csv(rating_file, header=0,
                                    usecols=['user_uid', 'element_uid', 'rating', 'ts'],
                                    dtype={
                                        'user_uid': np.int64,
                                        'element_uid': np.int64,
                                        'rating': np.float64,
                                        'ts': str
                                    })

        # re-map user and item ids
        map_ids(relevant_data, ["user_uid", "element_uid"])

        relevant_data["ts"] = relevant_data["ts"].apply(lambda x: x.split('.')[0] + x.split('.')[1])

        # set default names and data types
        normalize_names(relevant_data, ["user_uid", "element_uid", "rating", "ts"])
    elif data_set_category == "WikiLens":
        data_folder = Path(base_folder / f"WikiLens/{data_set}")
        rating_file = data_folder / Path(str(os.listdir(data_folder)[0]))

        # read file
        relevant_data = pd.read_csv(rating_file, header=0, sep='\t',
                                    names=['user', 'item', 'rating', 'timestamp'],
                                    dtype={
                                        'user': np.int64,
                                        'item': np.int64,
                                        'rating': np.float64,
                                        'timestamp': np.int64
                                    })

        # re-map user and item ids
        map_ids(relevant_data, ["user", "item"])

        # set default names and data types
        normalize_names(relevant_data, ["user", "item", "rating", "timestamp"])
    elif data_set_category == "Yelp":
        data_folder = Path(base_folder / f"Yelp/{data_set}")
        rating_file = data_folder / Path(str(os.listdir(data_folder)[0]))

        # read file
        final_dict = {'user_id': [], 'business_id': [], 'stars': [], 'date': []}
        with open(rating_file) as file:
            for line in file:
                dic = eval(line)
                if all(k in dic for k in ("user_id", "business_id", "stars", "date")):
                    final_dict['user_id'].append(dic['user_id'])
                    final_dict['business_id'].append(dic['business_id'])
                    final_dict['stars'].append(dic['stars'])
                    final_dict['date'].append(dic['date'])
        relevant_data = pd.DataFrame.from_dict(final_dict)

        # re-map user and item ids
        map_ids(relevant_data, ["user_id", "business_id"])

        # convert date
        relevant_data["date"] = relevant_data["date"].apply(
            lambda x: time.mktime(datetime.strptime(x, "%Y-%m-%d %H:%M:%S").timetuple()))

        # set default names and data types
        normalize_names(relevant_data, ["user_id", "business_id", "stars", "date"])
    elif data_set_category == "Yahoo":
        data_folder = Path(base_folder / f"Yahoo/{data_set}")

        if data_set == "movies":
            rating_files = [data_folder / "ydata-ymovies-user-movie-ratings-test-v1_0.txt",
                            data_folder / "ydata-ymovies-user-movie-ratings-train-v1_0.txt"]

            # read file
            relevant_data = pd.DataFrame()
            for rating_file in rating_files:
                ext = pd.read_csv(rating_file, header=None, sep='\t',
                                  names=['user', 'item', 'rating', 'converted_rating'],
                                  usecols=['user', 'item', 'rating'],
                                  dtype={
                                      'user': np.int64,
                                      'item': np.int64,
                                      'rating': np.float64
                                  })[['user', 'item', 'rating']]
                relevant_data = pd.concat([relevant_data, ext])

            # reset index to avoid duplicates
            relevant_data.reset_index(drop=True, inplace=True)

            # re-map user and item ids
            map_ids(relevant_data, ["user", "item"])

            # set default names and data types
            normalize_names(relevant_data, ["user", "item", "rating"])
        elif data_set == "music1":
            rating_file = data_folder / Path(str(os.listdir(data_folder)[0]))

            # read file
            relevant_data = pd.read_csv(rating_file, header=None, sep='\t',
                                        names=['user', 'item', 'rating'],
                                        dtype={
                                            'user': np.int64,
                                            'item': np.int64,
                                            'rating': np.float64
                                        })

            # re-map user and item ids
            map_ids(relevant_data, ["user", "item"])

            # fix never play ratings
            relevant_data.loc[relevant_data["rating"] == 255, ["rating"]] = 0

            # set default names and data types
            normalize_names(relevant_data, ["user", "item", "rating"])
        elif data_set == "music2":
            rating_files = [data_folder / f"test_{i}.txt" for i in range(0, 10)] + \
                           [data_folder / f"train_{i}.txt" for i in range(0, 10)]

            # read file
            relevant_data = pd.DataFrame()
            for rating_file in rating_files:
                ext = pd.read_csv(rating_file, header=None, sep='\t',
                                  names=['user', 'item', 'rating'],
                                  dtype={
                                      'user': np.int64,
                                      'item': np.int64,
                                      'rating': np.float64
                                  })[['user', 'item', 'rating']]
                relevant_data = pd.concat([relevant_data, ext])

            # reset index to avoid duplicates
            relevant_data.reset_index(drop=True, inplace=True)

            # re-map user and item ids
            map_ids(relevant_data, ["user", "item"])

            # set default names and data types
            normalize_names(relevant_data, ["user", "item", "rating"])
        elif data_set == "music3":
            rating_files = [data_folder / "ydata-ymusic-rating-study-v1_0-test.txt",
                            data_folder / "ydata-ymusic-rating-study-v1_0-train.txt"]

            # read file
            relevant_data = pd.DataFrame()
            for rating_file in rating_files:
                ext = pd.read_csv(rating_file, header=None, sep='\t',
                                  names=['user', 'item', 'rating'],
                                  dtype={
                                      'user': np.int64,
                                      'item': np.int64,
                                      'rating': np.float64
                                  })[['user', 'item', 'rating']]
                relevant_data = pd.concat([relevant_data, ext])

            # reset index to avoid duplicates
            relevant_data.reset_index(drop=True, inplace=True)

            # re-map user and item ids
            map_ids(relevant_data, ["user", "item"])

            # set default names and data types
            normalize_names(relevant_data, ["user", "item", "rating"])
    else:
        print(f"No data set category matching {data_set_category} was found.")

    stats = {"num_instances": len(relevant_data),
             "num_users": len(relevant_data["user"].unique()),
             "num_items": len(relevant_data["item"].unique()),
             "min_rating": float(relevant_data["rating"].min()),
             "max_rating": float(relevant_data["rating"].max())}
    stats["sparsity"] = 100 - (stats["num_instances"] * 100) / (stats["num_users"] * stats["num_items"])
    print(f"Number of instances: {stats['num_instances']}\n"
          f"Number of users: {stats['num_users']}\n"
          f"Number of items: {stats['num_items']}\n"
          f"Minimum rating: {stats['min_rating']}\n"
          f"Maximum rating: {stats['max_rating']}\n"
          f"Sparsity: {stats['sparsity']:0.4f}%")
    if "timestamp" in list(relevant_data):
        stats["min_timestamp"] = int(relevant_data["timestamp"].min())
        stats["max_timestamp"] = int(relevant_data["timestamp"].max())
        print(f"Minimum timestamp: {stats['min_timestamp']}\n"
              f"Maximum timestamp: {stats['max_timestamp']}")

    print("Writing processed data to file.")
    if relevant_data is not None:
        if not os.path.exists(f"{base_folder}/.processed/{data_set_category}/{data_set}"):
            os.makedirs(f"{base_folder}/.processed/{data_set_category}/{data_set}")
        relevant_data.to_hdf(f"{base_folder}/.processed/{data_set_category}/{data_set}/processed.h5",
                             key='relevant_data', mode='w')
        stats["time_to_process_and_save"] = time.perf_counter() - start
        print(f"Processed and saved in {stats['time_to_process_and_save']:0.4f} seconds.")
        with open(f"{base_folder}/.processed/{data_set_category}/{data_set}/stats.json", 'w') as json_file:
            json.dump(stats, json_file)

    print(f"Processed {data_set_category} {data_set}.")
    if return_timestamp:
        return relevant_data, stats
    else:
        return relevant_data[["user", "item", "rating"]], stats


def load_data(data_to_process: List[Tuple[str, str]], base_folder: Path, return_timestamp: bool, load_only_stats: bool):
    for data in data_to_process:
        yield load_and_process_data_set(data[0], data[1], base_folder, return_timestamp, load_only_stats, False)
