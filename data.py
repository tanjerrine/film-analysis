import pandas as pd
import os.path
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

RKO_MOVIE_FNAME = "data/RKO_movie_titles_1940s.csv"
PARAMOUNT_MOVIE_FNAME = "data/PARA_movie_titles_1940s.csv"
ALL_MOVIE_FNAME = "data/movie_titles_1940s.csv"

def lead_file_name(genre, studio=None):
    genre_a = genre + "/" if genre else ""
    genre_b = "_" + genre if genre else ""
    p = "" if studio is None else studio + "_"
    n = "data/{}{}40s{}_lead_actor.csv".format(genre_a, p, genre_b)
    return n

def remove_dup(fname, label):
    df = pd.read_csv(fname, index_col=False)
    df = df.drop_duplicates(subset=[label])
    df.to_csv(fname, index=False)

def save_csv(fname, df, override=False):
    if override or not os.path.isfile(fname):
        filepath = Path(fname)  
        filepath.parent.mkdir(parents=True, exist_ok=True)  
        df.to_csv(filepath, index=False)  
        print("Saved {}".format(fname))

def year_film(x, limit=1960):
    if x == '\\N': return False
    return int(x) < limit

def load_data():
    if not os.path.isfile("data/movie_titles.csv"):
        iter_csv = pd.read_csv('title.basics.tsv', delimiter="\t", iterator=True, chunksize=1000, index_col=False)
        df = pd.concat([chunk[chunk['titleType'] == "movie"] for chunk in iter_csv])
        filepath = Path('data/movie_titles.csv')  
        filepath.parent.mkdir(parents=True, exist_ok=True)  
        df.to_csv(filepath, index=False)  
        print("Done with movie_titles.csv")
    
    mfile = ALL_MOVIE_FNAME
    if not os.path.isfile(mfile):
        df = pd.read_csv("data/movie_titles.csv", index_col=False)
        df = df[~df["startYear"].isin(["\\N"])]
        df["startYear"] = df["startYear"].astype(int)
        df = df[df["startYear"]>=1940]
        df = df[df["startYear"]<1950]

        filepath = Path(mfile)  
        filepath.parent.mkdir(parents=True, exist_ok=True)  
        df.to_csv(filepath, index=False) 
#tt0036342
    mlist_40_60 = pd.read_csv(mfile, index_col=False)['tconst']
    if not os.path.isfile("data/title_principals.csv"):
        iter_csv = pd.read_csv('title.principals.tsv', delimiter="\t", iterator=True, chunksize=1000, index_col=False)
        df = pd.concat([chunk[chunk['tconst'].isin(mlist_40_60)] for chunk in iter_csv])
        filepath = Path('data/title_principals.csv')  
        filepath.parent.mkdir(parents=True, exist_ok=True)  
        df.to_csv(filepath, index=False)  
        print("Done with title_principals.csv")

    #get names, excluding people born after 1960
    if not os.path.isfile("data/name_basics.csv"):
        iter_csv = pd.read_csv('name.basics.tsv', delimiter="\t", iterator=True, chunksize=1000, index_col=False)
        df = pd.concat([chunk[chunk['birthYear'].apply(year_film)] for chunk in iter_csv])
        filepath = Path('data/name_basics.csv')  
        filepath.parent.mkdir(parents=True, exist_ok=True)  
        df.to_csv(filepath, index=False)  
        print("Done with name_titles.csv")

def get_lead_female_genre(genre, is_female=True, studio=None):
    genre_str_a = genre + "/" if genre else ""
    genre_str_b = "_" + genre if genre else ""
    if studio is None:
        mtitle_names = 'data/{}movie_titles_1940s{}.csv'.format(genre_str_a, genre_str_b)
    else: 
        mtitle_names = 'data/{}{}_movie_titles_1940s{}.csv'.format(genre_str_a, studio, genre_str_b)
    if not os.path.isfile(mtitle_names):
        if studio is None:
            df = pd.read_csv("data/movie_titles_1940s.csv", index_col=False)
        elif studio == "RKO":
            df = pd.read_csv(RKO_MOVIE_FNAME, index_col=False)
        elif studio == "PARA":
            df = pd.read_csv(PARAMOUNT_MOVIE_FNAME, index_col=False)
        else:
            print("invalid studio name")
            exit(1)
        df = df[df["genres"].str.find(genre) > -1]

        filepath = Path(mtitle_names)  
        filepath.parent.mkdir(parents=True, exist_ok=True)  
        df.to_csv(filepath, index=False) 
        print("made ", mtitle_names)

    df_old = pd.read_csv(mtitle_names, index_col=False)
    # m_list = list(df_old["tconst"])

    principals_name = 'data/{}title_principals_40s{}.csv'.format(genre_str_a, genre_str_b)
    if not os.path.isfile(principals_name):
        # iter_csv = pd.read_csv('title.principals.tsv', delimiter="\t", iterator=True, chunksize=1000, index_col=False)
        # df = pd.concat([chunk[chunk['tconst'].isin(m_list)] for chunk in iter_csv])
        iter_csv = pd.read_csv('data/title_principals.csv', iterator=True, chunksize=1000, index_col=False)
        df = pd.concat([chunk for chunk in iter_csv])
        df = df.merge(df_old, how="inner", on="tconst")
        df.drop(columns=["primaryTitle","startYear","genres"], inplace=True)
        
        filepath = Path(principals_name)  
        filepath.parent.mkdir(parents=True, exist_ok=True)  
        df.to_csv(filepath, index=False)  
        print("Done with {}".format(principals_name))

    def first_female(group):
        if (is_female): females = group[group['category'] == 'actress']
        else: females = group[group['category'] == 'actor']
        if not females.empty:
            return females.iloc[0]  # Select the first female
        else:
            return None
    # get top actress of 40-60s noir
    iter_csv = pd.read_csv(principals_name, iterator=True, chunksize=1000, index_col=False)
    df = pd.concat([chunk.groupby('tconst').apply(first_female) for chunk in iter_csv])
    df.dropna(subset=['tconst'], inplace=True) #bc of chunking, some groups have nan
    df.reset_index(drop=True, inplace=True)
    df.drop(columns=['category', 'job', 'characters'], inplace=True)

    df2 = pd.read_csv(mtitle_names, index_col=False)
    mov_actress_df = pd.merge(df, df2, on='tconst', how='inner')

    df3 = pd.read_csv('data/name_basics.csv', index_col=False)
    mov_lead_birth_df = pd.merge(mov_actress_df, df3, on='nconst', how='inner')
    mov_lead_birth_df["age"] = mov_lead_birth_df["startYear"] - mov_lead_birth_df["birthYear"]
    # print(df_old[~df_old['tconst'].isin(mov_lead_birth_df['tconst'])])

    # print(mov_lead_birth_df)
    f_lead_act_name = 'data/{}{}40s{}_lead_actress.csv'.format(genre_str_a, ("" if studio is None else "{}_".format(studio)), genre_str_b)
    if not is_female:
        f_df = pd.read_csv(f_lead_act_name, index_col=False)
        columns_to_keep = ['tconst', 'ordering', 'nconst', 'primaryName', "knownForTitles", "age"]
        mov_lead_birth_df = mov_lead_birth_df.loc[:, columns_to_keep]
        mov_lead_birth_df=mov_lead_birth_df.merge(f_df, how='inner', on='tconst', suffixes=["_M", "_F"])

    if is_female: lead_act_name = f_lead_act_name
    else: lead_act_name = 'data/{}{}40s{}_lead_actor.csv'.format(genre_str_a, ("" if studio is None else "{}_".format(studio)), genre_str_b)
    filepath = Path(lead_act_name)
    filepath.parent.mkdir(parents=True, exist_ok=True)  
    mov_lead_birth_df = mov_lead_birth_df.drop_duplicates(subset=["tconst"])
    mov_lead_birth_df.to_csv(filepath, index=False)  
    print("Done with {}".format(lead_act_name))

if __name__ == "__main__":

    load_data()

    def save_studio():
        iter_csv = pd.read_csv(ALL_MOVIE_FNAME, iterator=True, chunksize=1000, index_col=False)
        titles_df = pd.concat([chunk for chunk in iter_csv])

        rko_df = pd.read_csv("data/Paramount_1940s.csv", index_col=False)
        rko_df["Year"] = rko_df["Date"].apply(lambda x: x.split(","))
        print(rko_df[rko_df["Year"].apply(lambda x: len(x) < 2)])
        rko_df["Year"] = rko_df["Date"].apply(lambda x: x.split(",")[1].strip(" *"))
        rko_df = pd.merge(rko_df, titles_df, left_on="Title", right_on="primaryTitle") 
        rko_df.drop(columns=["Year", "Date", "Title", "originalTitle", "titleType", "isAdult", "endYear", "runtimeMinutes"], inplace=True)
        print(rko_df)
        save_csv(PARAMOUNT_MOVIE_FNAME, rko_df)

    # save_studio()

    # rko_df = pd.read_csv(RKO_MOVIE_FNAME, index_col=False)
        
    def get_df_list(studio, genres):
        dfs = []

        for g in genres:
            df_temp = pd.read_csv(lead_file_name((g if g != "All" else None), studio), index_col=False)
            # if g == "Romance" and studio == "RKO":
            #     df_temp["age_gap"] = df_temp["age_M"]-df_temp["age_F"]
            #     save_csv(lead_file_name(g, studio), df_temp, True)
            dfs.append(df_temp)
        
        # for i, g in enumerate(genres):
        #     dfs[i]["age_gap"] = dfs[i]["age_M"]-dfs[i]["age_F"]
        #     save_csv(lead_file_name(g, studio), dfs[i])
        
        return dfs

    studio = "PARA"
    datacol = "age_F"
    genres = ["Comedy", "Noir", "Western", "Crime", "Action", "Romance"]

    is_female = True
    mt_df = pd.read_csv(ALL_MOVIE_FNAME, index_col=False)
    def first_female(group):
        if (is_female): females = group[group['category'] == 'actress']
        else: females = group[group['category'] == 'actor']
        if not females.empty:
            return females.iloc[0]  # Select the first female
        else:
            return None
    # get top actress of 40-60s noir
    # iter_csv = pd.read_csv("data/title_principals.csv", iterator=True, chunksize=1000, index_col=False)
    # df = pd.concat([chunk for chunk in iter_csv])
    # df = df.merge(mt_df, how="inner", on="tconst")
    # df.drop(columns=["primaryTitle","startYear","genres"], inplace=True)
    
    # save_csv("data/title_principals_40s.csv", df, True)


    # get_lead_female_genre(None, False, studio=None)

    n_gen = len(genres)

    # one-time fix
    # for genre in genres:
    #     iter_csv = pd.read_csv("data/{}/movie_titles_1940s_{}.csv".format(genre, genre), iterator=True, chunksize=1000, index_col=False)
    #     titles_df = pd.concat([chunk[chunk["startYear"].apply(lambda x: year_film(x, 1950))] for chunk in iter_csv])
    #     save_csv("data/{}/movie_titles_1940s_{}.csv".format(genre, genre), titles_df, True)
        
    # dfs = get_df_list(studio, genres)

    # df1 = pd.read_csv(lead_file_name("Comedy", studio), index_col=False)

    # df2 = pd.read_csv(lead_file_name("Noir", studio), index_col=False)

    # df3 = pd.read_csv(lead_file_name("Western", studio), index_col=False)
    # df4 = pd.read_csv(lead_file_name("Crime", "RKO"), index_col=False)

    
    # print("{}, {}, {}, {}".format(np.mean((df1["age_M"]-df1["age_F"])), np.mean((df2["age_M"]-df2["age_F"])), np.mean((df3["age_M"]-df3["age_F"])), np.mean((df4["age_M"]-df4["age_F"]))))
    # all 40-60: 9.675736961451246, 9.289637952559302, 10.129068462401795, 9.417910447761194, 8.37726913970008
    # all 40-50: 9.681643132220795, 9.921717171717171, 9.544433094994893, 9.417910447761194, 8.663917525773195
    # RKO 40s: 11.097014925373134, 9.148148148148149, 6.2745098039215685, 9.417910447761194, 12.333333333333334
    # fig, ax = plt.subplots()
    # ax.hist(df3["age_M"]-df3["age_F"], linewidth=0.5, edgecolor="white")
    # plt.show()
    dfs_all = get_df_list(None, genres)
    dfs_RKO = get_df_list("RKO", genres)
    dfs_PARA = get_df_list("PARA", genres)
#######################################################
    sorted_dfs_all = []
    for i, dfi in enumerate(dfs_PARA):
        old = dfi[dfi["age_F"] >= 40]
        dfi["prop_old"] = len(old)/len(dfi)
        # print("{}: {:.3f}".format(genres[i],len(old)/len(dfi)))
        # print(len(dfi))
        sorted_dfs_all.append(sorted(dfi["age_F"]))
    # print("+++++++++++++++++++")
    for i, dfi in enumerate(dfs_RKO):
        old = dfi[dfi["age_F"] >= 40]
        dfi["prop_old"] = len(old)/len(dfi)
        # print("{}: {:.3f}".format(genres[i],len(old)/len(dfi)))
        # print(len(dfi))
        sorted_dfs_all.append(sorted(dfi["age_F"]))
    # print("+++++++++++++++++++")
    for i, dfi in enumerate(dfs_all):
        old = dfi[dfi["age_F"] >= 40]
        dfi["prop_old"] = len(old)/len(dfi)
        # print("{}: {:.3f}".format(genres[i],len(old)/len(dfi)))
        sorted_dfs_all.append(sorted(dfi["age_F"]))

    # sorted1 = sorted(df1["age_F"])
    # sorted2 = sorted(df2["age_F"])
    # sorted3 = sorted(df3["age_F"])
    # sorted4 = sorted(df4["age_F"])

    # fig, ax = plt.subplots()
    # for i, g in enumerate(genres):
    #     ax.ecdf(sorted_dfs_all[i], label = g)
    # # ax.ecdf(sorted1, label="comedy")
    # # ax.ecdf(sorted2, label="noir")
    # # ax.ecdf(sorted3, label="western")
    # # ax.ecdf(sorted4, label="crime")
    # plt.legend()
    # ax1 = plt.subplot(1, 3, 1)
    # ax1.ecdf(sorted1)
    # ax2 = plt.subplot(1, 3, 2)
    # ax2.ecdf(sorted2)
    # ax3 = plt.subplot(1, 3, 3)
    # ax3.ecdf(sorted3)
    # plt.show()
    # plt.savefig(fname='plots/cdf_ages_f_PARA')
    # plt.close()
    print(max(dfs_all[5]["age_F"]))

################################################

    def grouped_barplots():
        # fig, ax = plt.subplots()

        # data_group1 = [np.mean(dfi["age_M"]-dfi["age_F"]) for dfi in dfs_all]
        # data_group2 = [np.mean(dfi["age_M"]-dfi["age_F"]) for dfi in dfs_RKO]
        data_group1 = [np.mean(dfi[datacol]) for dfi in dfs_all]
        data_group2 = [np.mean(dfi[datacol]) for dfi in dfs_RKO]
        data_group3 = [np.mean(dfi[datacol]) for dfi in dfs_PARA]
        avg_f_df = pd.DataFrame({"All": data_group1, "RKO": data_group2, "PARA": data_group3}, index=genres)
        ax = avg_f_df.plot.bar(rot=0)
        plt.xlabel('Genre')
        plt.ylabel('Proportion of Films')
        plt.title("Proportion of Films with Lead Actress Older than 40, by Genre")
        # plt.show()
        plt.savefig(fname='plots/prop_old_grouped_barplot')
        plt.close()
    # grouped_barplots()

    # print(df2[datacol])
    def box_plot():
        plt.figure(figsize=(8,6))
        # plt.boxplot([dfi[datacol] for dfi in dfs_all], patch_artist=True, labels=genres ,medianprops={"color": "green", "linewidth": 0.5},
        #                 whiskerprops={"color": "C0", "linewidth": 1.5},
        #                 capprops={"color": "C0", "linewidth": 1.5})
                        # , positions=[2, 4, 6], widths=1.5, patch_artist=True,
                        # showmeans=False, showfliers=False,
                        # medianprops={"color": "white", "linewidth": 0.5},
                        # boxprops={"facecolor": "C0", "edgecolor": "white",
                        #         "linewidth": 0.5},
                        # whiskerprops={"color": "C0", "linewidth": 1.5},
                        # capprops={"color": "C0", "linewidth": 1.5})

        # ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
        #     ylim=(0, 8), yticks=np.arange(1, 8))
        plt.grid(True, linestyle='--', alpha=0.7)
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        dcolors = [c for c in colors for _ in (0,1)]
        plot_boxplt = plt.boxplot([dfi[datacol] for dfi in dfs_all], patch_artist=True, labels=genres ,medianprops={"color": "white", "linewidth": 0.75},
                        whiskerprops={"linewidth": 1.5}, boxprops={"facecolor": "C0", "edgecolor": "white", "linewidth": 0.5},
                        capprops={"color": "C0", "linewidth": 1.5}, flierprops = dict(marker='.'), showmeans=False)
        for patch, color in zip(plot_boxplt['boxes'], colors):
            patch.set_facecolor(color)
        for cap, whisker, dcolor in zip(plot_boxplt['caps'], plot_boxplt['whiskers'], dcolors):
            whisker.set(color=dcolor)
            cap.set(color=dcolor)

        plt.title("Age Distribution of Lead Female Actor of 1940s films by Genre")
        plt.ylabel("Age")
        plt.xlabel("Genre")
        # plt.show()
        # plt.close()
        plt.savefig(fname='plots/All_F_age_boxplots')
    box_plot()
#########################################
    # plt.close()
    # fig, ax = plt.subplots()
    # # df = pd.read_csv("data/Comedy/RKO_40-60_Comedy_lead_actor.csv")
    # data = df1["ordering_F"]
    # ax.hist(data, bins=range(1,int(max(data)+1)), linewidth=0.5, edgecolor="white")
    # plt.axvline(data.mean(), color='red', linestyle='dashed', linewidth=2)
    # # ax.hist(df["ordering"], bins=range(1,int(max(df["ordering"])+1)), linewidth=0.5, edgecolor="white")
    # # plt.axvline(df['ordering'].mean(), color='red', linestyle='dashed', linewidth=2)
    # print(data.mean())
    # plt.show()
    # plt.savefig(fname='RKO_noir_age_diff')
    # Orderings:
    #    RKO: 2.246268656716418, 2, 2.5686274509803924, 2.1791044776119404
