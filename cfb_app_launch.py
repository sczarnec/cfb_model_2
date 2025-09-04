import streamlit as st
import pandas as pd
import numpy as np
import csv
import math
import xgboost as xgb
from streamlit_js_eval import streamlit_js_eval, get_user_agent
import random
import itertools

# set page config
st.set_page_config(layout="wide")



#### USER VIEW DETECTION

# read view size in user's browser
width = streamlit_js_eval(js_expressions="window.innerWidth", key="get_width")
ua = get_user_agent() or ""
# create benchmark for mobile vs desktop
is_mobile = (width is not None and width < 768) or ("Mobi" in ua)
# set to true for testing
#is_mobile = True




#### READ IN CSVs


# load historical betting data
historical_data = pd.read_csv("winnings_calc_current.csv", encoding="utf-8", sep=",", header=0)

# load theoretical data ready for theor game predictions
theor_prepped = pd.read_csv("theor_prepped.csv", encoding="utf-8", sep=",", header=0)
theor_prepped = theor_prepped.drop(theor_prepped.columns[0], axis=1)
# load aggregated theoretical data ready for power rankings
theor_agg = pd.read_csv("theoretical_agg.csv", encoding="utf-8", sep=",", header=0)
theor_agg = theor_agg.drop(theor_agg.columns[0], axis=1)

# load  models and variables
# cover
cover_model = xgb.Booster()
cover_model.load_model("most_recent_actual_t1_cover_model.model")
cover_vars = pd.read_csv("most_recent_actual_t1_covermodel_var_list.csv", encoding="utf-8", header=0).iloc[:, 1]
cover_vars = cover_vars.astype(str).str.strip().tolist()
# wp
wp_model = xgb.Booster()
wp_model.load_model("most_recent_t1_win_model.model")
wp_vars = pd.read_csv("most_recent_t1_winmodel_var_list.csv", encoding="utf-8", header=0).iloc[:, 1]
wp_vars = wp_vars.astype(str).str.strip().tolist()
# over/under
over_model = xgb.Booster()
over_model.load_model("most_recent_actual_over_model.model")
over_vars = pd.read_csv("most_recent_actual_overmodel_var_list.csv", encoding="utf-8", header=0).iloc[:, 1]
over_vars = over_vars.astype(str).str.strip().tolist()

# load in data ready for this week's predictions
sample_data = pd.read_csv("model_prepped_this_week.csv", encoding="utf-8", sep=",", header=0)

# load team info data
team_info = pd.read_csv("team_info.csv", encoding="utf-8", sep=",", header=0)

# load ranking data
poll_rankings = pd.read_csv("this_week_ranks.csv", encoding="utf-8", sep=",", header=0)

# extract this week from data
this_week = theor_prepped['week'].max()





#### FUNCTIONS WE'LL NEED

# convert book ml to win prob, create some related cols
def moneyline_to_winprob(ml):
    """
    Args:
    - ml: moneyline as numeric

    Output:
    - moneyline in wp format as prob between 0-1
    """
    if ml < 0:
        # negative ml
        return round(-ml / (-ml + 100), 4)
    else:
        # positive ml
        return round(100 / (ml + 100), 4)


# convert win prob to ml
def winprob_to_moneyline(prob):
    """
    Args:
    - prob: win prob in 0-1 numeric format

    Output:
    - moneyline in numeric format
    """
    if prob >= 0.5:
        return round(-100 * prob / (1 - prob), 0)
    else:
        return round(100 * (1 - prob) / prob, 0)




# predict using a given model
def predict_with_model(model, var_list, data):
    """
    Args:
    - model: model to use
    - var_list: selected features
    - data: dataset for prediction

    Output
    - predicions
    """

    # python handles xgboost differently than r, need to manually stop at ntree that R stopped at
    best_ntree = int(model.attr("best_ntreelimit"))
    dmatrix = xgb.DMatrix(data[var_list])
    # predict
    return model.predict(dmatrix,iteration_range=(0, best_ntree))



# function to make sure games are on same side (there are two predictions for each game with teams reversed in t1_team and t2_team, this is to bring to one)
def normalize_row(row):
    """
    Args:
    - row = row of data in a dataframe

    Output:
    - modified row (either flipped or not)
    """


    # keep this game if t1_team is home or it's neutral and t1_team>t2_team alphabetically 
    keep_original = ((row["t1_home"] == 1) or ((row["neutral_site"] == 1) and (row["t1_team"] > row["t2_team"])))
    # if meets criteria, keep row
    if keep_original:
        return row
    # if not, flip for aggregation
    else:
        row["t1_team"], row["t2_team"] = row["t2_team"], row["t1_team"]
        row["cover_prob_pred"] = 1-row["cover_prob_pred"]
        row["win_prob_pred"] = 1 - row["win_prob_pred"]
        return row



# big ass function to prep this week's data, predict, and get in format for our app
def make_merged_predictions(data):
        

        ## Predictions


        # predict the 3 outcomes for this week
        cp_preds = predict_with_model(cover_model, cover_vars, data)
        wp_preds = predict_with_model(wp_model, wp_vars, data)
        op_preds = predict_with_model(over_model, over_vars, data)

        # results dfs
        results_df = data[["t1_team", "t2_team", "t1_home", "neutral_site", "game_id"]].copy()
        results_df["cover_prob_pred"] = cp_preds
        results_df["win_prob_pred"] = wp_preds
        results_df["over_prob_pred"] = op_preds


        # use row flip function
        normalized_df = results_df.apply(normalize_row, axis=1)

        # aggregate the two predictions of each game
        aggregated_df = normalized_df.groupby(
            ["game_id", "t1_team", "t2_team", "neutral_site"], as_index=False
        ).agg({
            "cover_prob_pred": "mean",
            "win_prob_pred": "mean",
            "over_prob_pred": "mean"
        })


        # create t2 win prob col
        aggregated_df["t2_win_prob_pred"] = (1- aggregated_df["win_prob_pred"])

        # change predicted ml to win prob
        aggregated_df["t1_pred_moneyline"] = aggregated_df["win_prob_pred"].apply(winprob_to_moneyline)
        aggregated_df["t2_pred_moneyline"] = aggregated_df["t2_win_prob_pred"].apply(winprob_to_moneyline)

        # change col names to be better for UI, select needed cols
        final_predictions_df = aggregated_df.rename(columns={
            "t1_team": "Home",
            "t2_team": "Away",
            "neutral_site": "Neutral?",
            "cover_prob_pred": "Pred Home Cover Prob",
            "t1_pred_moneyline": "Pred Home Moneyline",
            "t2_pred_moneyline": "Pred Away Moneyline",
            "over_prob_pred": "Pred Over Prob",
            "win_prob_pred": "Pred Home Win Prob",
            "t2_win_prob_pred": "Pred Away Win Prob"
        })[["Home", "Away", "Neutral?", "Pred Home Cover Prob", "Pred Home Moneyline", "Pred Away Moneyline", "Pred Over Prob", "Pred Home Win Prob",
            "Pred Away Win Prob", "game_id"]]


        # join in more info we need for betting comparisons
        betting_join = data[["t1_book_spread", "book_over_under", "t1_moneyline", "t2_moneyline", "t1_conference", "t2_conference", "game_id", "t1_team"]]
        merged_predictions = final_predictions_df.merge(betting_join, left_on=["game_id","Home"], right_on=["game_id", "t1_team"], how="left")




        ## Betting Recs


        # Spread

        # betting recs
        merged_predictions.loc[merged_predictions['Pred Home Cover Prob']>.5,"Spread Bet on:"] = merged_predictions['Home']
        merged_predictions.loc[merged_predictions['Pred Home Cover Prob']<=.5,"Spread Bet on:"] = merged_predictions['Away']
        merged_predictions["Spread Value"] = round(abs(merged_predictions['Pred Home Cover Prob'] - .5)*100,2)




        # Moneyline


        # change ml to win prob
        merged_predictions["t1_ml_prob"] = merged_predictions["t1_moneyline"].apply(moneyline_to_winprob)
        merged_predictions["t2_ml_prob"] = merged_predictions["t2_moneyline"].apply(moneyline_to_winprob)

        # betting recs
        conditions_ml = [
            merged_predictions["t1_ml_prob"]  < merged_predictions["Pred Home Win Prob"],
            merged_predictions["t2_ml_prob"]  < merged_predictions["Pred Away Win Prob"]
        ]
        conditions_ml_val = [
            merged_predictions["Pred Away Win Prob"] - merged_predictions["t2_ml_prob"] < merged_predictions["Pred Home Win Prob"] - merged_predictions["t1_ml_prob"],
            merged_predictions["Pred Home Win Prob"] - merged_predictions["t1_ml_prob"] < merged_predictions["Pred Away Win Prob"] - merged_predictions["t2_ml_prob"]
        ]
        choices_ml_bet = [
            merged_predictions["Home"],
            merged_predictions["Away"]
        ]
        choices_ml_val = [
            merged_predictions["Pred Home Win Prob"] - merged_predictions["t1_ml_prob"],
            merged_predictions["Pred Away Win Prob"] - merged_predictions["t2_ml_prob"]
        ]
        choices_ml_losing_val = [
            merged_predictions["Pred Away Win Prob"] - merged_predictions["t2_ml_prob"],
            merged_predictions["Pred Home Win Prob"] - merged_predictions["t1_ml_prob"]
        ]


        # who to bet on
        merged_predictions["ML Bet on:"] = np.select(conditions_ml, choices_ml_bet, default="Neither")
        # value on better side
        merged_predictions["ML Value"] = np.select(conditions_ml_val, choices_ml_val, default=0)
        # value on worse side
        merged_predictions["ML Losing Value"] = np.select(conditions_ml_val, choices_ml_losing_val, default=0)
        # round
        merged_predictions["ML Value"] = round(merged_predictions["ML Value"]*100,2)
        merged_predictions["ML Losing Value"] = round(merged_predictions["ML Losing Value"]*100,2)




        # Over/Under

        # betting recs
        merged_predictions.loc[merged_predictions["Pred Over Prob"]>.5,"O/U Bet on:"] = "Over"
        merged_predictions.loc[merged_predictions["Pred Over Prob"]<.5,"O/U Bet on:"] = "Under"
        merged_predictions["O/U Value"] = round(abs(merged_predictions["Pred Over Prob"] - .5)*100,2)
            


        # make cols cleaner
        merged_predictions["Pred Home Win Prob"] = round(merged_predictions["Pred Home Win Prob"]*100,2)
        merged_predictions["Pred Away Win Prob"] = round(merged_predictions["Pred Away Win Prob"]*100,2)
        merged_predictions["t1_ml_prob"] = round(merged_predictions["t1_ml_prob"]*100,2)
        merged_predictions["t2_ml_prob"] = round(merged_predictions["t2_ml_prob"]*100,2)
        merged_predictions["Pred Over Prob"] = round(merged_predictions["Pred Over Prob"]*100,2)
        merged_predictions["Pred Home Cover Prob"] = round(merged_predictions["Pred Home Cover Prob"]*100,2)
        # matchup col
        merged_predictions["matchup"] = merged_predictions["Away"] + " vs " + merged_predictions["Home"]
        # add logos and colors
        merged_predictions = merged_predictions.merge(team_info, left_on = ["Home"], right_on = ["school"], how = "left")
        merged_predictions = merged_predictions.rename(columns = {"color":"home_color", "logo": "home_logo"})
        merged_predictions = merged_predictions.drop(["school"], axis = 1)
        merged_predictions = merged_predictions.merge(team_info, left_on = ["Away"], right_on = ["school"], how = "left")
        merged_predictions = merged_predictions.rename(columns = {"color":"away_color", "logo": "away_logo"})
        merged_predictions = merged_predictions.drop(["school"], axis = 1)




        

        ## Interpolation for Expected Returns


        # nope, this function still isn't over

        # keep going with expected returns determined from linear interpolation and midpoints
        def interpolate_from_buckets(bin_edges, bucket_vals, x, moneyline=False):
            """
            Args:
            - bin_edges: upper and lower limits of the bins
            - bucket_vals: bin expected returns for the midpoints between the edges
            - x: values we're mapping to ('value' column for the bets)
            - moneyline: boolean, return NA if value under 0 because we shouldn't be betting

            Output:
            - array of interpolated values
            """

            # initialize edges, values, and the bet values
            bin_edges = np.asarray(bin_edges, dtype=float)
            bucket_vals = np.asarray(bucket_vals, dtype=float)
            x = np.atleast_1d(x).astype(float)
            
            # find midpoints
            mids = (bin_edges[:-1] + bin_edges[1:]) / 2.0
            
            # interpolate
            out = np.interp(x, mids, bucket_vals, 
                            left=bucket_vals[0], right=bucket_vals[-1])
            
            # moneyline rule
            if moneyline:
                out[x <= 0] = np.nan
            
            # treat NAs
            out[np.isnan(x)] = np.nan
            
            return out
        


        # Spread Interpolation

        spread_edges = [0, 1, 2, 3, 6, 8, 15]
        spread_vals  = [-2, 0, 1, 2, 3, 4]

        merged_predictions["Spread Exp Return"] = interpolate_from_buckets(
            spread_edges, spread_vals, merged_predictions["Spread Value"].values
        )
        merged_predictions["Spread Exp Return"] = round(merged_predictions["Spread Exp Return"],2)




        # Moneyline interpolation
        ml_edges = [0, 1, 2, 4, 6, 9, 15, 20]
        ml_vals  = [-3, 1, 2, 3, 4, 5, 5.5]  

        merged_predictions["ML Exp Return"] = interpolate_from_buckets(
            ml_edges, ml_vals, merged_predictions["ML Value"].values, moneyline=True
        )
        merged_predictions["ML Exp Return"] = round(merged_predictions["ML Exp Return"],2)





        ## Formatting Stuff


        # Spread

        # flip signs for spread, depending on who is bet on
        merged_predictions["spread_temp"] = np.where(
            merged_predictions["Spread Bet on:"].eq(merged_predictions["Home"]),
            pd.to_numeric(merged_predictions["t1_book_spread"], errors="coerce"),      
            -pd.to_numeric(merged_predictions["t1_book_spread"], errors="coerce"),   
        )

        # format with + if needed
        merged_predictions["spread_fmt"] = merged_predictions["spread_temp"].map(lambda x: f"{x:+.1f}")

        # combine into string
        merged_predictions["Spread Bet on: "] = merged_predictions["Spread Bet on:"] + " " + merged_predictions["spread_fmt"]



        # Moneyline

        # find moneyline, depending who is bet on
        merged_predictions["ml_temp"] = np.where(
            merged_predictions["ML Bet on:"].eq(merged_predictions["Home"]),
            pd.to_numeric(merged_predictions["t1_moneyline"], errors="coerce"),
            np.where(
                merged_predictions["ML Bet on:"].eq(merged_predictions["Away"]),
                pd.to_numeric(merged_predictions["t2_moneyline"], errors="coerce"),
                np.nan
            ),
        )

        # format with + if needed
        merged_predictions["ml_fmt"] = merged_predictions["ml_temp"].map(lambda x: f"{x:+.0f}" if pd.notnull(x) else "")

        # combine into string
        merged_predictions["ML Bet on: "] = merged_predictions["ML Bet on:"] + " " + merged_predictions["ml_fmt"]




        # Over/Under

        # format as string
        merged_predictions["ou_fmt"] = merged_predictions["book_over_under"].astype(str)
        merged_predictions["O/U Bet on: "] = merged_predictions["O/U Bet on:"] + " " + merged_predictions["ou_fmt"]



        # Last Stuff

        # drop cols we don't need
        merged_predictions = merged_predictions.drop(["spread_temp", "spread_fmt", "ml_temp", "ml_fmt", "ou_fmt"], axis=1)

 
        # final df with cleaner names
        merged_predictions = merged_predictions.rename(columns={
            "t1_book_spread": "Book Home Spread",
            "t1_moneyline": "Book Home Moneyline",
            "t2_moneyline": "Book Away Moneyline",
            "t1_ml_prob": "Book Home Win Prob",
            "t2_ml_prob": "Book Away Win Prob",
            "book_over_under": "Book O/U",
            "t1_conference": "Home Conf",
            "t2_conference": "Away Conf"
        })





        # finally done
        return merged_predictions        





#### PREDICT GAMES FOR THIS WEEK


# call big ass function with anti-climatic single argument
merged_predictions = make_merged_predictions(sample_data)





#### THEORETICAL MODIFICATIONS

# we used predicted sportsbook spreads for theoretical games
# these matchups we actually do know the spread for so let's use that, modified slightly for H/A/N


# mark what the original game location was in df
sample_data_cp = sample_data.copy()
sample_data_cp["og_han"] = 0
sample_data_cp["real_game"] = 0
sample_data_cp.loc[sample_data_cp["t1_home"] == 1, "og_han"] = 1
sample_data_cp.loc[(sample_data_cp["t1_home"] == 0) & (sample_data_cp["neutral_site"] == 0), "og_han"] = 2
sample_data_cp.loc[sample_data_cp["neutral_site"] == 1, "og_han"] = 3

# home only df, modify book spread based on original location
t1_home_only = sample_data_cp.copy()
t1_home_only["t1_home"] = 1
t1_home_only["neutral_site"] = 0
t1_home_only.loc[t1_home_only["og_han"]==2, "t1_book_spread"] = t1_home_only["t1_book_spread"] - 2.5
t1_home_only.loc[t1_home_only["og_han"]==3, "t1_book_spread"] = t1_home_only["t1_book_spread"] - 1.25
t1_home_only.loc[t1_home_only["og_han"]==1, "real_game"] = 1
t1_home_only.drop(["og_han"], axis=1)

# away only df, modify book spread based on original location
t1_away_only = sample_data_cp.copy()
t1_away_only["t1_home"] = 0
t1_away_only["neutral_site"] = 0
t1_away_only.loc[t1_away_only["og_han"]==1, "t1_book_spread"] = t1_away_only["t1_book_spread"] + 2.5
t1_away_only.loc[t1_away_only["og_han"]==3, "t1_book_spread"] = t1_away_only["t1_book_spread"] + 1.25
t1_away_only.loc[t1_away_only["og_han"]==2, "real_game"] = 1
t1_away_only.drop(["og_han"], axis=1)

# neutral only df, modify book spread based on original location
t1_neutral_only = sample_data_cp.copy()
t1_neutral_only["t1_home"] = 0
t1_neutral_only["neutral_site"] = 1
t1_neutral_only.loc[t1_neutral_only["og_han"]==1, "t1_book_spread"] = t1_neutral_only["t1_book_spread"] + 1.25
t1_neutral_only.loc[t1_neutral_only["og_han"]==2, "t1_book_spread"] = t1_neutral_only["t1_book_spread"] - 1.25
t1_neutral_only.loc[t1_neutral_only["og_han"]==3, "real_game"] = 1
t1_neutral_only.drop(["og_han"],axis=1)

# predict the three dfs
wp_preds_ho = predict_with_model(wp_model, wp_vars, t1_home_only)
wp_preds_ao = predict_with_model(wp_model, wp_vars, t1_away_only)
wp_preds_no = predict_with_model(wp_model, wp_vars, t1_neutral_only)

# create results dfs
t1_home_only_short = t1_home_only[["t1_team", "t2_team", "t1_home", "neutral_site", "game_id", "real_game"]]
results_df_ho = t1_home_only_short.copy()
results_df_ho["wp_pred"] = wp_preds_ho
t1_away_only_short = t1_away_only[["t1_team", "t2_team", "t1_home", "neutral_site", "game_id", "real_game"]]
results_df_ao = t1_away_only_short.copy()
results_df_ao["wp_pred"] = wp_preds_ao
t1_neutral_only_short = t1_neutral_only[["t1_team", "t2_team", "t1_home", "neutral_site", "game_id", "real_game"]]
results_df_no = t1_neutral_only_short.copy()
results_df_no["wp_pred"] = wp_preds_no
# concatenate dfs together
results_tw_theor = pd.concat([results_df_ho, results_df_ao, results_df_no], axis=0, ignore_index=True)


# bring gameids to same side
# new normalize row func
def normalize_row(row):
    keep_original = ((row["t1_home"] == 1) or ((row["neutral_site"] == 1) and (row["t1_team"] > row["t2_team"])))
    if keep_original:
        return row
    else:
        row["t1_team"], row["t2_team"] = row["t2_team"], row["t1_team"]
        row["wp_pred"] = 1-row["wp_pred"]
        if row["neutral_site"]==1:
            row["t1_home"] = 0
        else:
            row["t1_home"] = (1-row["t1_home"])
        return row
normalized_df_tw_theor = results_tw_theor.apply(normalize_row, axis=1)

# aggregate each pair of gameids
aggregated_df_tw_theor = normalized_df_tw_theor.groupby(
    ["game_id", "t1_team", "t2_team", "t1_home", "neutral_site", "real_game"], as_index=False
).agg({
    "wp_pred": "mean"
})
# make cols cleaner
aggregated_df_tw_theor["wp_pred"] = round(aggregated_df_tw_theor["wp_pred"],4)
aggregated_df_tw_theor["new_win"] = (aggregated_df_tw_theor["wp_pred"] > .5).astype(int)


# merge to theoretical data, if in dataset
theor_prepped = theor_prepped.merge(aggregated_df_tw_theor, on=["t1_team", "t2_team", "t1_home", "neutral_site"],
                                     how="left")


# change cols in theoretical data for pdiff and win to that of new predictions for those rows
theor_prepped.loc[theor_prepped["wp_pred"].notna(), "pred_t1_wp"] = theor_prepped["wp_pred"]
theor_prepped.loc[theor_prepped["new_win"].notna(), "t1_win"] = theor_prepped["new_win"]
theor_prepped = theor_prepped.drop(["wp_pred", "new_win"], axis =1)

# change who the winning and losing team cols are in case that changed
theor_prepped.loc[theor_prepped["pred_t1_wp"]>.5, "winning_team"] = theor_prepped["t1_team"]
theor_prepped.loc[theor_prepped["pred_t1_wp"]<=.5, "winning_team"] = theor_prepped["t2_team"]
theor_prepped.loc[theor_prepped["pred_t1_wp"]<=.5, "losing_team"] = theor_prepped["t1_team"]
theor_prepped.loc[theor_prepped["pred_t1_wp"]>.5, "losing_team"] = theor_prepped["t2_team"]
theor_prepped.loc[theor_prepped["pred_t1_wp"]>.5, "winning_logo"] = theor_prepped["t1_logo"]
theor_prepped.loc[theor_prepped["pred_t1_wp"]<=.5, "winning_logo"] = theor_prepped["t2_logo"]
theor_prepped.loc[theor_prepped["pred_t1_wp"]<=.5, "losing_logo"] = theor_prepped["t1_logo"]
theor_prepped.loc[theor_prepped["pred_t1_wp"]>.5, "losing_logo"] = theor_prepped["t2_logo"]
theor_prepped.loc[theor_prepped["pred_t1_wp"]>.5, "winning_color"] = theor_prepped["t1_color"]
theor_prepped.loc[theor_prepped["pred_t1_wp"]<=.5, "winning_color"] = theor_prepped["t2_color"]
theor_prepped.loc[theor_prepped["pred_t1_wp"]<=.5, "losing_color"] = theor_prepped["t1_color"]
theor_prepped.loc[theor_prepped["pred_t1_wp"]>.5, "losing_color"] = theor_prepped["t2_color"]

theor_prepped["pred_t1_wp"] = round(theor_prepped["pred_t1_wp"]*100,2)
theor_prepped.loc[theor_prepped["pred_t1_wp"]>50, "pred_winner_wp"] = theor_prepped["pred_t1_wp"]
theor_prepped.loc[theor_prepped["pred_t1_wp"]<=50, "pred_winner_wp"] = 100-theor_prepped["pred_t1_wp"]


# set page width set
st.set_page_config(layout="wide")





#### ABOUT PAGE

def about_page():
  
  st.title('About Czar College Football')
  
  st.markdown(
          """
          <style>
          .custom-font {
              font-size: 20px;
          }
          </style>
          <div class="custom-font">This site is built around the use of an XGBoost model to predict
          cover probability, win probability, and over probability for FBS College Football games. 
          The goal is to predict future outcomes for fun and try to beat the accuracy of sportsbook models.</div>
          """,
          unsafe_allow_html=True
      )
      
  st.write("  ")
  st.write("  ")
      
  st.markdown(
          """
          <style>
          .custom-font {
              font-size: 20px;
          }
          </style>
          <div class="custom-font">As of 2025, we are now using model v2.0. This model uses a rolling window
          to re-train and re-test performance each week, allowing it to be adaptable. This also lets us examine historical
          performance on all available games. </div>
          """,
          unsafe_allow_html=True
      )


  st.write("  ")
  st.write("  ")
      
  st.markdown(
          """
          <style>
          .custom-font {
              font-size: 20px;
          }
          </style>
          <div class="custom-font">As a note, this was built for entertainment purposes. While it's
          fun to use this information to try to beat the books, performance is merely estimated, not guaranteed.
          We are not responsible for bet outcomes.</div>
          """,
          unsafe_allow_html=True
      ) 
      
  st.write("  ")
  st.write("  ")
      

  
   
  




#### HISTORICAL RESULTS PAGE

def historical_results_page():
    
    st.title('Model Performance on Historical Test Data')

    st.write("This is how well our model would perform on previous games over the last few years versus how well a random 50/50 guesser would perform.")


    # initialize historical df
    df = historical_data.copy()
    df.columns = df.columns.str.strip()



    ## Filters

    # format season for filter
    df['season'] = df['season'].astype(str)

    # create value column for filter
    df['cover_prob_from_50'] = abs(df['pred_t1_cp'] - .5)

    # get unique teams for filter
    unique_teams = pd.unique(df[['t1_team', 't2_team']].values.ravel('K'))
    unique_teams_sorted = sorted(unique_teams)

    # get unique weeks for filter
    unique_weeks = sorted(df['week'].unique())

    # get unique seasons
    unique_seasons = sorted(df['season'].unique(), reverse=True)

    # get unique conferences
    unique_conf = pd.unique(df[['t1_conference', 't2_conference']].values.ravel('K'))
    unique_conf_sorted = sorted(unique_conf)    





    ##  Create columns for layout

    col1, col2, col3, col4 = st.columns([3, 1, 8, 5])  



    ## Filter Col
    with col1:
      
        # Filters
        st.write("### Filters")

        # teams
        team_options = st.selectbox(
            "Team", 
            options=["All"] + unique_teams_sorted,  
            index=0  
        )
        # weeks
        week_options = st.multiselect(
            "Week", 
            options=["All"] + unique_weeks,  
            default=["All"]  
        )
        # season
        season_options = st.multiselect(
            "Season", 
            options=["All"] + unique_seasons,  
            default=["All"]  
        )
        # conference
        conf_options = st.selectbox(
            "Conference", 
            options=["All"] + unique_conf_sorted,  
            index = 0 
        )        
        # cover confidence
        cover_conf_lower, cover_conf_upper = st.slider(
            "Cover Confidence",
            min_value=float(int(math.floor(df['cover_confidence'].min()))),
            max_value=float(round(df['cover_confidence'].max(),2))+.01,
            value=(float((int(math.floor(df['cover_confidence'].min())))), float(round(df['cover_confidence'].max(),2))+.01),
            step = .01
        )
        # ML value
        ml_value_lower, ml_value_upper = st.slider(
            "ML Value",
            min_value=float(round(df['ml_value'].min(),2))-.01,
            max_value=float(round(df['ml_value'].max(),2))+.01,
            value=(float(round(df['ml_value'].min(),2))-.01 , float(round(df['ml_value'].max(),2))+.01),
            step=.01
        )
        # over/under confidence
        ou_conf_lower, ou_conf_upper = st.slider(
            "O/U Confidence",
            min_value=float(int(math.floor(df['ou_confidence'].min()))),
            max_value=float(round(df['ou_confidence'].max(),2))+.01,
            value=(float((int(math.floor(df['ou_confidence'].min())))), float(round(df['ou_confidence'].max(),2))+.01),
            step = .01
        )        
        # book spread
        book_sp_lower, book_sp_upper = st.slider(
            "Home Book Spread",
            min_value=int(math.floor(df['t1_book_spread'].min())),
            max_value=int(math.ceil(df['t1_book_spread'].max())),
            value=(int(math.floor(df['t1_book_spread'].min())), int(math.ceil(df['t1_book_spread'].max())))
        ) 
        # book moneyline
        book_home_ml_prob_lower, book_home_ml_prob_upper = st.slider(
            "Book Home ML Prob",
            min_value=float(int(math.floor(df['t1_ml_prob'].min()))),
            max_value=float(int(math.ceil(df['t1_ml_prob'].max()))),
            value=(float(int(math.floor(df['t1_ml_prob'].min()))), float(int(math.ceil(df['t1_ml_prob'].max())))),
            step = .01
        )
        # book over under line
        book_tp_lower, book_tp_upper = st.slider(
            "Book Over/Under",
            min_value=int(math.floor(df['book_over_under'].min())),
            max_value=int(math.ceil(df['book_over_under'].max())),
            value=(int(math.floor(df['book_over_under'].min())), int(math.ceil(df['book_over_under'].max())))
        )        
        # exclude ML Nas?
        exclude_na_ml = st.checkbox("Exclude NAs in the ml columns?", value=False)
        
        
        
     
    
    # column for space
    with col2:
        st.write("")
        


    # Filtered Df Show
    with col4:

        # apply filters based on selections
        filtered_hist_df = df.copy()

        # team
        if team_options != "All":
            filtered_hist_df = filtered_hist_df[filtered_hist_df['t1_team'].eq(team_options) | filtered_hist_df['t2_team'].eq(team_options)]
        else:
            filtered_hist_df = df.copy()
        # week
        if "All" not in week_options:
            filtered_hist_df = filtered_hist_df[filtered_hist_df['week'].isin(week_options)]
        # season
        if "All" not in season_options:
            filtered_hist_df = filtered_hist_df[filtered_hist_df['season'].isin(season_options)]
        # conference
        if conf_options != "All":
            filtered_hist_df = filtered_hist_df[filtered_hist_df['t1_conference'].eq(conf_options) | filtered_hist_df['t2_conference'].eq(conf_options)]             
        # confidences/values
        filtered_hist_df = filtered_hist_df[
            (filtered_hist_df['cover_confidence'] >= cover_conf_lower) &
            (filtered_hist_df['cover_confidence'] <= cover_conf_upper)|
            (filtered_hist_df['cover_confidence'].isna())
            ]
        filtered_hist_df = filtered_hist_df[
            (filtered_hist_df['ml_value'] >= ml_value_lower) & 
            (filtered_hist_df['ml_value'] <= ml_value_upper) |
            (filtered_hist_df['ml_value'].isna())
        ]        
        filtered_hist_df = filtered_hist_df[
            (filtered_hist_df['ou_confidence'] >= ou_conf_lower) &
            (filtered_hist_df['ou_confidence'] <= ou_conf_upper)|
            (filtered_hist_df['ou_confidence'].isna())
            ]        
        # Betting Lines
        filtered_hist_df = filtered_hist_df[
            (filtered_hist_df['t1_book_spread'] >= book_sp_lower) & 
            (filtered_hist_df['t1_book_spread'] <= book_sp_upper) |
            (filtered_hist_df['t1_book_spread'].isna())
        ]
        filtered_hist_df = filtered_hist_df[
            (filtered_hist_df['t1_ml_prob'] >= book_home_ml_prob_lower) & 
            (filtered_hist_df['t1_ml_prob'] <= book_home_ml_prob_upper) |
            (filtered_hist_df['t1_ml_prob'].isna())
        ]    
        filtered_hist_df = filtered_hist_df[
            (filtered_hist_df['book_over_under'] >= book_tp_lower) & 
            (filtered_hist_df['book_over_under'] <= book_tp_upper) |
            (filtered_hist_df['book_over_under'].isna())
        ]
        # exclude ML NAs
        if exclude_na_ml:
            filtered_hist_df = filtered_hist_df.dropna(subset=['ml_winnings'])
            

        # show the filtered dataframe
        st.write("### Test Data")
        st.dataframe(filtered_hist_df)
        
    
    
    ## Betting Results
    with col3:
            
            st.write("### Betting Results")

            
            ### SPREAD
            
            # Calculate our return on investment
            our_return_spread = filtered_hist_df['spread_winnings'].sum(skipna=True) / filtered_hist_df['spread_winnings'].count()

            # format it as a percentage and dollars ($100 bet)
            our_return_percentage_spread = f"{(our_return_spread * 100) - 100:.2f}%"  # If you want to show it as a percentage
            our_return_dollars_spread = f"{(100 * our_return_spread):.2f}"
            
            # Calculate bettor average return on investment
            naive_return_spread = filtered_hist_df['naive_spread_winnings'].sum(skipna=True) / filtered_hist_df['naive_spread_winnings'].count()

            # format it as a percentage and dollars ($100 bet)
            naive_return_percentage_spread = f"{(naive_return_spread * 100) - 100:.2f}%"  # If you want to show it as a percentage
            naive_return_dollars_spread = f"{(100 * naive_return_spread):.2f}"
            
            # diff between ours and avg return
            ours_over_naive_spread = f"{100 * our_return_spread - 100 * naive_return_spread:.2f}"
            naive_over_ours_spread = f"{100 * naive_return_spread - 100 * our_return_spread:.2f}"

            
            # format return print
            if our_return_spread >= 1:
                st.markdown(f"""
                        <div style="font-size:30px; font-weight:bold; color:green; text-align:center;">
                            Spread: {our_return_percentage_spread}
                        </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                        <div style="font-size:30px; font-weight:bold; color:red; text-align:center;">
                            Spread: {our_return_percentage_spread}
                        </div>
                    """, unsafe_allow_html=True)
                
            st.write("")


            # format betting strategy
            st.markdown(f"""
                    For the spread, we are betting if we predict one team to cover the spread instead of the other. So, if Team 1 is 
                        predicted to win by 4 when their spread is at -1, we bet on their side.
                """, unsafe_allow_html=True)



            # format return explanations
            if our_return_spread >= 1:
                st.markdown(f"""
                    Using our model to bet the spread in these games would give us a <span style="color:green"><b>{our_return_percentage_spread}</b></span> return on our investment.
                    In other words, if we evenly split \$100 between the games, we would finish with <span style="color:green"><b>\${our_return_dollars_spread}</b></span>.
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    Using our model to bet the spread in these games would give us a <span style="color:red"><b>{our_return_percentage_spread}</b></span> return on our investment.
                    In other words, if we evenly split \$100 between the games, we would finish with <span style="color:red"><b>\${our_return_dollars_spread}</b></span>.
                """, unsafe_allow_html=True)
            
            
            # format naive comparison
            if our_return_spread > naive_return_spread:
                st.markdown(f"""
                    The average bettor would earn a **{naive_return_percentage_spread}** return on our investment, both sides being bet evenly.
                    If they evenly split \$100 between the games, we would finish with <span><b>\${naive_return_dollars_spread}</b></span>, which is <span style="color:green"><b>${ours_over_naive_spread}</b></span> less than we made.
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    The average bettor would earn a **{naive_return_percentage_spread}** return on their investment, both sides being bet evenly.
                    If they evenly split \$100 between the games, they would finish with <span><b>\${naive_return_dollars_spread}</b></span>, which is <span style="color:red"><b>${naive_over_ours_spread}</b></span> more than we made.
                """, unsafe_allow_html=True)
            
            # list sample size
            filtered_row_total_spread = filtered_hist_df['spread_winnings'].count()
            st.markdown(f"""
                <span><i>using a sample of {filtered_row_total_spread} games for spread calculations</span>
                """, unsafe_allow_html=True)
                
                
            st.write("")
            st.write("")
                
                
                


           ### MONEYLINE
            
            # our return on investment
            our_return_moneyline = filtered_hist_df['ml_winnings'].sum(skipna=True) / filtered_hist_df['ml_winnings'].count()

            # format it as a percentage and dollars ($100 bet)
            our_return_percentage_moneyline = f"{(our_return_moneyline * 100) - 100:.2f}%"  # If you want to show it as a percentage
            our_return_dollars_moneyline = f"{(100 * our_return_moneyline):.2f}"
            
            # average return on investment
            naive_return_moneyline = filtered_hist_df['naive_ml_winnings'].sum(skipna=True) / filtered_hist_df['naive_ml_winnings'].count()

            # format it as a percentage and dollars ($100 bet)
            naive_return_percentage_moneyline = f"{(naive_return_moneyline * 100) - 100:.2f}%"  # If you want to show it as a percentage
            naive_return_dollars_moneyline = f"{(100 * naive_return_moneyline):.2f}"
            
            # diff between ours and average
            ours_over_naive_moneyline = f"{100 * our_return_moneyline - 100 * naive_return_moneyline:.2f}"
            naive_over_ours_moneyline = f"{100 * naive_return_moneyline - 100 * our_return_moneyline:.2f}"

            
            # format overall print          
            if our_return_moneyline >= 1:
                st.markdown(f"""
                        <div style="font-size:30px; font-weight:bold; color:green; text-align:center;">
                            Moneyline: {our_return_percentage_moneyline}
                        </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                        <div style="font-size:30px; font-weight:bold; color:red; text-align:center;">
                            Moneyline: {our_return_percentage_moneyline}
                        </div>
                    """, unsafe_allow_html=True)
                
            st.write("")



            # format betting strategy
            st.markdown(f"""
                    For moneyline, we are betting only if we detect value on a certain side. So, if Team 1 has a moneyline of +240
                        but our win probability model says it should really be +160, we bet on their side. 
                """, unsafe_allow_html=True)

            # format return description
            if our_return_moneyline >= 1:
                st.markdown(f"""
                    Using our model to bet the moneyline in these games would give us a <span style="color:green"><b>{our_return_percentage_moneyline}</b></span> return on our investment.
                    In other words, if we evenly split \$100 between the games, we would finish with <span style="color:green"><b>\${our_return_dollars_moneyline}</b></span>.
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    Using our model to bet the moneyline in these games would give us a <span style="color:red"><b>{our_return_percentage_moneyline}</b></span> return on our investment.
                    In other words, if we evenly split \$100 between the games, we would finish with <span style="color:red"><b>\${our_return_dollars_moneyline}</b></span>.
                """, unsafe_allow_html=True)
            
            
            # format naive comparison
            if our_return_moneyline > naive_return_moneyline:
                st.markdown(f"""
                    A conservative bettor would earn a **{naive_return_percentage_moneyline}** return on their investment if they bet both sides.
                    If they evenly split \$100 between the games, we would finish with <span><b>\${naive_return_dollars_moneyline}</b></span>, which is <span style="color:green"><b>${ours_over_naive_moneyline}</b></span> less than we made.
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    A conservative bettor would earn a **{naive_return_percentage_moneyline}** return on their investment if they bet both sides.
                    If they evenly split \$100 between the games, they would finish with <span><b>\${naive_return_dollars_moneyline}</b></span>, which is <span style="color:red"><b>${naive_over_ours_moneyline}</b></span> more than we made.
                """, unsafe_allow_html=True)
                
            
            # list sample size
            filtered_row_total_moneyline = filtered_hist_df['ml_winnings'].count()
            st.markdown(f"""
                <span><i>using a sample of {filtered_row_total_moneyline} games for moneyline calculations</span>
                """, unsafe_allow_html=True)

            st.write("")
            st.write("")                





           ### OVER/UNDER
            
            # our return on investment
            our_return_ou = filtered_hist_df['ou_winnings'].sum(skipna=True) / filtered_hist_df['ou_winnings'].count()

            # format it as a percentage and dollars ($100 bet)
            our_return_percentage_ou = f"{(our_return_ou * 100) - 100:.2f}%"  # If you want to show it as a percentage
            our_return_dollars_ou = f"{(100 * our_return_ou):.2f}"
            
            # average return on investment
            naive_return_ou = filtered_hist_df['naive_ou_winnings'].sum(skipna=True) / filtered_hist_df['naive_ou_winnings'].count()

            # format it as a percentage and dollars ($100 bet)
            naive_return_percentage_ou = f"{(naive_return_ou * 100) - 100:.2f}%"  # If you want to show it as a percentage
            naive_return_dollars_ou = f"{(100 * naive_return_ou):.2f}"
            
            # diff between ours and average
            ours_over_naive_ou = f"{100 * our_return_ou - 100 * naive_return_ou:.2f}"
            naive_over_ours_ou = f"{100 * naive_return_ou - 100 * our_return_ou:.2f}"

            
            # format overall print          
            if our_return_ou >= 1:
                st.markdown(f"""
                        <div style="font-size:30px; font-weight:bold; color:green; text-align:center;">
                            Over/Under: {our_return_percentage_ou}
                        </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                        <div style="font-size:30px; font-weight:bold; color:red; text-align:center;">
                            Over/Under: {our_return_percentage_ou}
                        </div>
                    """, unsafe_allow_html=True)
                
            st.write("")



            # format betting strategy
            st.markdown(f"""
                    For over/under, we are betting the side that our model predicts to be correct. So, if the line is 37.5 and our model
                    predicts 40 total points, we bet the over. 
                """, unsafe_allow_html=True)

            # format return description
            if our_return_ou >= 1:
                st.markdown(f"""
                    Using our model to bet over/under in these games would give us a <span style="color:green"><b>{our_return_percentage_ou}</b></span> return on our investment.
                    In other words, if we evenly split \$100 between the games, we would finish with <span style="color:green"><b>\${our_return_dollars_ou}</b></span>.
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    Using our model to bet over/under in these games would give us a <span style="color:red"><b>{our_return_percentage_ou}</b></span> return on our investment.
                    In other words, if we evenly split \$100 between the games, we would finish with <span style="color:red"><b>\${our_return_dollars_ou}</b></span>.
                """, unsafe_allow_html=True)
            
            
            # format naive comparison
            if our_return_ou > naive_return_ou:
                st.markdown(f"""
                    A conservative bettor would earn a **{naive_return_percentage_ou}** return on their investment if they bet both sides.
                    If they evenly split \$100 between the games, we would finish with <span><b>\${naive_return_dollars_ou}</b></span>, which is <span style="color:green"><b>${ours_over_naive_ou}</b></span> less than we made.
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    A conservative bettor would earn a **{naive_return_percentage_ou}** return on their investment if they bet both sides.
                    If they evenly split \$100 between the games, they would finish with <span><b>\${naive_return_dollars_ou}</b></span>, which is <span style="color:red"><b>${naive_over_ours_ou}</b></span> more than we made.
                """, unsafe_allow_html=True)
                

            # list sample size
            filtered_row_total_ou = filtered_hist_df['ou_winnings'].count()
            st.markdown(f"""
                <span><i>using a sample of {filtered_row_total_ou} games for over/under calculations</span>
                """, unsafe_allow_html=True)


    st.write(" ")
    st.write(" ")

    st.divider()

    # further notes
    st.write("These returns are assuming you bet the same units for hundreds of games. If you use the model on one game, the return is either gonna be 0 or ~95%. If you bet on 100, it will look more like this projected reutrn. It's a sample size game: it won't get every game right but will be profitable over time.")
    st.write("Based on 2021-24 backtesting: the spread model is significantly profitable and increases profitability as confidence increases. The moneyline model is significantly profitable and increases profitability as value increases. The O/U model is not profitable.")
    st.write("'Confidence' represents (our predicted probability - 50%). For example, if a team has a 55% chance to cover, their cover confidence is 5%. Similarly, ML Value is a direct comparison of our predicted win " \
    "probability vs the book's implicit win probability from ML. For example, if the book's win probability is 55% and ours is 60%, that's a 5% value.")
    st.write("As a note, some moneyline values for games are incomplete in the data source. These appear as NAs. Others may be NA due to a lack of prior data in the season for one of the teams.")
    





                
                
#### PLAYOFF PREDICTION PAGE 
#               
def playoff_page():
  
  # Streamlit App
  st.title("The Playoff")
  
  st.write("Customize your playoff! Select what teams you want in each seed. The model will predict each game, but you can override with the checkbox.")

  st.write("  ")


  
  # create columns for layout
  col1, col2, col3, col4, col5, col6, col7, col8, col9 = st.columns([4,2,4,1,4,1,4,1,4])
  
  


  ## Filters

  with col1:
    

      # put default playoff teams from rankings into list
      playoff_teams = []
      for i in poll_rankings["school"]:
            playoff_teams.append(i)
      # find indices of teams
      seeds = []
      for value in playoff_teams:
        try:
            unique_teams = sorted(theor_prepped["t1_team"].dropna().unique())
            index = unique_teams.index(value)
            # in case team isn't there
        except ValueError:
            index = 30
        seeds.append(index)
    
      
      # select boxes with team defaults
      seed1 = st.selectbox("1 Seed", sorted(theor_prepped["t1_team"].dropna().unique()), index = seeds[0])
      seed2 = st.selectbox("2 Seed", sorted(theor_prepped["t1_team"].dropna().unique()), index = seeds[1])
      seed3 = st.selectbox("3 Seed", sorted(theor_prepped["t1_team"].dropna().unique()), index = seeds[2])
      seed4 = st.selectbox("4 Seed", sorted(theor_prepped["t1_team"].dropna().unique()), index = seeds[3])
      seed5 = st.selectbox("5 Seed", sorted(theor_prepped["t1_team"].dropna().unique()), index = seeds[4])
      seed6 = st.selectbox("6 Seed", sorted(theor_prepped["t1_team"].dropna().unique()), index = seeds[5])
      seed7 = st.selectbox("7 Seed", sorted(theor_prepped["t1_team"].dropna().unique()), index = seeds[6])
      seed8 = st.selectbox("8 Seed", sorted(theor_prepped["t1_team"].dropna().unique()), index = seeds[7])
      seed9 = st.selectbox("9 Seed", sorted(theor_prepped["t1_team"].dropna().unique()), index = seeds[8])
      seed10 = st.selectbox("10 Seed", sorted(theor_prepped["t1_team"].dropna().unique()), index = seeds[9])
      seed11 = st.selectbox("11 Seed", sorted(theor_prepped["t1_team"].dropna().unique()), index = seeds[10])
      seed12 = st.selectbox("12 Seed", sorted(theor_prepped["t1_team"].dropna().unique()), index = seeds[11])
      
      
      

  ## First Round


  with col3:
    
        
        st.markdown(f"""
            <div style="font-size:30px; font-weight:bold; text-align:center; line-height:1.75; color:orange;">
                1st Round <br>
            </div>
        """, unsafe_allow_html=True)
        
        
        ## FR1

        # check for override
        if "override_fr1" not in st.session_state:
            st.session_state.override_fr1 = False
        # teams
        results_fr1 = theor_prepped[
            (theor_prepped["t1_team"] == seed8) &
            (theor_prepped["t2_team"] == seed9) &
            (theor_prepped["t1_home"] == 1)]
    
        # header
        st.markdown(f"""
            <div style="font-size:20px; font-weight:bold; text-align:center; line-height:2; color:#87CEEB;">
                Game 1 <br>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
                <div style="font-size:25px; font-weight:bold; text-align:center; line-height:1;">
                    {seed8} <br> vs <br> {seed9}
                </div>
            """, unsafe_allow_html=True)
        
        # go with predicted winner or override?
        if st.session_state.override_fr1 == True:
            
            st.markdown("<div style='display: flex; justify-content: center;'><img src='" + results_fr1.iloc[0]["losing_logo"] + "' width='60'></div>", unsafe_allow_html=True)
            
            st.markdown(f"""
                <div style="font-size:15px; font-weight:bold; color:{results_fr1.iloc[0]["losing_color"]}; text-align:center;">
                      
                </div>
            """, unsafe_allow_html=True)
            
            fr1_winner = results_fr1.iloc[0]["losing_team"]
            
        else:
            st.markdown("<div style='display: flex; justify-content: center;'><img src='" + results_fr1.iloc[0]["winning_logo"] + "' width='60'></div>", unsafe_allow_html=True)
            
            st.markdown(f"""
                <div style="font-size:15px; font-weight:bold; color:{results_fr1.iloc[0]["winning_color"]}; text-align:center;">
                    {results_fr1.iloc[0]["pred_winner_wp"]}% wp
                </div>
            """, unsafe_allow_html=True)
            
            fr1_winner = results_fr1.iloc[0]["winning_team"]
        
            
        # override checkbox
        st.checkbox("Override Win?", value=st.session_state.override_fr1, key = "override_fr1")
        
        
        st.write(" ")
        
       
       
        
        ## FR2

        # check for override
        if "override_fr2" not in st.session_state:
            st.session_state.override_fr2 = False
        # teams
        results_fr2 = theor_prepped[
            (theor_prepped["t1_team"] == seed7) &
            (theor_prepped["t2_team"] == seed10) &
            (theor_prepped["t1_home"] == 1)]    
    
        # header
        st.markdown(f"""
            <div style="font-size:20px; font-weight:bold; text-align:center; line-height:2; color:#87CEEB;">
                Game 2 <br>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
                <div style="font-size:25px; font-weight:bold; text-align:center; line-height:1;">
                    {seed7} <br> vs <br> {seed10}
                </div>
            """, unsafe_allow_html=True)
        
        # go with predicted winner or override?
        if st.session_state.override_fr2 == True:
            
            st.markdown("<div style='display: flex; justify-content: center;'><img src='" + results_fr2.iloc[0]["losing_logo"] + "' width='60'></div>", unsafe_allow_html=True)
            
            st.markdown(f"""
                <div style="font-size:15px; font-weight:bold; color:{results_fr2.iloc[0]["winning_color"]}; text-align:center;">
                      
                </div>
            """, unsafe_allow_html=True)
            
            fr2_winner = results_fr2.iloc[0]["losing_team"]
            
        else:
            st.markdown("<div style='display: flex; justify-content: center;'><img src='" + results_fr2.iloc[0]["winning_logo"] + "' width='60'></div>", unsafe_allow_html=True)

            st.markdown(f"""
                <div style="font-size:15px; font-weight:bold; color:{results_fr2.iloc[0]["winning_color"]}; text-align:center;">
                    {results_fr2.iloc[0]["pred_winner_wp"]}% wp
                </div>
            """, unsafe_allow_html=True)
            
            fr2_winner = results_fr2.iloc[0]["winning_team"]
        
            
        # override checkbox
        st.checkbox("Override Win?", value=st.session_state.override_fr2, key = "override_fr2")
        
        
        st.write(" ")
        
        
        


        ## FR3

        # check for override
        if "override_fr3" not in st.session_state:
            st.session_state.override_fr3 = False
        # teams
        results_fr3 = theor_prepped[
            (theor_prepped["t1_team"] == seed6) &
            (theor_prepped["t2_team"] == seed11) &
            (theor_prepped["t1_home"] == 1)]   
    
        # header
        st.markdown(f"""
            <div style="font-size:20px; font-weight:bold; text-align:center; line-height:2; color:#87CEEB;">
                Game 3 <br>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
                <div style="font-size:25px; font-weight:bold; text-align:center; line-height:1;">
                    {seed6} <br> vs <br> {seed11}
                </div>
            """, unsafe_allow_html=True)
        
        # go with predicted winner or override?
        if st.session_state.override_fr3 == True:
            
            st.markdown("<div style='display: flex; justify-content: center;'><img src='" + results_fr3.iloc[0]["losing_logo"] + "' width='60'></div>", unsafe_allow_html=True)
            
            st.markdown(f"""
                <div style="font-size:15px; font-weight:bold; color:{results_fr3.iloc[0]["winning_color"]}; text-align:center;">
                      
                </div>
            """, unsafe_allow_html=True)
            
            fr3_winner = results_fr3.iloc[0]["losing_team"]
            
        else:
            st.markdown("<div style='display: flex; justify-content: center;'><img src='" + results_fr3.iloc[0]["winning_logo"] + "' width='60'></div>", unsafe_allow_html=True)
            
            st.markdown(f"""
                <div style="font-size:15px; font-weight:bold; color:{results_fr3.iloc[0]["winning_color"]}; text-align:center;">
                    {results_fr3.iloc[0]["pred_winner_wp"]}% wp
                </div>
            """, unsafe_allow_html=True)
            
            fr3_winner = results_fr3.iloc[0]["winning_team"]
        
            
        # override checkbox
        st.checkbox("Override Win?", value=st.session_state.override_fr3, key = "override_fr3")
        
        
        st.write(" ")
        
        
        
        
        
        ## FR4

        # check for override
        if "override_fr4" not in st.session_state:
            st.session_state.override_fr4 = False
        # teams
        results_fr4 = theor_prepped[
            (theor_prepped["t1_team"] == seed5) &
            (theor_prepped["t2_team"] == seed12) &
            (theor_prepped["t1_home"] == 1)]    
    
        # header
        st.markdown(f"""
            <div style="font-size:20px; font-weight:bold; text-align:center; line-height:2; color:#87CEEB;">
                Game 4 <br>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
                <div style="font-size:25px; font-weight:bold; text-align:center; line-height:1;">
                    {seed5} <br> vs <br> {seed12}
                </div>
            """, unsafe_allow_html=True)
        
        # go with predicted winner or override?
        if st.session_state.override_fr4 == True:
            
            st.markdown("<div style='display: flex; justify-content: center;'><img src='" + results_fr4.iloc[0]["losing_logo"] + "' width='60'></div>", unsafe_allow_html=True)
            
            st.markdown(f"""
                <div style="font-size:15px; font-weight:bold; color:{results_fr4.iloc[0]["winning_color"]}; text-align:center;">
                      
                </div>
            """, unsafe_allow_html=True)
            
            fr4_winner = results_fr4.iloc[0]["losing_team"]
            
        else:
            st.markdown("<div style='display: flex; justify-content: center;'><img src='" + results_fr4.iloc[0]["winning_logo"] + "' width='60'></div>", unsafe_allow_html=True)
            
            st.markdown(f"""
                <div style="font-size:15px; font-weight:bold; color:{results_fr4.iloc[0]["winning_color"]}; text-align:center;">
                    {results_fr4.iloc[0]["pred_winner_wp"]}% wp
                </div>
            """, unsafe_allow_html=True)
            
            fr4_winner = results_fr4.iloc[0]["winning_team"]
        
            
        # override checkbox
        st.checkbox("Override Win?", value=st.session_state.override_fr4, key = "override_fr4")
        
        

        
        
  ## Second Round
        
  with col5:
    
    
        st.markdown(f"""
            <div style="font-size:30px; font-weight:bold; text-align:center; line-height:1.75; color:orange;">
                Quarters <br>
            </div>
        """, unsafe_allow_html=True)



        ## Rose Bowl

        # check for override
        if "override_rb" not in st.session_state:
            st.session_state.override_rb = False
        # teams
        results_rb = theor_prepped[
            (theor_prepped["t1_team"] == seed1) &
            (theor_prepped["t2_team"] == fr1_winner) &
            (theor_prepped["neutral_site"] == 1)]            
    
        # header
        st.markdown(f"""
            <div style="font-size:20px; font-weight:bold; text-align:center; line-height:2; color:#87CEEB;">
                Rose Bowl <br>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
                <div style="font-size:25px; font-weight:bold; text-align:center; line-height:1;">
                    {seed1} <br> vs <br> {fr1_winner}
                </div>
            """, unsafe_allow_html=True)
        
        # go with predicted winner or override?
        if st.session_state.override_rb == True:
            
            st.markdown("<div style='display: flex; justify-content: center;'><img src='" + results_rb.iloc[0]["losing_logo"] + "' width='60'></div>", unsafe_allow_html=True)
            
            st.markdown(f"""
                <div style="font-size:15px; font-weight:bold; color:{results_rb.iloc[0]["winning_color"]}; text-align:center;">
                      
                </div>
            """, unsafe_allow_html=True)
            
            rb_winner = results_rb.iloc[0]["losing_team"]
            
        else:
            st.markdown("<div style='display: flex; justify-content: center;'><img src='" + results_rb.iloc[0]["winning_logo"] + "' width='60'></div>", unsafe_allow_html=True)
            
            st.markdown(f"""
                <div style="font-size:15px; font-weight:bold; color:{results_rb.iloc[0]["winning_color"]}; text-align:center;">
                    {results_rb.iloc[0]["pred_winner_wp"]}% wp
                </div>
            """, unsafe_allow_html=True)
            
            rb_winner = results_rb.iloc[0]["winning_team"]
        
            
        # override checkbox
        st.checkbox("Override Win?", value=st.session_state.override_rb, key = "override_rb")
        
        
        st.write(" ")
        
       
       
        
        ## Sugar Bowl

        # check for override
        if "override_sb" not in st.session_state:
            st.session_state.override_sb = False
        # teams
        results_sb = theor_prepped[
            (theor_prepped["t1_team"] == seed2) &
            (theor_prepped["t2_team"] == fr2_winner) &
            (theor_prepped["neutral_site"] == 1)]      
    
        # header
        st.markdown(f"""
            <div style="font-size:20px; font-weight:bold; text-align:center; line-height:2; color:#87CEEB;">
                Sugar Bowl <br>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
                <div style="font-size:25px; font-weight:bold; text-align:center; line-height:1;">
                    {seed2} <br> vs <br> {fr2_winner}
                </div>
            """, unsafe_allow_html=True)
        
        # go with predicted winner or override?
        if st.session_state.override_sb == True:
            
            st.markdown("<div style='display: flex; justify-content: center;'><img src='" + results_sb.iloc[0]["losing_logo"] + "' width='60'></div>", unsafe_allow_html=True)
            
            st.markdown(f"""
                <div style="font-size:15px; font-weight:bold; color:{results_sb.iloc[0]["winning_color"]}; text-align:center;">
                      
                </div>
            """, unsafe_allow_html=True)
            
            sb_winner = results_sb.iloc[0]["losing_team"]
            
        else:
            st.markdown("<div style='display: flex; justify-content: center;'><img src='" + results_sb.iloc[0]["winning_logo"] + "' width='60'></div>", unsafe_allow_html=True)
            
            st.markdown(f"""
                <div style="font-size:15px; font-weight:bold; color:{results_sb.iloc[0]["winning_color"]}; text-align:center;">
                    {results_sb.iloc[0]["pred_winner_wp"]}% wp
                </div>
            """, unsafe_allow_html=True)
            
            sb_winner = results_sb.iloc[0]["winning_team"]
        
            
        # override checkbox
        st.checkbox("Override Win?", value=st.session_state.override_sb, key = "override_sb")
        
        
        st.write(" ")
        
        
        


        ## Fiesta Bowl

        # check for override
        if "override_fb" not in st.session_state:
            st.session_state.override_fb = False
        # teams
        results_fb = theor_prepped[
            (theor_prepped["t1_team"] == seed3) &
            (theor_prepped["t2_team"] == fr3_winner) &
            (theor_prepped["neutral_site"] == 1)]     
    
        # header
        st.markdown(f"""
            <div style="font-size:20px; font-weight:bold; text-align:center; line-height:2; color:#87CEEB;">
                Fiesta Bowl <br>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
                <div style="font-size:25px; font-weight:bold; text-align:center; line-height:1;">
                    {seed3} <br> vs <br> {fr3_winner}
                </div>
            """, unsafe_allow_html=True)
        
        # go with predicted winner or override?
        if st.session_state.override_fb == True:
            
            st.markdown("<div style='display: flex; justify-content: center;'><img src='" + results_fb.iloc[0]["losing_logo"] + "' width='60'></div>", unsafe_allow_html=True)
            
            st.markdown(f"""
                <div style="font-size:15px; font-weight:bold; color:{results_fb.iloc[0]["winning_color"]}; text-align:center;">
                      
                </div>
            """, unsafe_allow_html=True)
            
            fb_winner = results_fb.iloc[0]["losing_team"]
            
        else:
            st.markdown("<div style='display: flex; justify-content: center;'><img src='" + results_fb.iloc[0]["winning_logo"] + "' width='60'></div>", unsafe_allow_html=True)
            
            st.markdown(f"""
                <div style="font-size:15px; font-weight:bold; color:{results_fb.iloc[0]["winning_color"]}; text-align:center;">
                    {results_fb.iloc[0]["pred_winner_wp"]}% wp
                </div>
            """, unsafe_allow_html=True)
            
            fb_winner = results_fb.iloc[0]["winning_team"]
        
            
        # override checkbox
        st.checkbox("Override Win?", value=st.session_state.override_fb, key = "override_fb")
        
        
        st.write(" ")
        
        
        
        
        
        ## Peach Bowl

        # check for override
        if "override_pb" not in st.session_state:
            st.session_state.override_pb = False
        # teams
        results_pb = theor_prepped[
            (theor_prepped["t1_team"] == seed4) &
            (theor_prepped["t2_team"] == fr4_winner) &
            (theor_prepped["neutral_site"] == 1)]     
    
        # header
        st.markdown(f"""
            <div style="font-size:20px; font-weight:bold; text-align:center; line-height:2; color:#87CEEB;">
                Peach Bowl <br>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
                <div style="font-size:25px; font-weight:bold; text-align:center; line-height:1;">
                    {seed4} <br> vs <br> {fr4_winner}
                </div>
            """, unsafe_allow_html=True)
        
        # go with predicted winner or override?
        if st.session_state.override_pb == True:
            
            st.markdown("<div style='display: flex; justify-content: center;'><img src='" + results_pb.iloc[0]["losing_logo"] + "' width='60'></div>", unsafe_allow_html=True)
            
            st.markdown(f"""
                <div style="font-size:15px; font-weight:bold; color:{results_pb.iloc[0]["winning_color"]}; text-align:center;">
                      
                </div>
            """, unsafe_allow_html=True)
            
            pb_winner = results_pb.iloc[0]["losing_team"]
            
        else:
            st.markdown("<div style='display: flex; justify-content: center;'><img src='" + results_pb.iloc[0]["winning_logo"] + "' width='60'></div>", unsafe_allow_html=True)
            
            st.markdown(f"""
                <div style="font-size:15px; font-weight:bold; color:{results_pb.iloc[0]["winning_color"]}; text-align:center;">
                    {results_pb.iloc[0]["pred_winner_wp"]}% wp
                </div>
            """, unsafe_allow_html=True)
            
            pb_winner = results_pb.iloc[0]["winning_team"]
        
            
        # override checkbox
        st.checkbox("Override Win?", value=st.session_state.override_pb, key = "override_pb")
        
        
        
        
  
  ## Third Round

  with col7:
    
    
        st.markdown(f"""
            <div style="font-size:30px; font-weight:bold; text-align:center; line-height:1.75; color:orange;">
                Semis <br>
            </div>
        """, unsafe_allow_html=True)



        ## Cotton Bowl

        # check for override
        if "override_cb" not in st.session_state:
            st.session_state.override_cb = False
        # teams
        results_cb = theor_prepped[
            (theor_prepped["t1_team"] == rb_winner) &
            (theor_prepped["t2_team"] == pb_winner) &
            (theor_prepped["neutral_site"] == 1)] 
    
        # header
        st.markdown(f"""
            <div style="font-size:20px; font-weight:bold; text-align:center; line-height:2; color:#87CEEB;">
                Cotton Bowl <br>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
                <div style="font-size:25px; font-weight:bold; text-align:center; line-height:1;">
                    {rb_winner} <br> vs <br> {pb_winner}
                </div>
            """, unsafe_allow_html=True)
        
        # go with predicted winner or override?
        if st.session_state.override_cb == True:
            
            st.markdown("<div style='display: flex; justify-content: center;'><img src='" + results_cb.iloc[0]["losing_logo"] + "' width='60'></div>", unsafe_allow_html=True)
            
            st.markdown(f"""
                <div style="font-size:15px; font-weight:bold; color:{results_cb.iloc[0]["winning_color"]}; text-align:center;">
                      
                </div>
            """, unsafe_allow_html=True)
            
            cb_winner = results_cb.iloc[0]["losing_team"]
            
        else:
            st.markdown("<div style='display: flex; justify-content: center;'><img src='" + results_cb.iloc[0]["winning_logo"] + "' width='60'></div>", unsafe_allow_html=True)
            
            st.markdown(f"""
                <div style="font-size:15px; font-weight:bold; color:{results_cb.iloc[0]["winning_color"]}; text-align:center;">
                    {results_cb.iloc[0]["pred_winner_wp"]}% wp
                </div>
            """, unsafe_allow_html=True)
            
            cb_winner = results_cb.iloc[0]["winning_team"]
        
            
        # override checkbox
        st.checkbox("Override Win?", value=st.session_state.override_cb, key = "override_cb")
        
        
        st.write(" ")
        
       
       
        
        ## Orange Bowl

        # check for override
        if "override_ob" not in st.session_state:
            st.session_state.override_ob = False
        # teams
        results_ob = theor_prepped[
            (theor_prepped["t1_team"] == sb_winner) &
            (theor_prepped["t2_team"] == fb_winner) &
            (theor_prepped["neutral_site"] == 1)]     
    
        # header
        st.markdown(f"""
            <div style="font-size:20px; font-weight:bold; text-align:center; line-height:2; color:#87CEEB;">
                Orange Bowl <br>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
                <div style="font-size:25px; font-weight:bold; text-align:center; line-height:1;">
                    {sb_winner} <br> vs <br> {fb_winner}
                </div>
            """, unsafe_allow_html=True)
        
        # go with predicted winner or override?
        if st.session_state.override_ob == True:
            
            st.markdown("<div style='display: flex; justify-content: center;'><img src='" + results_ob.iloc[0]["losing_logo"] + "' width='60'></div>", unsafe_allow_html=True)
            
            st.markdown(f"""
                <div style="font-size:15px; font-weight:bold; color:{results_ob.iloc[0]["winning_color"]}; text-align:center;">
                      
                </div>
            """, unsafe_allow_html=True)
            
            ob_winner = results_ob.iloc[0]["losing_team"]
            
        else:
            st.markdown("<div style='display: flex; justify-content: center;'><img src='" + results_ob.iloc[0]["winning_logo"] + "' width='60'></div>", unsafe_allow_html=True)
            
            st.markdown(f"""
                <div style="font-size:15px; font-weight:bold; color:{results_ob.iloc[0]["winning_color"]}; text-align:center;">
                    {results_ob.iloc[0]["pred_winner_wp"]}% wp
                </div>
            """, unsafe_allow_html=True)
            
            ob_winner = results_ob.iloc[0]["winning_team"]
        
            
        # override checkbox
        st.checkbox("Override Win?", value=st.session_state.override_ob, key = "override_ob")
        
        
        
        

  # Natty

  with col9:
    
        st.markdown(f"""
            <div style="font-size:30px; font-weight:bold; text-align:center; line-height:1.75; color:orange;">
                Natty <br>
            </div>
        """, unsafe_allow_html=True)


        ## National Championship

        # check for override
        if "override_nc" not in st.session_state:
            st.session_state.override_nc = False
        # teams
        results_nc = theor_prepped[
            (theor_prepped["t1_team"] == cb_winner) &
            (theor_prepped["t2_team"] == ob_winner) &
            (theor_prepped["neutral_site"] == 1)]     
    
        # header
        st.markdown(f"""
            <div style="font-size:20px; font-weight:bold; text-align:center; line-height:2; color:#87CEEB;">
                Championship <br>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
                <div style="font-size:25px; font-weight:bold; text-align:center; line-height:1;">
                    {cb_winner} <br> vs <br> {ob_winner}
                </div>
            """, unsafe_allow_html=True)
        
        # go with predicted winner or override?
        if st.session_state.override_nc == True:
            
            st.markdown("<div style='display: flex; justify-content: center;'><img src='" + results_nc.iloc[0]["losing_logo"] + "' width='60'></div>", unsafe_allow_html=True)
            
            st.markdown(f"""
                <div style="font-size:15px; font-weight:bold; color:{results_nc.iloc[0]["winning_color"]}; text-align:center;">
                      
                </div>
            """, unsafe_allow_html=True)
            
            nc_winner = results_nc.iloc[0]["losing_team"]
            
        else:
            st.markdown("<div style='display: flex; justify-content: center;'><img src='" + results_nc.iloc[0]["winning_logo"] + "' width='60'></div>", unsafe_allow_html=True)
            
            st.markdown(f"""
                <div style="font-size:15px; font-weight:bold; color:{results_nc.iloc[0]["winning_color"]}; text-align:center;">
                    {results_nc.iloc[0]["pred_winner_wp"]}% wp
                </div>
            """, unsafe_allow_html=True)
            
            nc_winner = results_nc.iloc[0]["winning_team"]
        
            
        # override checkbox
        st.checkbox("Override Win?", value=st.session_state.override_nc, key = "override_nc")
        
        
              

        
        

  
  
  
                
                

#### GAME PREDICTOR PAGE

def game_predictor_page():
  
  
  st.title("Game Predictor")
  
  st.write("Pick a combination of teams and see what the model predicts if they played today!")

  st.write(" ")


  # unique teams
  team_list = sorted(theor_prepped["t2_team"].dropna().unique())


  # default top 2 teams from rankings
  top2_teams = []
  for i in poll_rankings["school"][:2]:
          top2_teams.append(i)
  # find indices
  top2_indices = []
  for value in top2_teams:
        try:
            index = team_list.index(value)
        # default if nothing
        except ValueError:
            index = 30
        top2_indices.append(index)



  # create cols for layout
  col1, col2 = st.columns([1,2])
    

  
  ## Filters 
  with col1:
      
      
      st.write("### Filters")
      
      # select away team
      away_team = st.selectbox("Select Away Team", team_list, index = top2_indices[1])
      
      # select home team
      home_team = st.selectbox("Select Home Team", team_list, index = top2_indices[0])
      
      
      # neutral site or no? encode
      neutral_site_input = st.selectbox("Neutral Site?", ["No", "Yes"])
      neutral_site_ind = 1 if neutral_site_input == "Yes" else 0
      
      st.write("  ")
      st.write(" ")
      
      # most recent week
      st.markdown(f"""
          <div style="font-size:18px; font-weight:bold; text-align:center;">
              Data last updated after <br> <span style="color: orange;"> Week {this_week} </span>
          </div>
      """, unsafe_allow_html=True)

      
  
  ## Prediction Results Col 

  with col2:  
      
      # can't have a team playing itself
      if home_team == away_team:
          
        # formatting
        st.markdown(f"""
                                <div style="font-size:35px; font-weight:bold; text-align:center;">
                                    Please adjust one of the teams!
                                </div>
                            """, unsafe_allow_html=True)
        st.write("  ")
        st.write("  ")

      # if the matchup is good to go
      else:    
      
        if neutral_site_ind == 0:
            results = theor_prepped[(theor_prepped['t1_team'] == home_team) & (theor_prepped['t2_team'] == away_team) &
                                    (theor_prepped['t1_home'] == 1)]
        else:
            results = theor_prepped[(theor_prepped['t1_team'] == home_team) & (theor_prepped['t2_team'] == away_team) &
                                    (theor_prepped['neutral_site'] == 1)]

        
        
        # formatting
        st.markdown(f"""
                                <div style="font-size:35px; font-weight:bold; text-align:center;">
                                    {away_team} vs {home_team}...
                                </div>
                            """, unsafe_allow_html=True)
        st.write("  ")
        st.write("  ")
        
        
        st.markdown("<div style='display: flex; justify-content: center;'><img src='" + results.iloc[0]["winning_logo"] + "' width='200'></div>", unsafe_allow_html=True)
        
        st.write("  ")
        
        
        if results.iloc[0]["t1_win"] == 1:
                        st.markdown(f"""
                                <div style="font-size:35px; font-weight:bold; color:{results.iloc[0]["winning_color"]}; text-align:center;">
                                    {home_team} wins! </div> <br> <div style="font-size:25px; font-weight:bold; color:{results.iloc[0]["winning_color"]}; text-align:center;">
                                    {results.iloc[0]["pred_t1_wp"]}% win prob
                                </div>
                            """, unsafe_allow_html=True)
        else:
                        st.markdown(f"""
                                <div style="font-size:35px; font-weight:bold; color:{results.iloc[0]["winning_color"]}; text-align:center;">
                                    {away_team} wins! </div> <br> <div style="font-size:25px; font-weight:bold; color:{results.iloc[0]["winning_color"]}; text-align:center;">
                                    {100-results.iloc[0]["pred_t1_wp"]}% win prob
                                </div>
                            """, unsafe_allow_html=True)
                        
        st.write("")

        # mark it if it's a real game this week (prediction calculated from real book values)
        if results.iloc[0]["real_game"] == 1:
            st.markdown(f"""
                    <div style="font-size:35px; font-weight:bold; color: gold; text-align:center;">
                        Real Matchup
                    </div>
                """, unsafe_allow_html=True)            
                            




#### Power Rankings page

def power_rankings_page():

    st.title("The Czar Poll")
    
    st.write("If every team played each other home, away, and neutral, these are our predicted standings:")

    # conference filter
    unique_conf = sorted(theor_agg['Conf'].unique())

    conf_options = st.multiselect(
    "Conference", 
    options=["All"] + unique_conf,  
    default=["All"]  
    )

    # data frame filter
    theor_agg_filt = theor_agg
    if "All" not in conf_options:
        theor_agg_filt = theor_agg_filt[theor_agg_filt['Conf'].isin(conf_options)]

    # show df
    st.dataframe(theor_agg_filt)






#### This Week Picks Page

def this_week_page():

    ## Initialize Session States (currently not in the app, but incase we go back to dynamic predictions)

    # Spread
    if "show_spread_input" not in st.session_state:
        st.session_state.show_spread_input = False
    if "new_spread_raw" not in st.session_state:
        st.session_state.new_spread_raw = ""       
    if "new_spread_val" not in st.session_state:
        st.session_state.new_spread_val = None      
    if "new_spread_error" not in st.session_state:
        st.session_state.new_spread_error = ""

    if "show_hml_input" not in st.session_state:
        st.session_state.show_hml_input = False
    if "new_hml_raw" not in st.session_state:
        st.session_state.new_hml_raw = ""     
    if "new_hml_val" not in st.session_state:
        st.session_state.new_hml_val = None    
    if "new_hml_error" not in st.session_state:
        st.session_state.new_hml_error = ""          

    if "show_aml_input" not in st.session_state:
        st.session_state.show_aml_input = False
    if "new_aml_raw" not in st.session_state:
        st.session_state.new_aml_raw = ""         
    if "new_aml_val" not in st.session_state:
        st.session_state.new_aml_val = None       
    if "new_aml_error" not in st.session_state:
        st.session_state.new_aml_error = ""            

    if "show_over_input" not in st.session_state:
        st.session_state.show_over_input = False
    if "new_over_raw" not in st.session_state:
        st.session_state.new_over_raw = ""        
    if "new_over_val" not in st.session_state:
        st.session_state.new_over_val = None       
    if "new_over_error" not in st.session_state:
        st.session_state.new_over_error = ""        

    

    ## Helper Funcs

    # for when things are input
    def toggle_input():
        st.session_state.show_spread_input = not st.session_state.show_spread_input
        st.session_state.show_hml_input = not st.session_state.show_hml_input
        st.session_state.show_aml_input = not st.session_state.show_aml_input
        st.session_state.show_over_input = not st.session_state.show_over_input
     
    # re-run when needed
    def _force_rerun():
        if hasattr(st, "rerun"):
            st.rerun()
        else:
            _rerun()

    # save the input spread
    def save_new_spread():
        # find current session state val
        raw = (st.session_state.get("new_spread_raw", "") or "").strip()
        # if nothing, dont do anything and end
        if raw == "":
            st.session_state.new_spread_val = None
            st.session_state.new_spread_error = ""
            return
        # if something input, try turning it into a float if it's within acceptable range
        try:
            st.session_state.new_spread_val = float(raw)
            if (st.session_state.new_spread_val > old_spread_upper or st.session_state.new_spread_val < old_spread_lower):
                            st.session_state.new_spread_val = None
                            st.session_state.new_spread_error = "Please enter a more realistic value"
                            return
            st.session_state.new_spread_error = ""
        # return warning if error in input, return to default
        except ValueError:
            st.session_state.new_spread_val = None
            st.session_state.new_spread_error = "Please enter a numeric value"

        
    # save the input home ML
    def save_new_hml():
        # find current session state val
        raw = (st.session_state.get("new_hml_raw", "") or "").strip()
        # if nothing, dont do anything and end
        if raw == "":
            st.session_state.new_hml_val = None
            st.session_state.new_hml_error = ""
            return
        # if something input, try turning it into a float if it's within acceptable range
        try:
            if raw.startswith("+"):
                raw = raw[1:]            
            st.session_state.new_hml_val = float(raw)
            st.session_state.new_hml_error = ""
        # return warning if error in input, return to default
        except ValueError:
            st.session_state.new_hml_val = None
            st.session_state.new_hml_error = "Please enter a numeric value"     

    # save the input away ML
    def save_new_aml():
        # find current session state val
        raw = (st.session_state.get("new_aml_raw", "") or "").strip()
        # if nothing, dont do anything and end
        if raw == "":
            st.session_state.new_aml_val = None
            st.session_state.new_aml_error = ""
            return
        # if something input, try turning it into a float if it's within acceptable range
        try:
            if raw.startswith("+"):
                raw = raw[1:]
            st.session_state.new_aml_val = float(raw)
            st.session_state.new_aml_error = ""
        # return warning if error in input, return to default
        except ValueError:
            st.session_state.new_aml_val = None
            st.session_state.new_aml_error = "Please enter a numeric value"                      

    # save  the input over/under line
    def save_new_over():
        # find current session state val
        raw = (st.session_state.get("new_over_raw", "") or "").strip()
        # if nothing, dont do anything and end
        if raw == "":
            st.session_state.new_over_val = None
            st.session_state.new_over_error = ""
            return
        # if something input, try turning it into a float if it's within acceptable range
        try:
            st.session_state.new_over_val = float(raw)
            if (st.session_state.new_over_val > old_over_upper or st.session_state.new_over_val < old_over_lower):
                            st.session_state.new_over_val = None
                            st.session_state.new_over_error = "Please enter a more realistic value"
                            return            
            st.session_state.new_over_error = ""
        # return warning if error in input, return to default
        except ValueError:
            st.session_state.new_over_val = None
            st.session_state.new_over_error = "Please enter a numeric value"        

    # function to call when everything should be reset
    def reset_lines():
        st.session_state.show_spread_input = False
        st.session_state.new_spread_raw = ""
        st.session_state.new_spread_val = None  
        st.session_state.new_spread_error = ""
        
        st.session_state.show_hml_input = False
        st.session_state.new_hml_raw = ""
        st.session_state.new_hml_val = None  
        st.session_state.new_hml_error = ""  

        st.session_state.show_aml_input = False
        st.session_state.new_aml_raw = ""
        st.session_state.new_aml_val = None  
        st.session_state.new_aml_error = ""          

        st.session_state.show_over_input = False
        st.session_state.new_over_raw = ""
        st.session_state.new_over_val = None  
        st.session_state.new_over_error = ""      
   

   

  

    ## Heading
    st.title(f"Czar's Week {this_week} Picks")

    st.write("Welcome! Here are our picks for the week. You can toggle between the data frame view of all games or the more in-depth view of one game at a time.")

    st.write("PROFITABILITY NOTE: the spread and moneyline models are significantly profitable. The O/U model is not profitable.")

    st.write("More notes on how to use this page to bet are at the bottom")

    st.write(" ")  

    page_choice = st.selectbox(
        "Choose Display",
        ["All Games", "One Game"]
    )

    st.write(" ")
    st.write(" ")



    
    ## All Game View


    if page_choice == "All Games":


            # create layout cols
            col1, col2, col3 = st.columns([2, .5, 8])
            


            ## Filters Col

            with col1:


                st.markdown(
                        """
                        <style>
                        .custom-font {
                            font-size: 30px; font-weight:bold
                        }
                        </style>
                        <div class="custom-font">Filters</div>
                        """,
                        unsafe_allow_html=True)

                st.write(" ")



                # unique teams
                unique_teams = pd.unique(merged_predictions[['Home', 'Away']].values.ravel('K'))
                unique_teams_sorted = sorted(unique_teams)

                # unique conferences
                unique_conf = pd.unique(merged_predictions[['Home Conf', 'Away Conf']].values.ravel('K'))
                unique_conf_sorted = sorted(unique_conf)
                    

                # format dfs
                # display df depends on what bet type user filters for
                this_week_display_df = merged_predictions[["Spread Bet on: ", "ML Bet on: ", "O/U Bet on: ", "Home", "Away", "Spread Value",
                    "ML Value", "O/U Value", "Book Home Moneyline", "Book Away Moneyline", "Home Conf", "Away Conf", "Neutral?"]].sort_values(by="Spread Value", ascending=False)

                this_week_display_spread_df = merged_predictions[["Spread Bet on: ", "Spread Value", "Home", "Away", "Pred Home Cover Prob",
                                                                "Home Conf", "Away Conf", "Neutral?"]].sort_values(by="Spread Value", ascending=False)

                this_week_display_ml_df = merged_predictions[["ML Bet on: ", "ML Value", "Home", "Away", "Book Home Moneyline", "Book Away Moneyline", "Pred Home Win Prob", "Book Home Win Prob", "Pred Away Win Prob", 
                                                            "Book Away Win Prob", "Home Conf", "Away Conf", "Neutral?"]].sort_values(by="ML Value", ascending=False)

                this_week_display_ou_df = merged_predictions[["O/U Bet on: ", "O/U Value", "Home", "Away", "Pred Over Prob", "Home Conf", 
                                                            "Away Conf", "Neutral?"]].sort_values(by="O/U Value", ascending=False)      
            

                
                # bet type options (for display dfs above)
                bet_type_options = st.selectbox(
                    "Bet Type", 
                    options=["All", "Spread", "Moneyline", "Over/Under"],  
                    index=0  
                )

                # team select box
                team_options = st.selectbox(
                    "Team", 
                    options=["All"] + unique_teams_sorted,  
                    index=0  
                )             
                # conference seelct box
                conf_options = st.selectbox(
                    "Conference", 
                    options=["All"] + unique_conf_sorted,  
                    index = 0 
                )
                # bet type sort select box
                if bet_type_options == "All":
                    sort_options = st.selectbox(
                        "Sort Value By", 
                        options=["Default", "Spread", "Moneyline", "Over/Under"],  
                        index=0  
                    )
                else:
                    sort_options = st.selectbox(
                        "Sort Value By", 
                        options=["Default"],  
                        index=0)

                st.write("") 
                st.write("") 
                        


                # choose which df to display   
                if bet_type_options == "All":
                    picks_df = this_week_display_df
                elif bet_type_options == "Spread":
                    picks_df = this_week_display_spread_df
                elif bet_type_options == "Moneyline":
                    picks_df = this_week_display_ml_df
                else:
                    picks_df = this_week_display_ou_df

                # filter/sort display df based on filters
                if team_options != "All":
                    picks_df = picks_df[picks_df['Home'].eq(team_options) | picks_df['Away'].eq(team_options)]
                if conf_options != "All":
                    picks_df = picks_df[picks_df['Home Conf'].eq(conf_options) | picks_df['Away Conf'].eq(conf_options)] 
                if sort_options == "Spread":
                    picks_df.sort_values(by="Spread Value", ascending=False, inplace=True)
                elif sort_options == "Moneyline":
                    picks_df.sort_values(by="ML Value", ascending=False, inplace=True)
                elif sort_options == "Over/Under":
                    picks_df.sort_values(by="O/U Value", ascending=False, inplace=True)

            



            # column for space
            with col2:
                st.write("")



            ## Display Df Col
            
            with col3:

                st.dataframe(picks_df, height = 600)
          
    



    ## Single Game View
    
    else:
            
            
            ## If Mobile View

            if is_mobile:
            
                    # unique matchups
                    unique_matchups = sorted(pd.unique(merged_predictions["matchup"]))

                    st.write("")

                    # matchup select box
                    matchup_options = st.selectbox(
                        "Select Matchup", 
                        options=unique_matchups,  
                        index=0,
                        on_change=reset_lines
                    )    

                    # filter and format df
                    filtered_df = merged_predictions.loc[merged_predictions["matchup"]==matchup_options]  
                    filtered_df["Book Home Moneyline"] = filtered_df["Book Home Moneyline"].fillna(0)
                    filtered_df["Book Away Moneyline"] = filtered_df["Book Away Moneyline"].fillna(0)

                    # limits for manual inputs
                    old_spread_val = filtered_df.iloc[0]["Book Home Spread"]
                    old_spread_upper = old_spread_val+4
                    old_spread_lower = old_spread_val-4
                    old_over_val = filtered_df.iloc[0]["Book O/U"]
                    old_over_upper = old_over_val+5
                    old_over_lower = old_over_val-5   

                    
                    # if just spread values chaned, make new predictions
                    if st.session_state.new_spread_val is not None and st.session_state.new_over_val is None:
                        new_spread = st.session_state.new_spread_val
                        current_home = filtered_df.iloc[0]["Home"]
                        mask = (sample_data["t1_team"].eq(current_home)) | (sample_data["t2_team"].eq(current_home))
                        cut_sample_data = sample_data.loc[mask].copy()
                        cut_sample_data.loc[cut_sample_data["t1_team"].eq(current_home), "t1_book_spread"] = new_spread
                        cut_sample_data.loc[cut_sample_data["t2_team"].eq(current_home), "t1_book_spread"] = -new_spread
                        filtered_df = make_merged_predictions(cut_sample_data)
                    # if just over value is changed, make new predictions
                    elif st.session_state.new_spread_val is None and st.session_state.new_over_val is not None:
                        new_over = st.session_state.new_over_val
                        current_home = filtered_df.iloc[0]["Home"]
                        mask = (sample_data["t1_team"].eq(current_home)) | (sample_data["t2_team"].eq(current_home))
                        cut_sample_data = sample_data.loc[mask].copy()
                        cut_sample_data["book_over_under"] = new_over
                        filtered_df = make_merged_predictions(cut_sample_data)   
                    # if both spread and over value is changed, make new predictions 
                    elif st.session_state.new_spread_val is not None and st.session_state.new_over_val is not None:
                        new_spread = st.session_state.new_spread_val
                        new_over = st.session_state.new_over_val
                        current_home = filtered_df.iloc[0]["Home"]
                        mask = (sample_data["t1_team"].eq(current_home)) | (sample_data["t2_team"].eq(current_home))
                        cut_sample_data = sample_data.loc[mask].copy()
                        cut_sample_data.loc[cut_sample_data["t1_team"].eq(current_home), "t1_book_spread"] = new_spread
                        cut_sample_data.loc[cut_sample_data["t2_team"].eq(current_home), "t1_book_spread"] = -new_spread
                        cut_sample_data["book_over_under"] = new_over
                        filtered_df = make_merged_predictions(cut_sample_data) 


                    # if any of the lines are manually input, change the values in the df, depending on what was changed
                    if st.session_state.new_spread_val is not None or st.session_state.new_hml_val is not None or st.session_state.new_aml_val or st.session_state.new_over_val is not None:

                        # find game
                        current_home = filtered_df.iloc[0]["Home"]
                        mask = (sample_data["t1_team"].eq(current_home)) | (sample_data["t2_team"].eq(current_home))
                        cut_sample_data = sample_data.loc[mask].copy()
                        # if spread was changed, change in prediction prep df
                        if st.session_state.new_spread_val is not None:
                            new_spread = st.session_state.new_spread_val
                            cut_sample_data.loc[cut_sample_data["t1_team"].eq(current_home), "t1_book_spread"] = new_spread
                            cut_sample_data.loc[cut_sample_data["t2_team"].eq(current_home), "t1_book_spread"] = -new_spread
                        # if home moneyline was changed, change in prediction prep df
                        if st.session_state.new_hml_val is not None:
                            new_hml = st.session_state.new_hml_val
                            cut_sample_data.loc[cut_sample_data["t1_team"].eq(current_home), "t1_moneyline"] = new_hml
                            cut_sample_data.loc[cut_sample_data["t2_team"].eq(current_home), "t2_moneyline"] = new_hml   
                        # if away moneyline was changed, change in prediction prep df
                        if st.session_state.new_aml_val is not None:
                            new_aml = st.session_state.new_aml_val
                            cut_sample_data.loc[cut_sample_data["t1_team"].eq(current_home), "t2_moneyline"] = new_aml
                            cut_sample_data.loc[cut_sample_data["t2_team"].eq(current_home), "t1_moneyline"] = new_aml    
                        # if over/under line was changed, change in prediction prep df                   
                        if st.session_state.new_over_val is not None:
                            new_over = st.session_state.new_over_val
                            cut_sample_data["book_over_under"] = new_over
                        # make new predictions
                        filtered_df = make_merged_predictions(cut_sample_data) 
                

                    # create columns for layout
                    col1, col2, col3, col4, col5, col6, col7 = st.columns([3, .5, 3, .5, 3, .5, 3]) 




                    ## Second Col Displayed (Away)

                    with col3:

                            
                            st.write("")

                            # logo
                            st.markdown("<div style='display: flex; justify-content: center;'><img src='" + filtered_df.iloc[0]["away_logo"] + "' width='150'></div>", unsafe_allow_html=True)
                    
                            st.markdown(f"""
                                        <div style="font-size:35px; font-weight:bold; color:{filtered_df.iloc[0]["away_color"]}; text-align:center;">
                                            {filtered_df.iloc[0]["Away"]}
                                        </div>
                                    """, unsafe_allow_html=True)
                            
                            st.write(" ")
                            st.write(" ")
                            

                            # lines, predictions, and value
                            st.markdown(f"""
                                        <div style="font-size:20px; color: gray; text-align:center;">
                                            Pred Cover Prob: {100-filtered_df.iloc[0]["Pred Home Cover Prob"]}%
                                        </div>
                                    """, unsafe_allow_html=True)  
                                

                            if filtered_df.iloc[0]["Book Home Spread"] < 0:
                                st.markdown(f"""
                                            <div style="font-size:20px; color: gray; text-align:center;">
                                                Book Spread: +{filtered_df.iloc[0]["Book Home Spread"]*-1}
                                            </div>
                                        """, unsafe_allow_html=True)
                            else:
                                st.markdown(f"""
                                            <div style="font-size:20px; color: gray; text-align:center;">
                                                Book Spread: {filtered_df.iloc[0]["Book Home Spread"]*-1}
                                            </div>
                                        """, unsafe_allow_html=True)                             

                            if filtered_df.iloc[0]["Pred Home Cover Prob"] <= 50:
                                st.markdown(f"""
                                            <div style="font-size:20px; color: green; font-weight:bold; text-align:center;">
                                                To Cover Value: {filtered_df.iloc[0]["Spread Value"]}%
                                            </div>
                                        """, unsafe_allow_html=True)   
                            else:
                                st.markdown(f"""
                                            <div style="font-size:20px; color: red; font-weight:bold; text-align:center;">
                                                To Cover Value: {filtered_df.iloc[0]["Spread Value"]*-1}%
                                            </div>
                                        """, unsafe_allow_html=True)    
                            
                            st.write(" ")
                            st.write(" ")


                            if filtered_df.iloc[0]["Pred Away Moneyline"] > 0:
                                st.markdown(f"""
                                            <div style="font-size:20px; color: gray; text-align:center;">
                                                Pred ML: +{int(round(filtered_df.iloc[0]["Pred Away Moneyline"],0))}
                                            </div>
                                        """, unsafe_allow_html=True)   
                            else:
                                st.markdown(f"""
                                            <div style="font-size:20px; color: gray; text-align:center;">
                                                Pred ML: {int(round(filtered_df.iloc[0]["Pred Away Moneyline"],0))}
                                            </div>
                                        """, unsafe_allow_html=True)                           

                            st.markdown(f"""
                                        <div style="font-size:20px; color: gray; text-align:center;">
                                            Pred Win Prob: {round(filtered_df.iloc[0]["Pred Away Win Prob"],2)}%
                                        </div>
                                    """, unsafe_allow_html=True)                         

                            if filtered_df.iloc[0]["Book Away Moneyline"]>0:
                                    st.markdown(f"""
                                                <div style="font-size:20px; color: gray; text-align:center;">
                                                    Book ML: +{int(round(filtered_df.iloc[0]["Book Away Moneyline"],0))}
                                                </div>
                                            """, unsafe_allow_html=True)  
                            else:  
                                    st.markdown(f"""
                                                <div style="font-size:20px; color: gray; text-align:center;">
                                                    Book ML: {int(round(filtered_df.iloc[0]["Book Away Moneyline"],0))}
                                                </div>
                                            """, unsafe_allow_html=True)                             

                            st.markdown(f"""
                                        <div style="font-size:20px; color: gray; text-align:center;">
                                            Book Imp Win Prob: {round(filtered_df.iloc[0]["Book Away Win Prob"],2)}%
                                        </div>
                                    """, unsafe_allow_html=True)

                            if filtered_df.iloc[0]["Pred Away Win Prob"] - filtered_df.iloc[0]["Book Away Win Prob"] > filtered_df.iloc[0]["Pred Home Win Prob"] - filtered_df.iloc[0]["Book Home Win Prob"]:
                                if filtered_df.iloc[0]["ML Value"] > 0:
                                    st.markdown(f"""
                                                <div style="font-size:20px; color: green; font-weight:bold; text-align:center;">
                                                    To Hit Value: {filtered_df.iloc[0]["ML Value"]}%
                                                </div>
                                            """, unsafe_allow_html=True)
                                else:
                                    st.markdown(f"""
                                                <div style="font-size:20px; color: red; font-weight:bold; text-align:center;">
                                                    To Hit Value: {filtered_df.iloc[0]["ML Value"]}%
                                                </div>
                                            """, unsafe_allow_html=True)
                            else:
                                st.markdown(f"""
                                            <div style="font-size:20px; color: red; font-weight:bold; text-align:center;">
                                                To Hit Value: {filtered_df.iloc[0]["ML Losing Value"]}%
                                            </div>
                                        """, unsafe_allow_html=True)
                                

        



                    # col for space
                    with col2:
                        st.write("")


                    ## First Display Col (what to bet on)

                    with col1:

                        st.write(" ")

                        # what should we bet on
                        st.markdown(f"""
                                    <div style="font-size:25px; color: gold; font-weight:bold; text-align:center;">
                                        Bet on:
                                    </div>
                                """, unsafe_allow_html=True)   

                        st.write(" ")  

                        if filtered_df.iloc[0]["Spread Bet on:"] == filtered_df.iloc[0]["Home"]:
                            if filtered_df.iloc[0]["Book Home Spread"] > 0:
                                st.markdown(f"""
                                            <div style="font-size:25px; color: gold; font-weight:bold; text-align:center;">
                                                {filtered_df.iloc[0]["Spread Bet on:"]} +{filtered_df.iloc[0]["Book Home Spread"]}
                                            </div>
                                        """, unsafe_allow_html=True)    
                            else:
                                st.markdown(f"""
                                            <div style="font-size:25px; color: gold; font-weight:bold; text-align:center;">
                                                {filtered_df.iloc[0]["Spread Bet on:"]} {filtered_df.iloc[0]["Book Home Spread"]}
                                            </div>
                                        """, unsafe_allow_html=True)                           
                        else:           
                            if filtered_df.iloc[0]["Book Home Spread"]*-1 > 0:
                                st.markdown(f"""
                                            <div style="font-size:25px; color: gold; font-weight:bold; text-align:center;">
                                                {filtered_df.iloc[0]["Spread Bet on:"]} +{filtered_df.iloc[0]["Book Home Spread"]*-1}
                                            </div>
                                        """, unsafe_allow_html=True)  
                            else:
                                st.markdown(f"""
                                            <div style="font-size:25px; color: gold; font-weight:bold; text-align:center;">
                                                {filtered_df.iloc[0]["Spread Bet on:"]} {filtered_df.iloc[0]["Book Home Spread"]*-1}
                                            </div>
                                        """, unsafe_allow_html=True)                     
                            
                        if filtered_df.iloc[0]["ML Bet on:"] == filtered_df.iloc[0]["Home"]:
                            if filtered_df.iloc[0]["Book Home Moneyline"]>0:
                                st.markdown(f"""
                                        <div style="font-size:25px; color: gold; font-weight:bold; text-align:center;">
                                            {filtered_df.iloc[0]["ML Bet on:"]} +{int(filtered_df.iloc[0]["Book Home Moneyline"])}
                                        </div>
                                    """, unsafe_allow_html=True) 
                            else:
                                st.markdown(f"""
                                        <div style="font-size:25px; color: gold; font-weight:bold; text-align:center;">
                                            {filtered_df.iloc[0]["ML Bet on:"]} {int(filtered_df.iloc[0]["Book Home Moneyline"])}
                                        </div>
                                    """, unsafe_allow_html=True)                            
                        elif filtered_df.iloc[0]["ML Bet on:"] == filtered_df.iloc[0]["Away"]:  
                            if filtered_df.iloc[0]["Book Away Moneyline"]>0:
                                st.markdown(f"""
                                        <div style="font-size:25px; color: gold; font-weight:bold; text-align:center;">
                                            {filtered_df.iloc[0]["ML Bet on:"]} +{int(filtered_df.iloc[0]["Book Away Moneyline"])}
                                        </div>
                                    """, unsafe_allow_html=True)                      
                            else:
                                st.markdown(f"""
                                        <div style="font-size:25px; color: gold; font-weight:bold; text-align:center;">
                                            {filtered_df.iloc[0]["ML Bet on:"]} {int(filtered_df.iloc[0]["Book Away Moneyline"])}
                                        </div>
                                    """, unsafe_allow_html=True) 
                        else:
                            st.markdown(f"""
                                    <div style="font-size:25px; color: gray; font-weight:bold; text-align:center;">
                                        No ML Bet
                                    </div>
                                """, unsafe_allow_html=True)

                        st.markdown(f"""
                                    <div style="font-size:25px; color: gold; font-weight:bold; text-align:center;">
                                        {filtered_df.iloc[0]["O/U Bet on:"]} {filtered_df.iloc[0]["Book O/U"]}
                                    </div>
                                """, unsafe_allow_html=True) 


                        st.write(" ")                    
                        st.write(" ") 
                        st.write(" ") 
             

                        


                        # change lines button
                        st.markdown(
                            """
                            <style>
                            div[data-testid="stButton"] {display: flex; justify-content: center;}
                            </style>
                            """,
                            unsafe_allow_html=True
                        )
                        #st.button("Change book lines?", on_click=toggle_input)
                        
                        # change spread
                        if st.session_state.show_spread_input:
                            st.text_input(
                                "Enter new home book spread:",
                                key="new_spread_raw",
                                placeholder="e.g., -3.5",
                                on_change=save_new_spread,  
                            )
                            if st.session_state.new_spread_error:
                                st.warning(st.session_state.new_spread_error)
                        # change home ML
                        if st.session_state.show_hml_input:
                            st.text_input(
                                "Enter new book home ml:",
                                key="new_hml_raw",
                                placeholder="e.g., -230, 300",
                                on_change=save_new_hml,   # <- stores float into session + reruns
                            )
                            if st.session_state.new_hml_error:
                                st.warning(st.session_state.new_hml_error)  
                        # change away ML
                        if st.session_state.show_aml_input:
                            st.text_input(
                                "Enter new book away ml:",
                                key="new_aml_raw",
                                placeholder="e.g., -230, 300",
                                on_change=save_new_aml,   # <- stores float into session + reruns
                            )
                            if st.session_state.new_aml_error:
                                st.warning(st.session_state.new_aml_error)         
                        # change over/under line
                        if st.session_state.show_over_input:
                            st.text_input(
                                "Enter new book over under:",
                                key="new_over_raw",
                                placeholder="e.g., 48.5",
                                on_change=save_new_over,   # <- stores float into session + reruns
                            )
                            if st.session_state.new_over_error:
                                st.warning(st.session_state.new_over_error)  

                        # reset lines button
                        if st.session_state.show_spread_input or st.session_state.show_hml_input or st.session_state.show_aml_input or st.session_state.show_over_input:
                            st.markdown(
                                """
                                <style>
                                div[data-testid="stButton"] {display: flex; justify-content: center;}
                                </style>
                                """,
                                unsafe_allow_html=True
                            )
                            st.button("Reset Lines", on_click=reset_lines)         


                        st.write(" ")                    
                        st.write(" ") 
                        st.write(" ")                                        





                    # col for space
                    with col4:
                        st.write("")


                    ## Third Display Col (Home)

                    with col5:

                        st.write("") 

                        # logo
                        st.markdown("<div style='display: flex; justify-content: center;'><img src='" + filtered_df.iloc[0]["home_logo"] + "' width='150'></div>", unsafe_allow_html=True)
                                            
                        st.markdown(f"""
                                    <div style="font-size:35px; font-weight:bold; color:{filtered_df.iloc[0]["home_color"]}; text-align:center;">
                                        {filtered_df.iloc[0]["Home"]}
                                    </div>
                                """, unsafe_allow_html=True)       

                        st.write(" ")
                        st.write(" ")


                        # lines, predictions, and value
                        st.markdown(f"""
                                    <div style="font-size:20px; color: gray; text-align:center;">
                                        Pred Cover Prob: {filtered_df.iloc[0]["Pred Home Cover Prob"]}%
                                    </div>
                                """, unsafe_allow_html=True)  
                        
                                                    

                        if filtered_df.iloc[0]["Book Home Spread"] > 0:
                            st.markdown(f"""
                                        <div style="font-size:20px; color: gray; text-align:center;">
                                            Book Spread: +{filtered_df.iloc[0]["Book Home Spread"]}
                                        </div>
                                    """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                                        <div style="font-size:20px; color: gray; text-align:center;">
                                            Book Spread: {filtered_df.iloc[0]["Book Home Spread"]}
                                        </div>
                                    """, unsafe_allow_html=True)                             

                        if filtered_df.iloc[0]["Pred Home Cover Prob"] >= 50:
                            st.markdown(f"""
                                        <div style="font-size:20px; color: green; font-weight:bold; text-align:center;">
                                            To Cover Value: {filtered_df.iloc[0]["Spread Value"]}%
                                        </div>
                                    """, unsafe_allow_html=True)   
                        else:
                                st.markdown(f"""
                                        <div style="font-size:20px; color: red; font-weight:bold; text-align:center;">
                                            To Cover Value: {filtered_df.iloc[0]["Spread Value"]*-1}%
                                        </div>
                                    """, unsafe_allow_html=True)    
                        
                        st.write(" ")
                        st.write(" ")


                        if filtered_df.iloc[0]["Pred Home Moneyline"] > 0:
                            st.markdown(f"""
                                        <div style="font-size:20px; color: gray; text-align:center;">
                                            Pred ML: +{int(round(filtered_df.iloc[0]["Pred Home Moneyline"],0))}
                                        </div>
                                    """, unsafe_allow_html=True)   
                        else:
                            st.markdown(f"""
                                        <div style="font-size:20px; color: gray; text-align:center;">
                                            Pred ML: {int(round(filtered_df.iloc[0]["Pred Home Moneyline"],0))}
                                        </div>
                                    """, unsafe_allow_html=True)                           

                        st.markdown(f"""
                                    <div style="font-size:20px; color: gray; text-align:center;">
                                        Pred Win Prob: {round(filtered_df.iloc[0]["Pred Home Win Prob"],2)}%
                                    </div>
                                """, unsafe_allow_html=True)                         

                        if filtered_df.iloc[0]["Book Home Moneyline"]>0:
                                st.markdown(f"""
                                            <div style="font-size:20px; color: gray; text-align:center;">
                                                Book ML: +{int(round(filtered_df.iloc[0]["Book Home Moneyline"],0))}
                                            </div>
                                        """, unsafe_allow_html=True)  
                        else:  
                                st.markdown(f"""
                                            <div style="font-size:20px; color: gray; text-align:center;">
                                                Book ML: {int(round(filtered_df.iloc[0]["Book Home Moneyline"],0))}
                                            </div>
                                        """, unsafe_allow_html=True)                             

                        st.markdown(f"""
                                    <div style="font-size:20px; color: gray; text-align:center;">
                                        Book Imp Win Prob: {round(filtered_df.iloc[0]["Book Home Win Prob"],2)}%
                                    </div>
                                """, unsafe_allow_html=True)

                        if filtered_df.iloc[0]["Pred Home Win Prob"] - filtered_df.iloc[0]["Book Home Win Prob"] > filtered_df.iloc[0]["Pred Away Win Prob"] - filtered_df.iloc[0]["Book Away Win Prob"]:
                            if filtered_df.iloc[0]["ML Value"]>0:
                                st.markdown(f"""
                                            <div style="font-size:20px; color: green; font-weight:bold; text-align:center;">
                                                To Hit Value: {filtered_df.iloc[0]["ML Value"]}%
                                            </div>
                                        """, unsafe_allow_html=True)  
                            else:
                                st.markdown(f"""
                                            <div style="font-size:20px; color: red; font-weight:bold; text-align:center;">
                                                To Hit Value: {filtered_df.iloc[0]["ML Value"]}%
                                            </div>
                                        """, unsafe_allow_html=True)  
                        else:
                                st.markdown(f"""
                                        <div style="font-size:20px; color: red; font-weight:bold; text-align:center;">
                                            To Hit Value: {filtered_df.iloc[0]["ML Losing Value"]}% 
                                        </div>
                                    """, unsafe_allow_html=True)   
                    


                    # col for space
                    with col6:
                        st.write("")





                    ## Last Display Col (over/under)

                    with col7:
                            st.markdown(f"""
                                    <div style="font-size:35px; font-weight:bold; color:orange; text-align:center;">
                                        Over/Under
                                    </div>
                                """, unsafe_allow_html=True)       

                            st.write(" ")
                            st.write(" ")


                            # predictions, lines, and value
                            st.markdown(f"""
                                    <div style="font-size:20px; color: gray; text-align:center;">
                                        Pred Over Prob: {filtered_df.iloc[0]["Pred Over Prob"]}%
                                    </div>
                                """, unsafe_allow_html=True)

                            st.markdown(f"""
                                    <div style="font-size:20px; color: gray; text-align:center;">
                                        Book O/U: {filtered_df.iloc[0]["Book O/U"]}
                                    </div>
                                """, unsafe_allow_html=True)
                            
                            if filtered_df.iloc[0]["Pred Over Prob"] > 50:
                                st.markdown(f"""
                                        <div style="font-size:20px; color: green; font-weight:bold; text-align:center;">
                                            Over Value: {filtered_df.iloc[0]["O/U Value"]}%
                                        </div>
                                    """, unsafe_allow_html=True)
                                st.markdown(f"""
                                        <div style="font-size:20px; color: red; font-weight:bold; text-align:center;">
                                            Under Value: {filtered_df.iloc[0]["O/U Value"]*-1}%
                                        </div>
                                    """, unsafe_allow_html=True)
                            else:
                                st.markdown(f"""
                                        <div style="font-size:20px; color: red; font-weight:bold; text-align:center;">
                                            Over Value: {filtered_df.iloc[0]["O/U Value"]*-1}%
                                        </div>
                                    """, unsafe_allow_html=True)
                                st.markdown(f"""
                                        <div style="font-size:20px; color: green; font-weight:bold; text-align:center;">
                                            Under Value: {filtered_df.iloc[0]["O/U Value"]}%
                                        </div>
                                    """, unsafe_allow_html=True)
                                


                
            ## Desktop Version
                
            else:    
            

                    # unique matchups
                    unique_matchups = sorted(pd.unique(merged_predictions["matchup"]))

                    st.write("")

                    # matchup filter
                    matchup_options = st.selectbox(
                        "Select Matchup", 
                        options=unique_matchups,  
                        index=0,
                        on_change=reset_lines
                    )    

                    # formatting filtered df
                    filtered_df = merged_predictions.loc[merged_predictions["matchup"]==matchup_options] 
                    filtered_df["Book Home Moneyline"] = filtered_df["Book Home Moneyline"].fillna(0)
                    filtered_df["Book Away Moneyline"] = filtered_df["Book Away Moneyline"].fillna(0)

                    # limits for manual inputs (currently not in app buyt potentially in future) 
                    old_spread_val = filtered_df.iloc[0]["Book Home Spread"]
                    old_spread_upper = old_spread_val+4
                    old_spread_lower = old_spread_val-4
                    old_over_val = filtered_df.iloc[0]["Book O/U"]
                    old_over_upper = old_over_val+5
                    old_over_lower = old_over_val-5   

                    
                    # currently not in app, but manual input line directions for making new preds
                    # if only new spread val
                    if st.session_state.new_spread_val is not None and st.session_state.new_over_val is None:
                        new_spread = st.session_state.new_spread_val
                        current_home = filtered_df.iloc[0]["Home"]
                        mask = (sample_data["t1_team"].eq(current_home)) | (sample_data["t2_team"].eq(current_home))
                        cut_sample_data = sample_data.loc[mask].copy()
                        cut_sample_data.loc[cut_sample_data["t1_team"].eq(current_home), "t1_book_spread"] = new_spread
                        cut_sample_data.loc[cut_sample_data["t2_team"].eq(current_home), "t1_book_spread"] = -new_spread
                        filtered_df = make_merged_predictions(cut_sample_data)
                    # if only new over val
                    elif st.session_state.new_spread_val is None and st.session_state.new_over_val is not None:
                        new_over = st.session_state.new_over_val
                        current_home = filtered_df.iloc[0]["Home"]
                        mask = (sample_data["t1_team"].eq(current_home)) | (sample_data["t2_team"].eq(current_home))
                        cut_sample_data = sample_data.loc[mask].copy()
                        cut_sample_data["book_over_under"] = new_over
                        filtered_df = make_merged_predictions(cut_sample_data)    
                    # if new spread and over vals
                    elif st.session_state.new_spread_val is not None and st.session_state.new_over_val is not None:
                        new_spread = st.session_state.new_spread_val
                        new_over = st.session_state.new_over_val
                        current_home = filtered_df.iloc[0]["Home"]
                        mask = (sample_data["t1_team"].eq(current_home)) | (sample_data["t2_team"].eq(current_home))
                        cut_sample_data = sample_data.loc[mask].copy()
                        cut_sample_data.loc[cut_sample_data["t1_team"].eq(current_home), "t1_book_spread"] = new_spread
                        cut_sample_data.loc[cut_sample_data["t2_team"].eq(current_home), "t1_book_spread"] = -new_spread
                        cut_sample_data["book_over_under"] = new_over
                        filtered_df = make_merged_predictions(cut_sample_data) 


                    # if new inputs, make predictions and change df accordingly
                    if st.session_state.new_spread_val is not None or st.session_state.new_hml_val is not None or st.session_state.new_aml_val or st.session_state.new_over_val is not None:
                        current_home = filtered_df.iloc[0]["Home"]
                        mask = (sample_data["t1_team"].eq(current_home)) | (sample_data["t2_team"].eq(current_home))
                        cut_sample_data = sample_data.loc[mask].copy()
                        if st.session_state.new_spread_val is not None:
                            new_spread = st.session_state.new_spread_val
                            cut_sample_data.loc[cut_sample_data["t1_team"].eq(current_home), "t1_book_spread"] = new_spread
                            cut_sample_data.loc[cut_sample_data["t2_team"].eq(current_home), "t1_book_spread"] = -new_spread
                        if st.session_state.new_hml_val is not None:
                            new_hml = st.session_state.new_hml_val
                            cut_sample_data.loc[cut_sample_data["t1_team"].eq(current_home), "t1_moneyline"] = new_hml
                            cut_sample_data.loc[cut_sample_data["t2_team"].eq(current_home), "t2_moneyline"] = new_hml   
                        if st.session_state.new_aml_val is not None:
                            new_aml = st.session_state.new_aml_val
                            cut_sample_data.loc[cut_sample_data["t1_team"].eq(current_home), "t2_moneyline"] = new_aml
                            cut_sample_data.loc[cut_sample_data["t2_team"].eq(current_home), "t1_moneyline"] = new_aml                       
                        if st.session_state.new_over_val is not None:
                            new_over = st.session_state.new_over_val
                            cut_sample_data["book_over_under"] = new_over
                        filtered_df = make_merged_predictions(cut_sample_data) 
                

                    # create columns for layout
                    col1, col2, col3, col4, col5 = st.columns([3, .5, 3, .5, 3]) 



                    # First Display Col (Away team)

                    with col1:

                            
                            st.write("")



                            # logo
                            st.markdown("<div style='display: flex; justify-content: left;'><img src='" + filtered_df.iloc[0]["away_logo"] + "' width='150'></div>", unsafe_allow_html=True)
                    
                            st.markdown(f"""
                                        <div style="font-size:35px; font-weight:bold; color:{filtered_df.iloc[0]["away_color"]}; text-align:left;">
                                            {filtered_df.iloc[0]["Away"]}
                                        </div>
                                    """, unsafe_allow_html=True)
                            
                            st.write(" ")
                            st.write(" ")


                            # predictions, lines, and value

                            st.markdown(f"""
                                        <div style="font-size:20px; color: gray; text-align:left;">
                                            Pred Cover Prob: {100-filtered_df.iloc[0]["Pred Home Cover Prob"]}%
                                        </div>
                                    """, unsafe_allow_html=True)  
                                

                            if filtered_df.iloc[0]["Book Home Spread"] < 0:
                                st.markdown(f"""
                                            <div style="font-size:20px; color: gray; text-align:left;">
                                                Book Spread: +{filtered_df.iloc[0]["Book Home Spread"]*-1}
                                            </div>
                                        """, unsafe_allow_html=True)
                            else:
                                st.markdown(f"""
                                            <div style="font-size:20px; color: gray; text-align:left;">
                                                Book Spread: {filtered_df.iloc[0]["Book Home Spread"]*-1}
                                            </div>
                                        """, unsafe_allow_html=True)                             

                            if filtered_df.iloc[0]["Pred Home Cover Prob"] <= 50:
                                st.markdown(f"""
                                            <div style="font-size:20px; color: green; font-weight:bold; text-align:left;">
                                                To Cover Value: {filtered_df.iloc[0]["Spread Value"]}%
                                            </div>
                                        """, unsafe_allow_html=True)   
                            else:
                                st.markdown(f"""
                                            <div style="font-size:20px; color: red; font-weight:bold; text-align:left;">
                                                To Cover Value: {filtered_df.iloc[0]["Spread Value"]*-1}%
                                            </div>
                                        """, unsafe_allow_html=True)    
                            
                            st.write(" ")
                            st.write(" ")


                            if filtered_df.iloc[0]["Pred Away Moneyline"] > 0:
                                st.markdown(f"""
                                            <div style="font-size:20px; color: gray; text-align:left;">
                                                Pred ML: +{int(round(filtered_df.iloc[0]["Pred Away Moneyline"],0))}
                                            </div>
                                        """, unsafe_allow_html=True)   
                            else:
                                st.markdown(f"""
                                            <div style="font-size:20px; color: gray; text-align:left;">
                                                Pred ML: {int(round(filtered_df.iloc[0]["Pred Away Moneyline"],0))}
                                            </div>
                                        """, unsafe_allow_html=True)                           

                            st.markdown(f"""
                                        <div style="font-size:20px; color: gray; text-align:left;">
                                            Pred Win Prob: {round(filtered_df.iloc[0]["Pred Away Win Prob"],2)}%
                                        </div>
                                    """, unsafe_allow_html=True)                         

                            if filtered_df.iloc[0]["Book Away Moneyline"]>0:
                                    st.markdown(f"""
                                                <div style="font-size:20px; color: gray; text-align:left;">
                                                    Book ML: +{int(round(filtered_df.iloc[0]["Book Away Moneyline"],0))}
                                                </div>
                                            """, unsafe_allow_html=True)  
                            else:  
                                    st.markdown(f"""
                                                <div style="font-size:20px; color: gray; text-align:left;">
                                                    Book ML: {int(round(filtered_df.iloc[0]["Book Away Moneyline"],0))}
                                                </div>
                                            """, unsafe_allow_html=True)                             

                            st.markdown(f"""
                                        <div style="font-size:20px; color: gray; text-align:left;">
                                            Book Imp Win Prob: {round(filtered_df.iloc[0]["Book Away Win Prob"],2)}%
                                        </div>
                                    """, unsafe_allow_html=True)

                            if filtered_df.iloc[0]["Pred Away Win Prob"] - filtered_df.iloc[0]["Book Away Win Prob"] > filtered_df.iloc[0]["Pred Home Win Prob"] - filtered_df.iloc[0]["Book Home Win Prob"]:
                                if filtered_df.iloc[0]["ML Value"] > 0:
                                    st.markdown(f"""
                                                <div style="font-size:20px; color: green; font-weight:bold; text-align:left;">
                                                    To Hit Value: {filtered_df.iloc[0]["ML Value"]}%
                                                </div>
                                            """, unsafe_allow_html=True)
                                else:
                                    st.markdown(f"""
                                                <div style="font-size:20px; color: red; font-weight:bold; text-align:left;">
                                                    To Hit Value: {filtered_df.iloc[0]["ML Value"]}%
                                                </div>
                                            """, unsafe_allow_html=True)
                            else:
                                st.markdown(f"""
                                            <div style="font-size:20px; color: red; font-weight:bold; text-align:left;">
                                                To Hit Value: {filtered_df.iloc[0]["ML Losing Value"]}%
                                            </div>
                                        """, unsafe_allow_html=True)
                                

        

                    # col for space
                    with col2:
                        st.write("")



                    ## Second Display Col (what to bet on and over/under)

                    with col3:

                        st.write("")


                        # bet recommendations
                        st.markdown(f"""
                                    <div style="font-size:25px; color: gold; font-weight:bold; text-align:center;">
                                        Bet on:
                                    </div>
                                """, unsafe_allow_html=True)   

                        st.write(" ")  

                        if filtered_df.iloc[0]["Spread Bet on:"] == filtered_df.iloc[0]["Home"]:
                            if filtered_df.iloc[0]["Book Home Spread"] > 0:
                                st.markdown(f"""
                                            <div style="font-size:25px; color: gold; font-weight:bold; text-align:center;">
                                                {filtered_df.iloc[0]["Spread Bet on:"]} +{filtered_df.iloc[0]["Book Home Spread"]}
                                            </div>
                                        """, unsafe_allow_html=True)    
                            else:
                                st.markdown(f"""
                                            <div style="font-size:25px; color: gold; font-weight:bold; text-align:center;">
                                                {filtered_df.iloc[0]["Spread Bet on:"]} {filtered_df.iloc[0]["Book Home Spread"]}
                                            </div>
                                        """, unsafe_allow_html=True)                           
                        else:           
                            if filtered_df.iloc[0]["Book Home Spread"]*-1 > 0:
                                st.markdown(f"""
                                            <div style="font-size:25px; color: gold; font-weight:bold; text-align:center;">
                                                {filtered_df.iloc[0]["Spread Bet on:"]} +{filtered_df.iloc[0]["Book Home Spread"]*-1}
                                            </div>
                                        """, unsafe_allow_html=True)  
                            else:
                                st.markdown(f"""
                                            <div style="font-size:25px; color: gold; font-weight:bold; text-align:center;">
                                                {filtered_df.iloc[0]["Spread Bet on:"]} {filtered_df.iloc[0]["Book Home Spread"]*-1}
                                            </div>
                                        """, unsafe_allow_html=True)                     
                            
                        if filtered_df.iloc[0]["ML Bet on:"] == filtered_df.iloc[0]["Home"]:
                            if filtered_df.iloc[0]["Book Home Moneyline"]>0:
                                st.markdown(f"""
                                        <div style="font-size:25px; color: gold; font-weight:bold; text-align:center;">
                                            {filtered_df.iloc[0]["ML Bet on:"]} +{int(filtered_df.iloc[0]["Book Home Moneyline"])}
                                        </div>
                                    """, unsafe_allow_html=True) 
                            else:
                                st.markdown(f"""
                                        <div style="font-size:25px; color: gold; font-weight:bold; text-align:center;">
                                            {filtered_df.iloc[0]["ML Bet on:"]} {int(filtered_df.iloc[0]["Book Home Moneyline"])}
                                        </div>
                                    """, unsafe_allow_html=True)                            
                        elif filtered_df.iloc[0]["ML Bet on:"] == filtered_df.iloc[0]["Away"]:  
                            if filtered_df.iloc[0]["Book Away Moneyline"]>0:
                                st.markdown(f"""
                                        <div style="font-size:25px; color: gold; font-weight:bold; text-align:center;">
                                            {filtered_df.iloc[0]["ML Bet on:"]} +{int(filtered_df.iloc[0]["Book Away Moneyline"])}
                                        </div>
                                    """, unsafe_allow_html=True)                      
                            else:
                                st.markdown(f"""
                                        <div style="font-size:25px; color: gold; font-weight:bold; text-align:center;">
                                            {filtered_df.iloc[0]["ML Bet on:"]} {int(filtered_df.iloc[0]["Book Away Moneyline"])}
                                        </div>
                                    """, unsafe_allow_html=True) 
                        else:
                            st.markdown(f"""
                                    <div style="font-size:25px; color: gray; font-weight:bold; text-align:center;">
                                        No ML Bet
                                    </div>
                                """, unsafe_allow_html=True)

                        st.markdown(f"""
                                    <div style="font-size:25px; color: gold; font-weight:bold; text-align:center;">
                                        {filtered_df.iloc[0]["O/U Bet on:"]} {filtered_df.iloc[0]["Book O/U"]}
                                    </div>
                                """, unsafe_allow_html=True) 


                        st.write(" ")                    
                        st.write(" ") 
                        st.write(" ") 
                        st.write(" ") 
                        st.write(" ")                    
                        st.write(" ") 
                        st.write(" ") 
                        st.write(" ") 
                        st.write(" ")                    
                        st.write(" ") 
                        st.write(" ") 
                        st.write(" ")   
                        st.write(" ") 
                        st.write(" ")                



                        # over/under lines, predictions, and value

                        st.markdown(f"""
                                <div style="font-size:20px; color: gray; text-align:center;">
                                    Pred Over Prob: {filtered_df.iloc[0]["Pred Over Prob"]}%
                                </div>
                            """, unsafe_allow_html=True)

                        st.markdown(f"""
                                <div style="font-size:20px; color: gray; text-align:center;">
                                    Book O/U: {filtered_df.iloc[0]["Book O/U"]}
                                </div>
                            """, unsafe_allow_html=True)
                        
                        if filtered_df.iloc[0]["Pred Over Prob"] > 50:
                            st.markdown(f"""
                                    <div style="font-size:20px; color: green; font-weight:bold; text-align:center;">
                                        Over Value: {filtered_df.iloc[0]["O/U Value"]}%
                                    </div>
                                """, unsafe_allow_html=True)
                            st.markdown(f"""
                                    <div style="font-size:20px; color: red; font-weight:bold; text-align:center;">
                                        Under Value: {filtered_df.iloc[0]["O/U Value"]*-1}%
                                    </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                                    <div style="font-size:20px; color: red; font-weight:bold; text-align:center;">
                                        Over Value: {filtered_df.iloc[0]["O/U Value"]*-1}%
                                    </div>
                                """, unsafe_allow_html=True)
                            st.markdown(f"""
                                    <div style="font-size:20px; color: green; font-weight:bold; text-align:center;">
                                        Under Value: {filtered_df.iloc[0]["O/U Value"]}%
                                    </div>
                                """, unsafe_allow_html=True)
                            



                            st.write(" ")
                            st.write(" ")
                            st.write(" ")
                            




                        # button to toggle manual inputs
                        st.markdown(
                            """
                            <style>
                            div[data-testid="stButton"] {display: flex; justify-content: center;}
                            </style>
                            """,
                            unsafe_allow_html=True
                        )
                        #st.button("Change book lines?", on_click=toggle_input)

                        # spread input
                        if st.session_state.show_spread_input:
                            st.text_input(
                                "Enter new book spread:",
                                key="new_spread_raw",
                                placeholder="e.g., -3.5",
                                on_change=save_new_spread,   # <- stores float into session + reruns
                            )
                            if st.session_state.new_spread_error:
                                st.warning(st.session_state.new_spread_error)
                        # Home ML input
                        if st.session_state.show_hml_input:
                            st.text_input(
                                "Enter new book home ml:",
                                key="new_hml_raw",
                                placeholder="e.g., -230, 300",
                                on_change=save_new_hml,   # <- stores float into session + reruns
                            )
                            if st.session_state.new_hml_error:
                                st.warning(st.session_state.new_hml_error)  
                        # Away ML input
                        if st.session_state.show_aml_input:
                            st.text_input(
                                "Enter new book away ml:",
                                key="new_aml_raw",
                                placeholder="e.g., -230, 300",
                                on_change=save_new_aml,   # <- stores float into session + reruns
                            )
                            if st.session_state.new_aml_error:
                                st.warning(st.session_state.new_aml_error)         
                        # over/under input
                        if st.session_state.show_over_input:
                            st.text_input(
                                "Enter new book over under:",
                                key="new_over_raw",
                                placeholder="e.g., 48.5",
                                on_change=save_new_over,   # <- stores float into session + reruns
                            )
                            if st.session_state.new_over_error:
                                st.warning(st.session_state.new_over_error)  

                        # reset lines button
                        if st.session_state.show_spread_input or st.session_state.show_hml_input or st.session_state.show_aml_input or st.session_state.show_over_input:
                            st.markdown(
                                """
                                <style>
                                div[data-testid="stButton"] {display: flex; justify-content: center;}
                                </style>
                                """,
                                unsafe_allow_html=True
                            )
                            st.button("Reset Lines", on_click=reset_lines)                    





                    # col for space
                    with col4:
                        st.write("")


                    ## Final Display Col (Home Team)

                    with col5:

                        st.write("") 

                        # logo
                        st.markdown("<div style='display: flex; justify-content: right;'><img src='" + filtered_df.iloc[0]["home_logo"] + "' width='150'></div>", unsafe_allow_html=True)
                                            
                        st.markdown(f"""
                                    <div style="font-size:35px; font-weight:bold; color:{filtered_df.iloc[0]["home_color"]}; text-align:right;">
                                        {filtered_df.iloc[0]["Home"]}
                                    </div>
                                """, unsafe_allow_html=True)       

                        st.write(" ")
                        st.write(" ")


                        # lines, predictions, and value
                        st.markdown(f"""
                                    <div style="font-size:20px; color: gray; text-align:right;">
                                        {filtered_df.iloc[0]["Pred Home Cover Prob"]}% - Pred Cover Prob
                                    </div>
                                """, unsafe_allow_html=True)  
                        
                                                    

                        if filtered_df.iloc[0]["Book Home Spread"] > 0:
                            st.markdown(f"""
                                        <div style="font-size:20px; color: gray; text-align:right;">
                                            +{filtered_df.iloc[0]["Book Home Spread"]} - Book Spread
                                        </div>
                                    """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                                        <div style="font-size:20px; color: gray; text-align:right;">
                                            {filtered_df.iloc[0]["Book Home Spread"]} - Book Spread
                                        </div>
                                    """, unsafe_allow_html=True)                             

                        if filtered_df.iloc[0]["Pred Home Cover Prob"] >= 50:
                            st.markdown(f"""
                                        <div style="font-size:20px; color: green; font-weight:bold; text-align:right;">
                                            {filtered_df.iloc[0]["Spread Value"]}% - To Cover Value
                                        </div>
                                    """, unsafe_allow_html=True)   
                        else:
                                st.markdown(f"""
                                        <div style="font-size:20px; color: red; font-weight:bold; text-align:right;">
                                            {filtered_df.iloc[0]["Spread Value"]*-1}% - To Cover Value
                                        </div>
                                    """, unsafe_allow_html=True)    
                        
                        st.write(" ")
                        st.write(" ")


                        if filtered_df.iloc[0]["Pred Home Moneyline"] > 0:
                            st.markdown(f"""
                                        <div style="font-size:20px; color: gray; text-align:right;">
                                            +{int(round(filtered_df.iloc[0]["Pred Home Moneyline"],0))} - Pred ML
                                        </div>
                                    """, unsafe_allow_html=True)   
                        else:
                            st.markdown(f"""
                                        <div style="font-size:20px; color: gray; text-align:right;">
                                            {int(round(filtered_df.iloc[0]["Pred Home Moneyline"],0))} - Pred ML
                                        </div>
                                    """, unsafe_allow_html=True)                           

                        st.markdown(f"""
                                    <div style="font-size:20px; color: gray; text-align:right;">
                                        {round(filtered_df.iloc[0]["Pred Home Win Prob"],2)}% - Pred Win Prob
                                    </div>
                                """, unsafe_allow_html=True)                         

                        if filtered_df.iloc[0]["Book Home Moneyline"]>0:
                                st.markdown(f"""
                                            <div style="font-size:20px; color: gray; text-align:right;">
                                                +{int(round(filtered_df.iloc[0]["Book Home Moneyline"],0))} - Book ML
                                            </div>
                                        """, unsafe_allow_html=True)  
                        else:  
                                st.markdown(f"""
                                            <div style="font-size:20px; color: gray; text-align:right;">
                                                {int(round(filtered_df.iloc[0]["Book Home Moneyline"],0))} - Book ML
                                            </div>
                                        """, unsafe_allow_html=True)                             

                        st.markdown(f"""
                                    <div style="font-size:20px; color: gray; text-align:right;">
                                        {round(filtered_df.iloc[0]["Book Home Win Prob"],2)}% - Book Imp Win Prob
                                    </div>
                                """, unsafe_allow_html=True)

                        if filtered_df.iloc[0]["Pred Home Win Prob"] - filtered_df.iloc[0]["Book Home Win Prob"] > filtered_df.iloc[0]["Pred Away Win Prob"] - filtered_df.iloc[0]["Book Away Win Prob"]:
                            if filtered_df.iloc[0]["ML Value"]>0:
                                st.markdown(f"""
                                            <div style="font-size:20px; color: green; font-weight:bold; text-align:right;">
                                                {filtered_df.iloc[0]["ML Value"]}% - To Hit Value
                                            </div>
                                        """, unsafe_allow_html=True)  
                            else:
                                st.markdown(f"""
                                            <div style="font-size:20px; color: red; font-weight:bold; text-align:right;">
                                                {filtered_df.iloc[0]["ML Value"]}% - To Hit Value
                                            </div>
                                        """, unsafe_allow_html=True)  
                        else:
                                st.markdown(f"""
                                        <div style="font-size:20px; color: red; font-weight:bold; text-align:right;">
                                            {filtered_df.iloc[0]["ML Losing Value"]}% - To Hit Value 
                                        </div>
                                    """, unsafe_allow_html=True) 

    st.write(" ")      
    st.write(" ")    

    st.divider()                  


    # more notes
    st.write("'Value' is a way to sort our predicted best value picks. Cover Value is a team's Predicted Cover Probability minus 50%" \
    " (50% is what we need to say that team will cover, so how much over this threshold are we?)"
    ". ML Value is the percentage difference between our predicted win probability for the team with value" \
    " and sportsbooks' implied wp (converted from ML). Over/Under Value is also the game's Predicted Over Probability minus 50%.")

    st.write("To look at more information for Spread, ML, or O/U, select the 'Bet Type' filter. To sort by one of those values, select" \
    " that filter.")

    st.write("Our model is built around our data source's betting lines, so we only compute probabilities for those. Double check to make sure your line is close to ours.")
    st.write("For more information on model accuracy, navigate to the 'Betting Accuracy' page")      









#### Bet Builder Page

def bet_builder_page():
    st.title("Betslip Builder")
    

    ## Helper Function
    
    # function to find results based on betslip data and unit size
    def exp_results_finder(data, unit_size):
        """
        Args:
        - data: betslip data
        - unit_size: current unit size of portfolio

        Output:
        - total_rows: bet count
        - total_exp_return: expected return % 
        - total_exp_profit: expected return $
        - total_variance: portfolio variance
        - total_stdev: portfolio standard deviation
        - total_ci_lower: lower CI bound at 95%
        - total_ci_upper: upper CI bound at 95%
        """

        # create df for calculations
        calcs_df = data.copy()
        # compute expected profit based on units and expected return
        calcs_df["Exp Profit"] = calcs_df["Units"] * calcs_df["Exp Return"] / 100 * unit_size

        # if spread, odds are assumed -110
        calcs_df.loc[calcs_df["Type"] == "Spread", "odds"] = -110
        # if ML, we have those specific odds
        calcs_df.loc[calcs_df["Type"] == "ML", "odds"] = (calcs_df["Line"].str.replace("+", "", regex=False).astype(float))
        # winnings with a win of the bet
        calcs_df.loc[calcs_df["odds"] < 0, "potential_winnings"] = 100 / abs(calcs_df.loc[calcs_df["odds"] < 0, "odds"])
        calcs_df.loc[calcs_df["odds"] > 0, "potential_winnings"] = abs(calcs_df.loc[calcs_df["odds"] > 0, "odds"]) / 100
        # losses with a loss of the bet
        calcs_df["potential_losses"] = 1
        # probability of bet hitting based on expected return prob
        calcs_df["backtest_hit_prob"] = (
            (calcs_df["Exp Return"] / 100 + calcs_df["potential_losses"])
            / (calcs_df["potential_winnings"] + calcs_df["potential_losses"])
        )
        # variance for each bet
        calcs_df["per_bet_variance"] = (
            (calcs_df["Units"] * unit_size) ** 2
            * (
                calcs_df["backtest_hit_prob"] * calcs_df["potential_winnings"] ** 2
                + (1 - calcs_df["backtest_hit_prob"]) * calcs_df["potential_losses"] ** 2
            )
            - calcs_df["Exp Profit"] ** 2
        )

        # find final portfolio expected stats
        total_rows = len(st.session_state.chosen)
        total_exp_return = round(calcs_df["Exp Profit"].sum() / portfolio_amount * 100, 2) if portfolio_amount > 0 else 0.0
        total_exp_profit = round(calcs_df["Exp Profit"].sum(), 2)
        total_variance = round(calcs_df["per_bet_variance"].sum(), 2)
        total_stdev = math.sqrt(total_variance)
        total_ci_lower = total_exp_profit - 1.96 * total_stdev
        total_ci_upper = total_exp_profit + 1.96 * total_stdev

        return total_rows, total_exp_return, total_exp_profit, total_variance, total_stdev, total_ci_lower, total_ci_upper



    ## Prep Data

    # baseline spread available bets df, convert to same structure as ML
    bb_spread_df = (
        merged_predictions[["Home", "Away", "Spread Bet on:", "Spread Value", "Spread Exp Return", "Book Home Spread"]]
        .rename(columns={"Spread Bet on:": "Bet on:", "Spread Value": "Value", "Spread Exp Return": "Exp Return"})
    )
    bb_spread_df["Type"] = "Spread"
    bb_spread_df.loc[bb_spread_df["Bet on:"] == bb_spread_df["Home"], "Line"] = bb_spread_df.loc[
        bb_spread_df["Bet on:"] == bb_spread_df["Home"], "Book Home Spread"
    ]
    bb_spread_df.loc[bb_spread_df["Bet on:"] == bb_spread_df["Away"], "Line"] = (
        bb_spread_df.loc[bb_spread_df["Bet on:"] == bb_spread_df["Away"], "Book Home Spread"] * -1
    )
    bb_spread_df["Line"] = bb_spread_df["Line"].apply(lambda x: f"{x:+.1f}")
    bb_spread_df = bb_spread_df[["Bet on:", "Line", "Value", "Home", "Away", "Type", "Exp Return"]]

    # baseline ml available bets df, convert to same structure as spread
    bb_ml_df = (
        merged_predictions[
            ["Home", "Away", "ML Bet on:", "ML Value", "ML Exp Return", "Book Home Moneyline", "Book Away Moneyline"]
        ]
        .rename(columns={"ML Bet on:": "Bet on:", "ML Value": "Value", "ML Exp Return": "Exp Return"})
    )
    bb_ml_df["Type"] = "ML"
    bb_ml_df.loc[bb_ml_df["Bet on:"] == bb_ml_df["Home"], "Line"] = bb_ml_df.loc[
        bb_ml_df["Bet on:"] == bb_ml_df["Home"], "Book Home Moneyline"
    ]
    bb_ml_df.loc[bb_ml_df["Bet on:"] == bb_ml_df["Away"], "Line"] = bb_ml_df.loc[
        bb_ml_df["Bet on:"] == bb_ml_df["Away"], "Book Away Moneyline"
    ]
    bb_ml_df["Line"] = bb_ml_df["Line"].apply(lambda x: f"{x:+.0f}")
    bb_ml_df = bb_ml_df[["Bet on:", "Line", "Value", "Home", "Away", "Type", "Exp Return"]]

    # combine spread and ml
    bb_all_df = pd.concat([bb_spread_df, bb_ml_df], axis=0, ignore_index=True).sort_values(by="Exp Return", ascending=False)

    # sort by value and create new indices
    base_df = bb_all_df.sort_values("Value", ascending=False)
    base_df = base_df.set_index(pd.RangeIndex(len(base_df))).reset_index(drop=False).rename(columns={"index": "row_id"})



    ## Initialize session states

    # initialize available game pool (full dataset)
    if "pool" not in st.session_state:
        df = base_df.copy()
        df["pick"] = False
        st.session_state.pool = df
    # initialize chosen pool (empty)
    if "chosen" not in st.session_state:
        st.session_state.chosen = pd.DataFrame(
            columns=["row_id", "Home", "Away", "Type", "Bet on:", "Line", "Value", "Exp Return", "Units", "remove"]
        )
    # initialize Units (default to 1)
    if "Units" not in st.session_state.chosen.columns:
        st.session_state.chosen["Units"] = pd.Series(dtype="Int64")
    st.session_state.chosen["Units"] = st.session_state.chosen["Units"].fillna(1).astype("Int64")



    ## Available Games

    st.markdown("### Available Games")
    st.write("Check 'Pick' to add to betslip")

    # bet type display df option
    type_options = sorted(base_df["Type"].dropna().unique().tolist(), reverse=True)
    default_type = "Spread" if "Spread" in type_options else type_options[0]
    if "type_radio" not in st.session_state:
        st.session_state.type_radio = default_type
    # create radio for current type
    current_type = st.radio("", options=type_options, key="type_radio", horizontal=True)
    # df col configuration
    pool_col_config = {
        "row_id": st.column_config.Column("ID", width="small", disabled=True),
        "Home": st.column_config.TextColumn("Home", help="Home team"),
        "Away": st.column_config.TextColumn("Away", help="Away team"),
        "Type": st.column_config.TextColumn("Type", help="Spread or ML"),
        "Bet on:": st.column_config.TextColumn("Bet on", help="Which side"),
        "Line": st.column_config.TextColumn("Line", help="Line"),
        "Value": st.column_config.NumberColumn("Value"),
        "Exp Return": st.column_config.NumberColumn("Exp Return", help="Expected return"),
        "pick": st.column_config.CheckboxColumn("Pick", help="Select rows to add"),
    }
    # mask for current bet type filter
    pool_mask = st.session_state.pool["Type"] == current_type
    # view of pool display df
    pool_view = (
        st.session_state.pool.loc[
            pool_mask, ["row_id", "pick", "Bet on:", "Line", "Value", "Exp Return", "Home", "Away", "Type"]
        ]
        .copy()
        .sort_values(by="Exp Return", ascending=False)
    )
    # editor for pool df
    edited_pool = st.data_editor(
        pool_view,
        key="pool_editor",
        hide_index=True,
        use_container_width=True,
        column_config=pool_col_config,
        disabled=["row_id", "Home", "Away", "Type", "Bet on:", "Line", "Value", "Exp Return"], 
    )
    # if add picks button clicked
    if st.button("➜ Add picks to betslip"):
        # create mask for what data we're moving
        move_mask = edited_pool["pick"] == True
        to_move = edited_pool.loc[
            move_mask, ["row_id", "Home", "Away", "Type", "Bet on:", "Line", "Value", "Exp Return"]
        ].copy()
        # set units at 1 and initialize remove col, put in chosen df, take out of pool df
        if not to_move.empty:
            to_move["Units"] = 1
            to_move["remove"] = False
            st.session_state.chosen = (
                pd.concat([st.session_state.chosen, to_move], ignore_index=True)
                .drop_duplicates(subset=["row_id"], keep="first")
            )
            remaining_full_pool = st.session_state.pool[
                ~st.session_state.pool["row_id"].isin(to_move["row_id"])
            ].copy()
            remaining_full_pool["pick"] = False
            st.session_state.pool = remaining_full_pool.sort_values(by="Exp Return", ascending=False)

            st.rerun()




    ## Customize Betslip Settings

    st.divider()
    st.markdown("### Customize Betslip")

    # options for bow to break it down
    breakdown_options = ["Total Betslip Amount", "Unit Size"]
    breadown_default = (
        breakdown_options.index(st.session_state.breakdown_choice)
        if "breakdown_choice" in st.session_state
        else 0
    )
    # choose amount
    st.session_state.breakdown_choice = st.selectbox(
        "Choose to input a Total Betslip Amount that will be split among units or a Per-Unit Bet size:",
        options=breakdown_options,
        index=breadown_default,
    )

    # portfolio and unit size defaults
    portfolio_amount = 0.0
    unit_size = 0.0

    # sum units in the state currently
    total_units_in_state = int(st.session_state.chosen["Units"].sum()) if not st.session_state.chosen.empty else 0

    # compute portfolio amount and betslip amount, dependent on what the breakdown choice was
    if st.session_state.breakdown_choice == "Total Betslip Amount":
        if "portfolio_amount" not in st.session_state:
            st.session_state.portfolio_amount = st.text_input("Total size of betslip ($):", value=20)
        else:
            st.session_state.portfolio_amount = st.text_input(
                "Total size of betslip ($):", value=st.session_state.portfolio_amount
            )
        try:
            portfolio_amount = float(st.session_state.portfolio_amount)
        except:
            st.session_state.portfolio_amount = 20
            portfolio_amount = float(st.session_state.portfolio_amount)
            st.write("Invalid amount, reset to $20")

        if total_units_in_state > 0:
            unit_size = portfolio_amount / total_units_in_state
    if st.session_state.breakdown_choice == "Unit Size":
        if "unit_size" not in st.session_state:
            st.session_state.unit_size = st.text_input("Size of each bet unit ($):", value=2)
        else:
            st.session_state.unit_size = st.text_input("Size of each bet unit ($):", value=st.session_state.unit_size)
        try:
            unit_size = float(st.session_state.unit_size)
        except:
            st.session_state.unit_size = 2
            unit_size = float(st.session_state.unit_size)
            st.write("Invalid amount, reset to $2")

        portfolio_amount = unit_size * total_units_in_state if total_units_in_state > 0 else 0.0




    ## Betslip Optimizer

    st.divider()

    st.markdown("### Betslip Optimizer")
    st.write("Can't decide? Choose from the options below and we'll make one for you!")

    # button to show optimizer
    if "show_optimizer" not in st.session_state:
        st.session_state.show_optimizer = False
    if st.button("Use Optimizer"):
        st.session_state.show_optimizer = not st.session_state.show_optimizer

    # if it's shown...
    if st.session_state.show_optimizer:

        # ask what the objective is for optimizer
        objective_options = ["Min Return", "Max Standard Deviation"]
        objective_default = (
            objective_options.index(st.session_state.objective_choice)
            if "objective_choice" in st.session_state
            else 0
        )
        # set a value for the objective
        st.session_state.objective_choice = st.selectbox(
            "Either set a maximum standard deviation the betslip can't go above or a minimum return it can't go below:",
            options=objective_options,
            index=objective_default,
        )
        # try to convert to float, set at default if error
        if st.session_state.objective_choice == "Max Standard Deviation":
            max_stdev = st.text_input("Maximum Standard Deviation:", value=10)
            try:
                max_stdev = float(max_stdev)
            except:
                max_stdev = 10.0
        else:
            min_return = st.text_input("Minimum Return %:", value=3)
            try:
                min_return = float(min_return)
            except:
                min_return = 3.0
        # ask which types of bets to include
        types_to_include = st.selectbox("Types to include", options=["All", "Spread only", "ML only"], index=1)
        # ask about randomness (likelihood next optimal bet is not selected)
        random_param = float(st.slider("Randomness", min_value=0.0, max_value=0.5, value=0.2, step=0.05))


        # run the optimizer button
        if st.button("Run Optimizer"):
            # reset pool & chosen
            st.session_state.pool = base_df.assign(pick=False).copy()
            st.session_state.chosen = st.session_state.chosen.iloc[0:0].copy()

            # must have portfolio amount selected for breakdown or else it doesn't work
            if st.session_state.breakdown_choice == "Unit Size":
                st.markdown(
                    """
                    <div style="font-size:15px; color:orange; text-align:left;">
                        Please define a Total Betslip Amount in 'Customize Betslip' before running the optimizer
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.session_state.slip_message = False

            # once portfolio amount is selected...
            else:
                
                # portfilio optimizer function
                def portfolio_optimizer():
                    # initialize pool and chosen dfs and set units to 1
                    opt_df = None
                    all_df = base_df.copy()
                    all_df["Units"] = 1

                    # find spread and ML ids
                    spr_ids = all_df.loc[all_df["Type"] == "Spread", "row_id"].tolist()
                    ml_ids = all_df.loc[all_df["Type"] == "ML", "row_id"].tolist()

                    # if both bet types, include all indices
                    if types_to_include == "All":
                        combined_ids = [v for pair in itertools.zip_longest(spr_ids, ml_ids) for v in pair if v is not None]
                    # just include spread for spread only
                    elif types_to_include == "Spread only":
                        combined_ids = spr_ids
                    # just include ML otherwise
                    else:
                        combined_ids = ml_ids

                    # initialize stats to keep track
                    last_ok_df = None
                    total_stdev = None
                    total_exp_return_local = None

                    # for each id 
                    for r in combined_ids:
                        # if randomness doesnt skip iteration
                        if random.random() >= random_param:
                            # create mask and find row
                            mask = all_df["row_id"].eq(r)
                            row = all_df.loc[mask].copy()
                            # add to df
                            opt_df = row if opt_df is None else pd.concat([opt_df, row], axis=0)

                            # find unit size
                            if st.session_state.breakdown_choice == "Total Betslip Amount":
                                temp_unit_size = float(st.session_state.portfolio_amount) / opt_df["Units"].sum()
                            else:
                                temp_unit_size = float(unit_size)

                            # find results for given inputs based on current opt_df (chosen df)
                            (
                                _nrows,
                                total_exp_return_local,
                                _ep,
                                _var,
                                total_stdev,
                                _lo,
                                _hi,
                            ) = exp_results_finder(opt_df.copy(), temp_unit_size)

                            # if we've reached the stdev limit, return opt_df
                            if st.session_state.objective_choice == "Max Standard Deviation":
                                if total_stdev < max_stdev:
                                    return opt_df, total_stdev
                            # if we've passed the minimum return limit, return last iteration's df before it went under
                            else:
                                if total_exp_return_local < min_return:
                                    return last_ok_df if last_ok_df is not None else opt_df, total_exp_return_local
                                else:
                                    last_ok_df = opt_df.copy()

                    # if we didnt achieve our goals, return what df we have
                    if st.session_state.objective_choice == "Max Standard Deviation":
                        return opt_df, (total_stdev if total_stdev is not None else float("inf"))
                    else:
                        return opt_df, (total_exp_return_local if total_exp_return_local is not None else 0.0)

                # run portfolio optimizer
                if st.session_state.objective_choice == "Max Standard Deviation":
                    opt_df, _total_stdev = portfolio_optimizer()
                else:
                    opt_df, _total_exp_return = portfolio_optimizer()

                # find the opt_df indices and move to chosen on the actual page, remove from pool
                if opt_df is not None and not opt_df.empty:
                    opt_ids = opt_df["row_id"]

                    to_move = st.session_state.pool.loc[
                        st.session_state.pool["row_id"].isin(opt_ids),
                        ["row_id", "Home", "Away", "Type", "Bet on:", "Line", "Value", "Exp Return"],
                    ].copy()

                    if not to_move.empty:
                        to_move["Units"] = 1
                        to_move["remove"] = False

                        st.session_state.chosen = (
                            pd.concat([st.session_state.chosen, to_move], ignore_index=True)
                            .drop_duplicates(subset=["row_id"], keep="first")
                        )

                        st.session_state.pool = (
                            st.session_state.pool.loc[~st.session_state.pool["row_id"].isin(opt_ids)]
                            .assign(pick=False)
                            .sort_values(by="Exp Return", ascending=False)
                            .copy()
                        )

                        st.session_state.slip_message = True
                        st.rerun()

    # control messages, depending on if show optimizer is present or not
    if not st.session_state.get("show_optimizer", False):
        st.session_state.slip_message = False
    if st.session_state.get("slip_message", False):
        st.markdown(
            """
            <div style="font-size:15px; color:green; text-align:left;">
                Your betslip has been created!
            </div>
            """,
            unsafe_allow_html=True,
        )

    
    st.divider() 




    ## Chosen Betslip


    st.markdown("### Betslip")
    st.write("Double-click on a bet's 'Units' to change. Select 'Remove' to remove bet")
    st.write("Check the expected results of your bet at the bottom of the page")

    # manually adjustable units col config
    UnitsCol = getattr(st.column_config, "SelectboxColumn", None)
    if UnitsCol is None:
        units_config = st.column_config.NumberColumn("Units", min_value=1, max_value=10, step=1, help="How many units to bet")
    else:
        units_config = st.column_config.SelectboxColumn("Units", options=[1, 2, 3, 4, 5], help="How many units to bet")

    # overall col config
    chosen_col_config = {
        "row_id": st.column_config.Column("ID", width="small", disabled=True),
        "Home": st.column_config.TextColumn("Home"),
        "Away": st.column_config.TextColumn("Away"),
        "Type": st.column_config.TextColumn("Type"),
        "Bet on:": st.column_config.TextColumn("Bet on"),
        "Line": st.column_config.TextColumn("Line", help="Line"),
        "Value": st.column_config.NumberColumn("Value"),
        "Exp Return": st.column_config.NumberColumn("Exp Return"),
        "Units": units_config,
        "remove": st.column_config.CheckboxColumn("Remove", help="Move back to Available"),
    }
    # view of chosen df
    chosen_view = st.session_state.chosen[
        ["row_id", "remove", "Units", "Bet on:", "Line", "Value", "Exp Return", "Home", "Away", "Type"]
    ].copy()
    # editor for chosen df
    edited_chosen = st.data_editor(
        chosen_view,
        key="chosen_editor",
        hide_index=True,
        use_container_width=True,
        column_order=["row_id", "remove", "Units", "Bet on:", "Line", "Value", "Exp Return", "Home", "Away", "Type"],
        column_config=chosen_col_config,
        disabled=["row_id", "Home", "Away", "Type", "Bet on:", "Line", "Value", "Exp Return"], 
    )

    # if units editer, make sure integer
    if "Units" in edited_chosen.columns:
        try:
            edited_chosen["Units"] = edited_chosen["Units"].astype("Int64")
        except Exception:
            pass

    # rerun if something changed
    cols = ["row_id","Home","Away","Type","Bet on:","Line","Value","Exp Return","Units","remove"]
    if not st.session_state.chosen[cols].equals(edited_chosen[cols]):
        st.session_state.chosen = edited_chosen[cols].copy()
        st.rerun() 


    # export data to csv if clicked
    @st.cache_data
    def df_to_csv_bytes(df):
        return df.to_csv(index=False).encode("utf-8")

    export_cols = ["Home", "Away", "Type", "Bet on:", "Line", "Value", "Exp Return", "Units"]
    betslip = st.session_state.chosen[export_cols] if "chosen" in st.session_state else None

    if betslip is not None and not betslip.empty:
        st.download_button(
            label="⬇️ Export betslip as CSV",
            data=df_to_csv_bytes(betslip),
            file_name="betslip.csv",
            mime="text/csv",
        )

    
    # remove from betslip button
    if st.button("⬅ Remove from betslip"):
        # create mask and find col in chosen df
        back_mask = edited_chosen["remove"] == True
        to_back = edited_chosen.loc[
            back_mask, ["row_id", "Home", "Away", "Type", "Bet on:", "Line", "Value", "Exp Return"]
        ].copy()
        if not to_back.empty:
            to_back["pick"] = False

            # add back to pool df
            st.session_state.pool = (
                pd.concat([st.session_state.pool, to_back], ignore_index=True)
                .drop_duplicates(subset=["row_id"], keep="first")
                .sort_values(by="Exp Return", ascending=False)
            )
            # take out of chosen df
            st.session_state.chosen = edited_chosen.loc[
                ~back_mask, ["row_id", "Home", "Away", "Type", "Bet on:", "Line", "Value", "Exp Return", "Units", "remove"]
            ].copy()
            # rerun page
            st.rerun()


    # button to reset all bets in page and clear betslip
    if st.button("🔄 Reset to original", help="Restore Available from current data and clear Selected"):
        st.session_state.pool = base_df.assign(pick=False).copy()
        st.session_state.chosen = st.session_state.chosen.iloc[0:0].copy()
        st.session_state.show_optimizer = False
        st.rerun()




    ## Betslip Results

    # if we have bets in the betslip
    if not st.session_state.chosen.empty:
        st.divider()
        st.markdown("### Expected Betslip Results")

        # find results based on betslip
        total_rows, total_exp_return, total_exp_profit, total_variance, total_stdev, total_ci_lower, total_ci_upper = \
            exp_results_finder(st.session_state.chosen.copy(), unit_size if unit_size else 0.0)

        # print results
        st.write(f"Total Wagered: ${portfolio_amount:.2f}")
        st.write(f"Unit Size: ${round(unit_size, 2)}")
        st.write(f"Bet Count: {total_rows}")
        st.write(f"Expected Profit: ${total_exp_profit}")
        st.write(f"Expected Return: {total_exp_return}%")
        st.write(f"Standard Deviation: {round(total_stdev, 2)}")
        st.write(f"Profit 95% Confidence Interval: $ {round(total_ci_lower, 2)}, {round(total_ci_upper, 2)}")


    # additional notes
    st.divider()
    st.write("Expected Returns are conservative estimates of what the return for a given game is based on how bets of similar 'Value' have performed in the past.")
    st.write("There is not an exact linear relationship between Value and return, but games with higher value typically have a higher return.")

 
   


            

    
    
# sidebar navigation
st.sidebar.title("""Czar CFB""")
page = st.sidebar.radio("Go to", ("This Week's Picks", "Betslip Builder", "The Czar Poll", "The Playoff", "Game Predictor", "Betting Accuracy", "About"))




# display the selected page
if page == "About":
    about_page()
elif page == "Betting Accuracy":
    historical_results_page()
elif page == "Game Predictor":
    game_predictor_page()
elif page == "The Playoff":
    playoff_page()
elif page == "The Czar Poll":
    power_rankings_page() 
elif page == "This Week's Picks":
    this_week_page()    
elif page == "Betslip Builder":
    bet_builder_page()
