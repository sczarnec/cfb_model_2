import streamlit as st
import pandas as pd
import numpy as np
import csv
import math
import xgboost as xgb



#### READ IN CSVs


## GONNA HAVE TO CHANGE A LOT OF THESE ONCE AUTOMATED

# Read in historical betting data
historical_data = pd.read_csv("final_prediction_df_21to24.csv", encoding="utf-8", sep=",", header=0)


# load theoretical data
theor_prepped = pd.read_csv("theor_prepped_test.csv", encoding="utf-8", sep=",", header=0)
theor_prepped = theor_prepped.drop(theor_prepped.columns[0], axis=1)
theor_agg = pd.read_csv("theoretical_agg_test.csv", encoding="utf-8", sep=",", header=0)
theor_agg = theor_agg.drop(theor_agg.columns[0], axis=1)

# load  model
cover_model = xgb.Booster()
cover_model.load_model("saved_models/20247_actual_t1_cover_model.model")
# load point diff vars
cover_vars = pd.read_csv("saved_models/20247_actual_t1_covermodel_var_list.csv", encoding="utf-8", header=0).iloc[:, 1]
cover_vars = cover_vars.astype(str).str.strip().tolist()


# load win prob model
wp_model = xgb.Booster()
wp_model.load_model("saved_models/20247_t1_win_model.model")
# load point diff vars
wp_vars = pd.read_csv("saved_models/20247_t1_winmodel_var_list.csv", encoding="utf-8", header=0).iloc[:, 1]
wp_vars = wp_vars.astype(str).str.strip().tolist()

# load over/under model
over_model = xgb.Booster()
over_model.load_model("saved_models/20247_actual_over_model.model")
# load point diff vars
over_vars = pd.read_csv("saved_models/20247_actual_overmodel_var_list.csv", encoding="utf-8", header=0).iloc[:, 1]
over_vars = over_vars.astype(str).str.strip().tolist()

# load this week's data for XGB
sample_data = pd.read_csv("model_prepped_this_week_test.csv", encoding="utf-8", sep=",", header=0)

# load team info data
team_info = pd.read_csv("team_info.csv", encoding="utf-8", sep=",", header=0)

# extract this week value
this_week = theor_prepped['week'].max()




### This Week Predictions

# predict function
def predict_with_model(model, var_list, data):
    best_ntree = int(model.attr("best_ntreelimit"))
    dmatrix = xgb.DMatrix(data[var_list])
    return model.predict(dmatrix,iteration_range=(0, best_ntree))

def make_merged_predictions(data):

        # predict the 3 outcomes for this week
        cp_preds = predict_with_model(cover_model, cover_vars, data)
        wp_preds = predict_with_model(wp_model, wp_vars, data)
        op_preds = predict_with_model(over_model, over_vars, data)

        # results dfs
        results_df = data[["t1_team", "t2_team", "t1_home", "neutral_site", "game_id"]].copy()
        results_df["cover_prob_pred"] = cp_preds
        results_df["win_prob_pred"] = wp_preds
        results_df["over_prob_pred"] = op_preds

        # function to make sure games are on same side
        def normalize_row(row):
            keep_original = ((row["t1_home"] == 1) or ((row["neutral_site"] == 1) and (row["t1_team"] > row["t2_team"])))
            if keep_original:
                return row
            else:
                row["t1_team"], row["t2_team"] = row["t2_team"], row["t1_team"]
                row["cover_prob_pred"] = 1-row["cover_prob_pred"]
                row["win_prob_pred"] = 1 - row["win_prob_pred"]
                return row
        normalized_df = results_df.apply(normalize_row, axis=1)

        # aggregate the two predictions of each game
        aggregated_df = normalized_df.groupby(
            ["game_id", "t1_team", "t2_team", "neutral_site"], as_index=False
        ).agg({
            "cover_prob_pred": "mean",
            "win_prob_pred": "mean",
            "over_prob_pred": "mean"
        })


        # modify and create some cols, including predicted moneylines 
        aggregated_df["t2_win_prob_pred"] = (1- aggregated_df["win_prob_pred"])

        def winprob_to_moneyline(prob):
            if prob >= 0.5:
                return round(-100 * prob / (1 - prob), 0)
            else:
                return round(100 * (1 - prob) / prob, 0)
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


        ## Spread


        merged_predictions.loc[merged_predictions['Pred Home Cover Prob']>.5,"Spread Bet on:"] = merged_predictions['Home']
        merged_predictions.loc[merged_predictions['Pred Home Cover Prob']<=.5,"Spread Bet on:"] = merged_predictions['Away']
        merged_predictions["Spread Value"] = round(abs(merged_predictions['Pred Home Cover Prob'] - .5)*100,2)




        ## Moneyline

        # convert book ml to win prob, create some related cols
        def moneyline_to_winprob(ml):
            if ml < 0:
                # Negative moneyline (favorite)
                return round(-ml / (-ml + 100), 4)
            else:
                # Positive moneyline (underdog)
                return round(100 / (ml + 100), 4)
            
        merged_predictions["t1_ml_prob"] = merged_predictions["t1_moneyline"].apply(moneyline_to_winprob)
        merged_predictions["t2_ml_prob"] = merged_predictions["t2_moneyline"].apply(moneyline_to_winprob)


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


        # create ml value and bet on cols
        merged_predictions["ML Bet on:"] = np.select(conditions_ml, choices_ml_bet, default="Neither")
        merged_predictions["ML Value"] = np.select(conditions_ml_val, choices_ml_val, default=0)
        merged_predictions["ML Losing Value"] = np.select(conditions_ml_val, choices_ml_losing_val, default=0)
        merged_predictions["ML Value"] = round(merged_predictions["ML Value"]*100,2)
        merged_predictions["ML Losing Value"] = round(merged_predictions["ML Losing Value"]*100,2)



        ## Over/Under

        # create o/u value and bet on cols
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

        return merged_predictions        



merged_predictions = make_merged_predictions(sample_data)




### THEORETICAL MODIFICATIONS

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



### WELCOME PAGE
def welcome_page():
  
  st.title('Czar College Football')
  
  st.markdown(
          """
          <style>
          .custom-font {
              font-size: 20px;
          }
          </style>
          <div class="custom-font">Welcome! This site is built around the use of an XGBoost model to predict
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
      
  st.markdown(
          """
          <style>
          .custom-font {
              font-size: 24px;
          }
          </style>
          <div class="custom-font">Check out the Navigation menu on the top left to look at how our
          model predicts games (real and theoretical) or how it has performed historically.</div>
          """,
          unsafe_allow_html=True
      )
  
   
  




### HISTORICAL RESULTS PAGE
def historical_results_page():
    
    st.title('Model Performance on Historical Test Data')

    st.write("This is how well our model would perform on previous games over the last few years versus how well a random 50/50 guesser would perform. Only test data (games not included in model training) are included here.")
    st.write("Last thing: these returns are assuming you bet the same units for hundreds of games. If you use the model on one game, the return is either gonna be 0 or ~95%. If you bet on 100, it will look more like this projected reutrn. It's a sample size game: it won't get every game right but will hopefully be profitable over time.")
    st.write("More notes/considerations are at the bottom of this page")

    df = historical_data

    # convert point diff into spread
    #df['book_t1_spread'] = df['book_t1_point_diff']*-1

    # Clean column names to remove leading/trailing spaces
    df.columns = df.columns.str.strip()

    # Format "season" as a string
    df['season'] = df['season'].astype(str)

    df['cover_prob_from_50'] = abs(df['pred_t1_cp'] - .5)


    # Get unique teams and sort them in ascending order (for both t1 and t2)
    unique_teams = pd.unique(df[['t1_team', 't2_team']].values.ravel('K'))
    unique_teams_sorted = sorted(unique_teams)

    # Get unique weeks (ascending order)
    unique_weeks = sorted(df['week'].unique())

    # Get unique seasons (descending order)
    unique_seasons = sorted(df['season'].unique(), reverse=True)

    # Get unique teams and sort them in ascending order (for both t1 and t2)
    unique_conf = pd.unique(df[['t1_conference', 't2_conference']].values.ravel('K'))
    unique_conf_sorted = sorted(unique_conf)    

    # Create columns for layout
    col1, col2, col3, col4 = st.columns([3, 1, 8, 5])  

    with col1:
      
        # create filters
        st.write("### Filters")
        
        team_options = st.selectbox(
            "Team", 
            options=["All"] + unique_teams_sorted,  
            index=0  
        )

        week_options = st.multiselect(
            "Week", 
            options=["All"] + unique_weeks,  
            default=["All"]  
        )

        season_options = st.multiselect(
            "Season", 
            options=["All"] + unique_seasons,  
            default=["All"]  
        )

        conf_options = st.selectbox(
            "Conference", 
            options=["All"] + unique_conf_sorted,  
            index = 0 
        )        
        




        # Filter for book_t1_spread (manual number input with bounds)
        # use int and floor/ceiling so the bounds don't round down and exclude outer bound games
        cover_conf_lower, cover_conf_upper = st.slider(
            "Cover Confidence",
            min_value=float(int(math.floor(df['cover_confidence'].min()))),
            max_value=float(round(df['cover_confidence'].max(),2))+.01,
            value=(float((int(math.floor(df['cover_confidence'].min())))), float(round(df['cover_confidence'].max(),2))+.01),
            step = .01
        )

        # Filter for pred vs book (manual number input with bounds)
        # use int and floor/ceiling so the bounds don't round down and exclude outer bound games
        ml_value_lower, ml_value_upper = st.slider(
            "Our win prob compared to sportsbooks'",
            min_value=float(round(df['ml_value'].min(),2))-.01,
            max_value=float(round(df['ml_value'].max(),2))+.01,
            value=(float(round(df['ml_value'].min(),2))-.01 , float(round(df['ml_value'].max(),2))+.01),
            step=.01
        )

        ou_conf_lower, ou_conf_upper = st.slider(
            "O/U Confidence",
            min_value=float(int(math.floor(df['ou_confidence'].min()))),
            max_value=float(round(df['ou_confidence'].max(),2))+.01,
            value=(float((int(math.floor(df['ou_confidence'].min())))), float(round(df['ou_confidence'].max(),2))+.01),
            step = .01
        )        


        # Filter for book_over_under (manual number input with bounds)
        book_sp_lower, book_sp_upper = st.slider(
            "Home Book Spread",
            min_value=int(math.floor(df['t1_book_spread'].min())),
            max_value=int(math.ceil(df['t1_book_spread'].max())),
            value=(int(math.floor(df['t1_book_spread'].min())), int(math.ceil(df['t1_book_spread'].max())))
        ) 

        # Filter for book_home_ml_odds (manual number input with bounds)
        book_home_ml_prob_lower, book_home_ml_prob_upper = st.slider(
            "Book Home ML Prob",
            min_value=float(int(math.floor(df['t1_ml_prob'].min()))),
            max_value=float(int(math.ceil(df['t1_ml_prob'].max()))),
            value=(float(int(math.floor(df['t1_ml_prob'].min()))), float(int(math.ceil(df['t1_ml_prob'].max())))),
            step = .01
        )
  

        # Filter for book_over_under (manual number input with bounds)
        book_tp_lower, book_tp_upper = st.slider(
            "Book Over/Under",
            min_value=int(math.floor(df['book_over_under'].min())),
            max_value=int(math.ceil(df['book_over_under'].max())),
            value=(int(math.floor(df['book_over_under'].min())), int(math.ceil(df['book_over_under'].max())))
        )        

        
        
        # Filter to ask user whether they want to exclude NAs
        exclude_na_ml = st.checkbox("Exclude NAs in the ml columns?", value=False)
        
        
        
     
    
    # column for space
    with col2:
        st.write("")
        

    # last column, filtered data frame
    with col4:
        # Apply filters based on selections
        filtered_df = df

        if team_options != "All":
            filtered_df = filtered_df[filtered_df['t1_team'].eq(team_options) | filtered_df['t2_team'].eq(team_options)]
        else:
            filtered_df = df


        if "All" not in week_options:
            filtered_df = filtered_df[filtered_df['week'].isin(week_options)]

        if "All" not in season_options:
            filtered_df = filtered_df[filtered_df['season'].isin(season_options)]

        if conf_options != "All":
            filtered_df = filtered_df[filtered_df['t1_conference'].eq(conf_options) | filtered_df['t2_conference'].eq(conf_options)]             
        


        filtered_df = filtered_df[
            (filtered_df['cover_confidence'] >= cover_conf_lower) &
            (filtered_df['cover_confidence'] <= cover_conf_upper)|
            (filtered_df['cover_confidence'].isna())
            ]
        
        filtered_df = filtered_df[
            (filtered_df['ml_value'] >= ml_value_lower) & 
            (filtered_df['ml_value'] <= ml_value_upper) |
            (filtered_df['ml_value'].isna())
        ]        
        
        filtered_df = filtered_df[
            (filtered_df['ou_confidence'] >= ou_conf_lower) &
            (filtered_df['ou_confidence'] <= ou_conf_upper)|
            (filtered_df['ou_confidence'].isna())
            ]        



        


        filtered_df = filtered_df[
            (filtered_df['t1_book_spread'] >= book_sp_lower) & 
            (filtered_df['t1_book_spread'] <= book_sp_upper) |
            (filtered_df['t1_book_spread'].isna())
        ]

        filtered_df = filtered_df[
            (filtered_df['t1_ml_prob'] >= book_home_ml_prob_lower) & 
            (filtered_df['t1_ml_prob'] <= book_home_ml_prob_upper) |
            (filtered_df['t1_ml_prob'].isna())
        ]        


        filtered_df = filtered_df[
            (filtered_df['book_over_under'] >= book_tp_lower) & 
            (filtered_df['book_over_under'] <= book_tp_upper) |
            (filtered_df['book_over_under'].isna())
        ]
        
    
            
        # Exclude rows with NAs if the user selects the option
        if exclude_na_ml:
            filtered_df = filtered_df.dropna(subset=['ml_winnings'])
            

        # Show the filtered data in Streamlit
        st.write("### Test Data")
        st.dataframe(filtered_df)
        
    
    
    # column with results text
    with col3:
            
            st.write("### Betting Results")

            
            ### SPREAD
            
            # Calculate our return on investment
            our_return_spread = filtered_df['spread_winnings'].sum(skipna=True) / filtered_df['spread_winnings'].count()

            # format it as a percentage and dollars ($100 bet)
            our_return_percentage_spread = f"{(our_return_spread * 100) - 100:.2f}%"  # If you want to show it as a percentage
            our_return_dollars_spread = f"{(100 * our_return_spread):.2f}"
            
            # Calculate bettor average return on investment
            naive_return_spread = filtered_df['naive_spread_winnings'].sum(skipna=True) / filtered_df['naive_spread_winnings'].count()

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
            
            
            
            if our_return_spread > naive_return_spread:
                # Create the sentence with the calculated value for blank1
                st.markdown(f"""
                    The average bettor would earn a **{naive_return_percentage_spread}** return on our investment, both sides being bet evenly.
                    If they evenly split \$100 between the games, we would finish with <span><b>\${naive_return_dollars_spread}</b></span>, which is <span style="color:green"><b>${ours_over_naive_spread}</b></span> less than we made.
                """, unsafe_allow_html=True)
            else:
                # Create the sentence with the calculated value for blank1
                st.markdown(f"""
                    The average bettor would earn a **{naive_return_percentage_spread}** return on their investment, both sides being bet evenly.
                    If they evenly split \$100 between the games, they would finish with <span><b>\${naive_return_dollars_spread}</b></span>, which is <span style="color:red"><b>${naive_over_ours_spread}</b></span> more than we made.
                """, unsafe_allow_html=True)
                
            filtered_row_total_spread = filtered_df['spread_winnings'].count()
            
            st.markdown(f"""
                <span><i>using a sample of {filtered_row_total_spread} games for spread calculations</span>
                """, unsafe_allow_html=True)
                
                
            st.write("")
            st.write("")
                
                
                


           ### MONEYLINE
            
            # our return on investment
            our_return_moneyline = filtered_df['ml_winnings'].sum(skipna=True) / filtered_df['ml_winnings'].count()

            # format it as a percentage and dollars ($100 bet)
            our_return_percentage_moneyline = f"{(our_return_moneyline * 100) - 100:.2f}%"  # If you want to show it as a percentage
            our_return_dollars_moneyline = f"{(100 * our_return_moneyline):.2f}"
            
            # average return on investment
            naive_return_moneyline = filtered_df['naive_ml_winnings'].sum(skipna=True) / filtered_df['naive_ml_winnings'].count()

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




            st.markdown(f"""
                    For moneyline, we are betting only if we detect value on a certain side. So, if Team 1 has a moneyline of +240
                        but our win probability model says it should really be +160, we bet on their side. 
                """, unsafe_allow_html=True)

            # format description
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
            
            
            
            if our_return_moneyline > naive_return_moneyline:
                # Create the sentence with the calculated value for blank1
                st.markdown(f"""
                    A conservative bettor would earn a **{naive_return_percentage_moneyline}** return on their investment if they bet both sides.
                    If they evenly split \$100 between the games, we would finish with <span><b>\${naive_return_dollars_moneyline}</b></span>, which is <span style="color:green"><b>${ours_over_naive_moneyline}</b></span> less than we made.
                """, unsafe_allow_html=True)
            else:
                # Create the sentence with the calculated value for blank1
                st.markdown(f"""
                    A conservative bettor would earn a **{naive_return_percentage_moneyline}** return on their investment if they bet both sides.
                    If they evenly split \$100 between the games, they would finish with <span><b>\${naive_return_dollars_moneyline}</b></span>, which is <span style="color:red"><b>${naive_over_ours_moneyline}</b></span> more than we made.
                """, unsafe_allow_html=True)
                
            
            filtered_row_total_moneyline = filtered_df['ml_winnings'].count()
            
            st.markdown(f"""
                <span><i>using a sample of {filtered_row_total_moneyline} games for moneyline calculations</span>
                """, unsafe_allow_html=True)

            st.write("")
            st.write("")                





           ### OVER/UNDER
            
            # our return on investment
            our_return_ou = filtered_df['ou_winnings'].sum(skipna=True) / filtered_df['ou_winnings'].count()

            # format it as a percentage and dollars ($100 bet)
            our_return_percentage_ou = f"{(our_return_ou * 100) - 100:.2f}%"  # If you want to show it as a percentage
            our_return_dollars_ou = f"{(100 * our_return_ou):.2f}"
            
            # average return on investment
            naive_return_ou = filtered_df['naive_ou_winnings'].sum(skipna=True) / filtered_df['naive_ou_winnings'].count()

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




            st.markdown(f"""
                    For over/under, we are betting the side that our model predicts to be correct. So, if the line is 37.5 and our model
                    predicts 40 total points, we bet the over. 
                """, unsafe_allow_html=True)

            # format description
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
            
            
            
            if our_return_ou > naive_return_ou:
                # Create the sentence with the calculated value for blank1
                st.markdown(f"""
                    A conservative bettor would earn a **{naive_return_percentage_ou}** return on their investment if they bet both sides.
                    If they evenly split \$100 between the games, we would finish with <span><b>\${naive_return_dollars_ou}</b></span>, which is <span style="color:green"><b>${ours_over_naive_ou}</b></span> less than we made.
                """, unsafe_allow_html=True)
            else:
                # Create the sentence with the calculated value for blank1
                st.markdown(f"""
                    A conservative bettor would earn a **{naive_return_percentage_ou}** return on their investment if they bet both sides.
                    If they evenly split \$100 between the games, they would finish with <span><b>\${naive_return_dollars_ou}</b></span>, which is <span style="color:red"><b>${naive_over_ours_ou}</b></span> more than we made.
                """, unsafe_allow_html=True)
                
            
            filtered_row_total_ou = filtered_df['ou_winnings'].count()
            
            st.markdown(f"""
                <span><i>using a sample of {filtered_row_total_ou} games for over/under calculations</span>
                """, unsafe_allow_html=True)


    st.write(" ")
    st.write(" ")
    st.write("'Confidence' represents (our predicted probability - 50%). For example, if a team has a 55% chance to cover, their cover confidence is 5%. Similarly, ML Value is a direct comparison of our predicted win " \
    "probability vs the book's implicit win probability from ML. For example, if the book's win probability is 55% and ours is 60%, that's a 5% value.")
    st.write("As a note, some moneyline values for games are incomplete in the data source. These appear as NAs. Others may be NA due to a lack of prior data in the season for one of the teams.")
    





                
                
### PLAYOFF PREDICTION PAGE               
def playoff_page():
  
  # Streamlit App
  st.title("The Playoff")
  
  st.write("Customize your playoff! Select what teams you want in each seed. The model will predict each game, but you can override with the checkbox.")

  st.write("  ")


  neutral_site_ind_fr = 0
  neutral_site_ind_rest = 1
  
  # Create columns for layout
  col1, col2, col3, col4, col5, col6, col7, col8, col9 = st.columns([4,2,4,1,4,1,4,1,4])
  
  
  
  with col1:
    
      # Create select boxes with specific default values
      default_values = ["Oregon", "Georgia", "Boise State", "Arizona State", "Texas", "Penn State", 
                      "Notre Dame", "Ohio State", "Tennessee", "Illinois", "SMU", "Clemson"]

      seeds = []
      for value in default_values:
          
        try:
            # Find the index of the default value
            unique_teams = sorted(theor_prepped["t1_team"].dropna().unique())
            index = unique_teams.index(value)
        except ValueError:
            # If the value is not found, use a fallback index (e.g., -1 or 0)
            index = -1  # You can change this to 0 if you want a valid index
        seeds.append(index)
    
      
      # select boxes
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
      
      
      
  with col3:
    
        
        st.markdown(f"""
            <div style="font-size:30px; font-weight:bold; text-align:center; line-height:1.75; color:orange;">
                1st Round <br>
            </div>
        """, unsafe_allow_html=True)
        
        
        # FR1
        if "override_fr1" not in st.session_state:
            st.session_state.override_fr1 = False
    
        results_fr1 = theor_prepped[
            (theor_prepped["t1_team"] == seed8) &
            (theor_prepped["t2_team"] == seed9) &
            (theor_prepped["t1_home"] == 1)]
    
        # Display the header for the game
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
        
        # Conditional Display of Matchup based on session_state
        if st.session_state.override_fr1 == True:
            
            st.markdown("<div style='display: flex; justify-content: center;'><img src='" + results_fr1.iloc[0]["losing_logo"] + "' width='60'></div>", unsafe_allow_html=True)
            
            st.markdown(f"""
                <div style="font-size:15px; font-weight:bold; color:{results_fr1.iloc[0]["losing_color"]}; text-align:center;">
                      
                </div>
            """, unsafe_allow_html=True)
            
            fr1_winner = results_fr1.iloc[0]["losing_team"]
            
        else:
            st.markdown("<div style='display: flex; justify-content: center;'><img src='" + results_fr1.iloc[0]["winning_logo"] + "' width='60'></div>", unsafe_allow_html=True)
            
            # If override is checked, use a different color or text for the result
            st.markdown(f"""
                <div style="font-size:15px; font-weight:bold; color:{results_fr1.iloc[0]["winning_color"]}; text-align:center;">
                    {results_fr1.iloc[0]["pred_winner_wp"]}% wp
                </div>
            """, unsafe_allow_html=True)
            
            fr1_winner = results_fr1.iloc[0]["winning_team"]
        
            
        # Place the checkbox at the bottom of the page (after everything else)
        st.checkbox("Override Win?", value=st.session_state.override_fr1, key = "override_fr1")
        
        
        st.write(" ")
        
       
       
        
        # FR2
        if "override_fr2" not in st.session_state:
            st.session_state.override_fr2 = False

        results_fr2 = theor_prepped[
            (theor_prepped["t1_team"] == seed7) &
            (theor_prepped["t2_team"] == seed10) &
            (theor_prepped["t1_home"] == 1)]    
    
        # Display the header for the game
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
        
        # Conditional Display of Matchup based on session_state
        if st.session_state.override_fr2 == True:
            
            st.markdown("<div style='display: flex; justify-content: center;'><img src='" + results_fr2.iloc[0]["losing_logo"] + "' width='60'></div>", unsafe_allow_html=True)
            
            st.markdown(f"""
                <div style="font-size:15px; font-weight:bold; color:{results_fr2.iloc[0]["winning_color"]}; text-align:center;">
                      
                </div>
            """, unsafe_allow_html=True)
            
            fr2_winner = results_fr2.iloc[0]["losing_team"]
            
        else:
            st.markdown("<div style='display: flex; justify-content: center;'><img src='" + results_fr2.iloc[0]["winning_logo"] + "' width='60'></div>", unsafe_allow_html=True)
            
            # If override is checked, use a different color or text for the result
            st.markdown(f"""
                <div style="font-size:15px; font-weight:bold; color:{results_fr2.iloc[0]["winning_color"]}; text-align:center;">
                    {results_fr2.iloc[0]["pred_winner_wp"]}% wp
                </div>
            """, unsafe_allow_html=True)
            
            fr2_winner = results_fr2.iloc[0]["winning_team"]
        
            
        # Place the checkbox at the bottom of the page (after everything else)
        st.checkbox("Override Win?", value=st.session_state.override_fr2, key = "override_fr2")
        
        
        st.write(" ")
        
        
        


        # FR3
        if "override_fr3" not in st.session_state:
            st.session_state.override_fr3 = False

        results_fr3 = theor_prepped[
            (theor_prepped["t1_team"] == seed6) &
            (theor_prepped["t2_team"] == seed11) &
            (theor_prepped["t1_home"] == 1)]   
    
        # Display the header for the game
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
        
        # Conditional Display of Matchup based on session_state
        if st.session_state.override_fr3 == True:
            
            st.markdown("<div style='display: flex; justify-content: center;'><img src='" + results_fr3.iloc[0]["losing_logo"] + "' width='60'></div>", unsafe_allow_html=True)
            
            st.markdown(f"""
                <div style="font-size:15px; font-weight:bold; color:{results_fr3.iloc[0]["winning_color"]}; text-align:center;">
                      
                </div>
            """, unsafe_allow_html=True)
            
            fr3_winner = results_fr3.iloc[0]["losing_team"]
            
        else:
            st.markdown("<div style='display: flex; justify-content: center;'><img src='" + results_fr3.iloc[0]["winning_logo"] + "' width='60'></div>", unsafe_allow_html=True)
            
            # If override is checked, use a different color or text for the result
            st.markdown(f"""
                <div style="font-size:15px; font-weight:bold; color:{results_fr3.iloc[0]["winning_color"]}; text-align:center;">
                    {results_fr3.iloc[0]["pred_winner_wp"]}% wp
                </div>
            """, unsafe_allow_html=True)
            
            fr3_winner = results_fr3.iloc[0]["winning_team"]
        
            
        # Place the checkbox at the bottom of the page (after everything else)
        st.checkbox("Override Win?", value=st.session_state.override_fr3, key = "override_fr3")
        
        
        st.write(" ")
        
        
        
        
        
        # FR4
        if "override_fr4" not in st.session_state:
            st.session_state.override_fr4 = False
    
        results_fr4 = theor_prepped[
            (theor_prepped["t1_team"] == seed5) &
            (theor_prepped["t2_team"] == seed12) &
            (theor_prepped["t1_home"] == 1)]    
    
        # Display the header for the game
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
        
        # Conditional Display of Matchup based on session_state
        if st.session_state.override_fr4 == True:
            
            st.markdown("<div style='display: flex; justify-content: center;'><img src='" + results_fr4.iloc[0]["losing_logo"] + "' width='60'></div>", unsafe_allow_html=True)
            
            st.markdown(f"""
                <div style="font-size:15px; font-weight:bold; color:{results_fr4.iloc[0]["winning_color"]}; text-align:center;">
                      
                </div>
            """, unsafe_allow_html=True)
            
            fr4_winner = results_fr4.iloc[0]["losing_team"]
            
        else:
            st.markdown("<div style='display: flex; justify-content: center;'><img src='" + results_fr4.iloc[0]["winning_logo"] + "' width='60'></div>", unsafe_allow_html=True)
            
            # If override is checked, use a different color or text for the result
            st.markdown(f"""
                <div style="font-size:15px; font-weight:bold; color:{results_fr4.iloc[0]["winning_color"]}; text-align:center;">
                    {results_fr4.iloc[0]["pred_winner_wp"]}% wp
                </div>
            """, unsafe_allow_html=True)
            
            fr4_winner = results_fr4.iloc[0]["winning_team"]
        
            
        # Place the checkbox at the bottom of the page (after everything else)
        st.checkbox("Override Win?", value=st.session_state.override_fr4, key = "override_fr4")
        
        
        
        
        
        
  with col5:
    
    
        st.markdown(f"""
            <div style="font-size:30px; font-weight:bold; text-align:center; line-height:1.75; color:orange;">
                Quarters <br>
            </div>
        """, unsafe_allow_html=True)


        # Rose Bowl
        if "override_rb" not in st.session_state:
            st.session_state.override_rb = False
    
        results_rb = theor_prepped[
            (theor_prepped["t1_team"] == seed1) &
            (theor_prepped["t2_team"] == fr1_winner) &
            (theor_prepped["neutral_site"] == 1)]            
    
        # Display the header for the game
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
        
        # Conditional Display of Matchup based on session_state
        if st.session_state.override_rb == True:
            
            st.markdown("<div style='display: flex; justify-content: center;'><img src='" + results_rb.iloc[0]["losing_logo"] + "' width='60'></div>", unsafe_allow_html=True)
            
            st.markdown(f"""
                <div style="font-size:15px; font-weight:bold; color:{results_rb.iloc[0]["winning_color"]}; text-align:center;">
                      
                </div>
            """, unsafe_allow_html=True)
            
            rb_winner = results_rb.iloc[0]["losing_team"]
            
        else:
            st.markdown("<div style='display: flex; justify-content: center;'><img src='" + results_rb.iloc[0]["winning_logo"] + "' width='60'></div>", unsafe_allow_html=True)
            
            # If override is checked, use a different color or text for the result
            st.markdown(f"""
                <div style="font-size:15px; font-weight:bold; color:{results_rb.iloc[0]["winning_color"]}; text-align:center;">
                    {results_rb.iloc[0]["pred_winner_wp"]}% wp
                </div>
            """, unsafe_allow_html=True)
            
            rb_winner = results_rb.iloc[0]["winning_team"]
        
            
        # Place the checkbox at the bottom of the page (after everything else)
        st.checkbox("Override Win?", value=st.session_state.override_rb, key = "override_rb")
        
        
        st.write(" ")
        
       
       
        
        # Sugar Bowl
        if "override_sb" not in st.session_state:
            st.session_state.override_sb = False

        results_sb = theor_prepped[
            (theor_prepped["t1_team"] == seed2) &
            (theor_prepped["t2_team"] == fr2_winner) &
            (theor_prepped["neutral_site"] == 1)]      
    
        # Display the header for the game
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
        
        # Conditional Display of Matchup based on session_state
        if st.session_state.override_sb == True:
            
            st.markdown("<div style='display: flex; justify-content: center;'><img src='" + results_sb.iloc[0]["losing_logo"] + "' width='60'></div>", unsafe_allow_html=True)
            
            st.markdown(f"""
                <div style="font-size:15px; font-weight:bold; color:{results_sb.iloc[0]["winning_color"]}; text-align:center;">
                      
                </div>
            """, unsafe_allow_html=True)
            
            sb_winner = results_sb.iloc[0]["losing_team"]
            
        else:
            st.markdown("<div style='display: flex; justify-content: center;'><img src='" + results_sb.iloc[0]["winning_logo"] + "' width='60'></div>", unsafe_allow_html=True)
            
            # If override is checked, use a different color or text for the result
            st.markdown(f"""
                <div style="font-size:15px; font-weight:bold; color:{results_sb.iloc[0]["winning_color"]}; text-align:center;">
                    {results_sb.iloc[0]["pred_winner_wp"]}% wp
                </div>
            """, unsafe_allow_html=True)
            
            sb_winner = results_sb.iloc[0]["winning_team"]
        
            
        # Place the checkbox at the bottom of the page (after everything else)
        st.checkbox("Override Win?", value=st.session_state.override_sb, key = "override_sb")
        
        
        st.write(" ")
        
        
        


        # Fiesta Bowl
        if "override_fb" not in st.session_state:
            st.session_state.override_fb = False

        results_fb = theor_prepped[
            (theor_prepped["t1_team"] == seed3) &
            (theor_prepped["t2_team"] == fr3_winner) &
            (theor_prepped["neutral_site"] == 1)]     
    
        # Display the header for the game
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
        
        # Conditional Display of Matchup based on session_state
        if st.session_state.override_fb == True:
            
            st.markdown("<div style='display: flex; justify-content: center;'><img src='" + results_fb.iloc[0]["losing_logo"] + "' width='60'></div>", unsafe_allow_html=True)
            
            st.markdown(f"""
                <div style="font-size:15px; font-weight:bold; color:{results_fb.iloc[0]["winning_color"]}; text-align:center;">
                      
                </div>
            """, unsafe_allow_html=True)
            
            fb_winner = results_fb.iloc[0]["losing_team"]
            
        else:
            st.markdown("<div style='display: flex; justify-content: center;'><img src='" + results_fb.iloc[0]["winning_logo"] + "' width='60'></div>", unsafe_allow_html=True)
            
            # If override is checked, use a different color or text for the result
            st.markdown(f"""
                <div style="font-size:15px; font-weight:bold; color:{results_fb.iloc[0]["winning_color"]}; text-align:center;">
                    {results_fb.iloc[0]["pred_winner_wp"]}% wp
                </div>
            """, unsafe_allow_html=True)
            
            fb_winner = results_fb.iloc[0]["winning_team"]
        
            
        # Place the checkbox at the bottom of the page (after everything else)
        st.checkbox("Override Win?", value=st.session_state.override_fb, key = "override_fb")
        
        
        st.write(" ")
        
        
        
        
        
        # Peach Bowl
        if "override_pb" not in st.session_state:
            st.session_state.override_pb = False

        results_pb = theor_prepped[
            (theor_prepped["t1_team"] == seed4) &
            (theor_prepped["t2_team"] == fr4_winner) &
            (theor_prepped["neutral_site"] == 1)]     
    
        # Display the header for the game
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
        
        # Conditional Display of Matchup based on session_state
        if st.session_state.override_pb == True:
            
            st.markdown("<div style='display: flex; justify-content: center;'><img src='" + results_pb.iloc[0]["losing_logo"] + "' width='60'></div>", unsafe_allow_html=True)
            
            st.markdown(f"""
                <div style="font-size:15px; font-weight:bold; color:{results_pb.iloc[0]["winning_color"]}; text-align:center;">
                      
                </div>
            """, unsafe_allow_html=True)
            
            pb_winner = results_pb.iloc[0]["losing_team"]
            
        else:
            st.markdown("<div style='display: flex; justify-content: center;'><img src='" + results_pb.iloc[0]["winning_logo"] + "' width='60'></div>", unsafe_allow_html=True)
            
            # If override is checked, use a different color or text for the result
            st.markdown(f"""
                <div style="font-size:15px; font-weight:bold; color:{results_pb.iloc[0]["winning_color"]}; text-align:center;">
                    {results_pb.iloc[0]["pred_winner_wp"]}% wp
                </div>
            """, unsafe_allow_html=True)
            
            pb_winner = results_pb.iloc[0]["winning_team"]
        
            
        # Place the checkbox at the bottom of the page (after everything else)
        st.checkbox("Override Win?", value=st.session_state.override_pb, key = "override_pb")
        
        
        
        
  


  with col7:
    
    
        st.markdown(f"""
            <div style="font-size:30px; font-weight:bold; text-align:center; line-height:1.75; color:orange;">
                Semis <br>
            </div>
        """, unsafe_allow_html=True)


        # Cotton Bowl
        if "override_cb" not in st.session_state:
            st.session_state.override_cb = False

        results_cb = theor_prepped[
            (theor_prepped["t1_team"] == rb_winner) &
            (theor_prepped["t2_team"] == pb_winner) &
            (theor_prepped["neutral_site"] == 1)] 
    
        # Display the header for the game
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
        
        # Conditional Display of Matchup based on session_state
        if st.session_state.override_cb == True:
            
            st.markdown("<div style='display: flex; justify-content: center;'><img src='" + results_cb.iloc[0]["losing_logo"] + "' width='60'></div>", unsafe_allow_html=True)
            
            st.markdown(f"""
                <div style="font-size:15px; font-weight:bold; color:{results_cb.iloc[0]["winning_color"]}; text-align:center;">
                      
                </div>
            """, unsafe_allow_html=True)
            
            cb_winner = results_cb.iloc[0]["losing_team"]
            
        else:
            st.markdown("<div style='display: flex; justify-content: center;'><img src='" + results_cb.iloc[0]["winning_logo"] + "' width='60'></div>", unsafe_allow_html=True)
            
            # If override is checked, use a different color or text for the result
            st.markdown(f"""
                <div style="font-size:15px; font-weight:bold; color:{results_cb.iloc[0]["winning_color"]}; text-align:center;">
                    {results_cb.iloc[0]["pred_winner_wp"]}% wp
                </div>
            """, unsafe_allow_html=True)
            
            cb_winner = results_cb.iloc[0]["winning_team"]
        
            
        # Place the checkbox at the bottom of the page (after everything else)
        st.checkbox("Override Win?", value=st.session_state.override_cb, key = "override_cb")
        
        
        st.write(" ")
        
       
       
        
        # Orange Bowl
        if "override_ob" not in st.session_state:
            st.session_state.override_ob = False

        results_ob = theor_prepped[
            (theor_prepped["t1_team"] == sb_winner) &
            (theor_prepped["t2_team"] == fb_winner) &
            (theor_prepped["neutral_site"] == 1)]     
    
        # Display the header for the game
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
        
        # Conditional Display of Matchup based on session_state
        if st.session_state.override_ob == True:
            
            st.markdown("<div style='display: flex; justify-content: center;'><img src='" + results_ob.iloc[0]["losing_logo"] + "' width='60'></div>", unsafe_allow_html=True)
            
            st.markdown(f"""
                <div style="font-size:15px; font-weight:bold; color:{results_ob.iloc[0]["winning_color"]}; text-align:center;">
                      
                </div>
            """, unsafe_allow_html=True)
            
            ob_winner = results_ob.iloc[0]["losing_team"]
            
        else:
            st.markdown("<div style='display: flex; justify-content: center;'><img src='" + results_ob.iloc[0]["winning_logo"] + "' width='60'></div>", unsafe_allow_html=True)
            
            # If override is checked, use a different color or text for the result
            st.markdown(f"""
                <div style="font-size:15px; font-weight:bold; color:{results_ob.iloc[0]["winning_color"]}; text-align:center;">
                    {results_ob.iloc[0]["pred_winner_wp"]}% wp
                </div>
            """, unsafe_allow_html=True)
            
            ob_winner = results_ob.iloc[0]["winning_team"]
        
            
        # Place the checkbox at the bottom of the page (after everything else)
        st.checkbox("Override Win?", value=st.session_state.override_ob, key = "override_ob")
        
        
        
        
        


  with col9:
    
        st.markdown(f"""
            <div style="font-size:30px; font-weight:bold; text-align:center; line-height:1.75; color:orange;">
                Natty <br>
            </div>
        """, unsafe_allow_html=True)


        # National Championship
        if "override_nc" not in st.session_state:
            st.session_state.override_nc = False

        results_nc = theor_prepped[
            (theor_prepped["t1_team"] == cb_winner) &
            (theor_prepped["t2_team"] == ob_winner) &
            (theor_prepped["neutral_site"] == 1)]     
    
        # Display the header for the game
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
        
        # Conditional Display of Matchup based on session_state
        if st.session_state.override_nc == True:
            
            st.markdown("<div style='display: flex; justify-content: center;'><img src='" + results_nc.iloc[0]["losing_logo"] + "' width='60'></div>", unsafe_allow_html=True)
            
            st.markdown(f"""
                <div style="font-size:15px; font-weight:bold; color:{results_nc.iloc[0]["winning_color"]}; text-align:center;">
                      
                </div>
            """, unsafe_allow_html=True)
            
            nc_winner = results_nc.iloc[0]["losing_team"]
            
        else:
            st.markdown("<div style='display: flex; justify-content: center;'><img src='" + results_nc.iloc[0]["winning_logo"] + "' width='60'></div>", unsafe_allow_html=True)
            
            # If override is checked, use a different color or text for the result
            st.markdown(f"""
                <div style="font-size:15px; font-weight:bold; color:{results_nc.iloc[0]["winning_color"]}; text-align:center;">
                    {results_nc.iloc[0]["pred_winner_wp"]}% wp
                </div>
            """, unsafe_allow_html=True)
            
            nc_winner = results_nc.iloc[0]["winning_team"]
        
            
        # Place the checkbox at the bottom of the page (after everything else)
        st.checkbox("Override Win?", value=st.session_state.override_nc, key = "override_nc")
        
        
              

        
        

  
  
  
                
                

# GAME PREDICTOR PAGE
def game_predictor_page():
  
  
  # Streamlit App
  st.title("Game Predictor")
  
  st.write("Pick a combination of teams and see what the model predicts if they played today!")

  # Create columns for layout
  col1, col2 = st.columns([1,2])
    
  
  # filters
  with col1:
      
      
      st.write("### Filters")
      
      # Select Away Team
      away_team = st.selectbox("Select Away Team", sorted(theor_prepped["t2_team"].dropna().unique()), index = 57)
      
      # Select Home Team
      home_team = st.selectbox("Select Home Team", sorted(theor_prepped["t1_team"].dropna().unique()), index = 75)
      
      
      # Dropdown menu for Yes or No for neutral site
      neutral_site_input = st.selectbox("Neutral Site?", ["No", "Yes"])
      
      # Encode Yes as 1 and No as 0
      neutral_site_ind = 1 if neutral_site_input == "Yes" else 0
      
      st.write("  ")
      st.write(" ")
      
      st.markdown(f"""
          <div style="font-size:18px; font-weight:bold; text-align:center;">
              Data last updated after <br> <span style="color: orange;"> Week {this_week} </span>
          </div>
      """, unsafe_allow_html=True)

      
  
  # column two for predictions
  with col2:  
      
      if home_team == away_team:
          
        # formatting
        st.markdown(f"""
                                <div style="font-size:35px; font-weight:bold; text-align:center;">
                                    Please adjust one of the teams!
                                </div>
                            """, unsafe_allow_html=True)
        st.write("  ")
        st.write("  ")

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

        if results.iloc[0]["real_game"] == 1:
            st.markdown(f"""
                    <div style="font-size:35px; font-weight:bold; color: gold; text-align:center;">
                        Real Matchup
                    </div>
                """, unsafe_allow_html=True)            
                            
                            


def power_rankings_page():

    # Streamlit App
    st.title("The Czar Poll")
    
    st.write("If every team played each other home, away, and neutral, these are our predicted standings:")

    unique_conf = sorted(theor_agg['Conf'].unique())


    conf_options = st.multiselect(
    "Conference", 
    options=["All"] + unique_conf,  
    default=["All"]  
    )

    theor_agg_filt = theor_agg
    if "All" not in conf_options:
        theor_agg_filt = theor_agg_filt[theor_agg_filt['Conf'].isin(conf_options)]

    # Show the filtered data in Streamlit
    st.table(theor_agg_filt)






def this_week_page():

    # --- Add near the top of this_week_page(), before you need the value ---
    # State init
    if "show_spread_input" not in st.session_state:
        st.session_state.show_spread_input = False
    if "new_spread_raw" not in st.session_state:
        st.session_state.new_spread_raw = ""          # raw string from text_input
    if "new_spread_val" not in st.session_state:
        st.session_state.new_spread_val = None        # parsed float or None
    if "new_spread_error" not in st.session_state:
        st.session_state.new_spread_error = ""

    if "show_hml_input" not in st.session_state:
        st.session_state.show_hml_input = False
    if "new_hml_raw" not in st.session_state:
        st.session_state.new_hml_raw = ""          # raw string from text_input
    if "new_hml_val" not in st.session_state:
        st.session_state.new_hml_val = None        # parsed float or None
    if "new_hml_error" not in st.session_state:
        st.session_state.new_hml_error = ""          

    if "show_aml_input" not in st.session_state:
        st.session_state.show_aml_input = False
    if "new_aml_raw" not in st.session_state:
        st.session_state.new_aml_raw = ""          # raw string from text_input
    if "new_aml_val" not in st.session_state:
        st.session_state.new_aml_val = None        # parsed float or None
    if "new_aml_error" not in st.session_state:
        st.session_state.new_aml_error = ""            

    if "show_over_input" not in st.session_state:
        st.session_state.show_over_input = False
    if "new_over_raw" not in st.session_state:
        st.session_state.new_over_raw = ""          # raw string from text_input
    if "new_over_val" not in st.session_state:
        st.session_state.new_over_val = None        # parsed float or None
    if "new_over_error" not in st.session_state:
        st.session_state.new_over_error = ""        

    # Helpers
    def toggle_input():
        st.session_state.show_spread_input = not st.session_state.show_spread_input
        st.session_state.show_hml_input = not st.session_state.show_hml_input
        st.session_state.show_aml_input = not st.session_state.show_aml_input
        st.session_state.show_over_input = not st.session_state.show_over_input
     

    def _force_rerun():
        # works on both older and newer Streamlit
        if hasattr(st, "rerun"):
            st.rerun()
        else:
            st.experimental_rerun()

    def save_new_spread():
        raw = (st.session_state.get("new_spread_raw", "") or "").strip()
        if raw == "":
            # empty input -> clear override
            st.session_state.new_spread_val = None
            st.session_state.new_spread_error = ""
            #_force_rerun()
            return
        try:
            st.session_state.new_spread_val = float(raw)
            st.session_state.new_spread_error = ""
        except ValueError:
            st.session_state.new_spread_val = None
            st.session_state.new_spread_error = "Please enter a numeric value"

        

    def save_new_hml():
        raw = (st.session_state.get("new_hml_raw", "") or "").strip()
        if raw == "":
            # empty input -> clear override
            st.session_state.new_hml_val = None
            st.session_state.new_hml_error = ""
            #_force_rerun()
            return
        try:
            if raw.startswith("+"):
                raw = raw[1:]            
            st.session_state.new_hml_val = float(raw)
            st.session_state.new_hml_error = ""
        except ValueError:
            st.session_state.new_hml_val = None
            st.session_state.new_hml_error = "Please enter a numeric value"     


    def save_new_aml():
        raw = (st.session_state.get("new_aml_raw", "") or "").strip()
        if raw == "":
            # empty input -> clear override
            st.session_state.new_aml_val = None
            st.session_state.new_aml_error = ""
            #_force_rerun()
            return
        try:
            if raw.startswith("+"):
                raw = raw[1:]
            st.session_state.new_aml_val = float(raw)
            st.session_state.new_aml_error = ""
        except ValueError:
            st.session_state.new_aml_val = None
            st.session_state.new_aml_error = "Please enter a numeric value"                      


    def save_new_over():
        raw = (st.session_state.get("new_over_raw", "") or "").strip()
        if raw == "":
            # empty input -> clear override
            st.session_state.new_over_val = None
            st.session_state.new_over_error = ""
            #_force_rerun()
            return
        try:
            st.session_state.new_over_val = float(raw)
            st.session_state.new_over_error = ""
        except ValueError:
            st.session_state.new_over_val = None
            st.session_state.new_over_error = "Please enter a numeric value"        

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
   

   

  

    # Streamlit App
    st.title("This Week's Picks")

    st.write("Here are our picks for the week. You can toggle between the data frame view or one game at a time.")

    st.write("'Value' is a way to sort our predicted best value picks. Spread Value is the difference between our predicted point" \
    " differential sportsbooks' (spread*-1). ML Value is the percentage difference between our predicted win probability for the team with value" \
    " and sportsbooks' implied wp (converted from ML). Over/Under Value is the difference in our predicted total points and sportsbooks'.")

    st.write("To look at more information for Spread, ML, or O/U, select the 'Bet Type' filter. To sort by one of those values, select" \
    " that filter.")

    page_choice = st.selectbox(
        "Choose Display",
        ["All Games", "One Game"]
    )

    st.markdown(f"""
    <div style="font-size:35px; font-weight:bold; text-align:left;">
        <br> <span style="color: orange;"> Week {this_week} </span>
    </div>
    """, unsafe_allow_html=True) 

    


    if page_choice == "All Games":


            # Create columns for layout
            col1, col2, col3 = st.columns([2, .5, 8])
            



            with col1:



                # Get unique teams and sort them in ascending order (for both t1 and t2)
                unique_teams = pd.unique(merged_predictions[['Home', 'Away']].values.ravel('K'))
                unique_teams_sorted = sorted(unique_teams)

                # Get unique teams and sort them in ascending order (for both t1 and t2)
                unique_conf = pd.unique(merged_predictions[['Home Conf', 'Away Conf']].values.ravel('K'))
                unique_conf_sorted = sorted(unique_conf)
                    
        


                this_week_display_df = merged_predictions[["Home", "Away", "Spread Bet on:", "ML Bet on:", "O/U Bet on:", "Spread Value",
                    "ML Value", "O/U Value", "Home Conf", "Away Conf", "Neutral?"]].sort_values(by="Spread Value", ascending=False)

                this_week_display_spread_df = merged_predictions[["Home", "Away", "Spread Bet on:", "Spread Value", "Pred Home Cover Prob", "Book Home Spread",
                                                                "Home Conf", "Away Conf", "Neutral?"]].sort_values(by="Spread Value", ascending=False)

                this_week_display_ml_df = merged_predictions[["Home", "Away", "ML Bet on:", "ML Value", "Pred Home Win Prob", "Book Home Win Prob", "Pred Away Win Prob", 
                                                            "Book Away Win Prob", "Home Conf", "Away Conf", "Neutral?"]].sort_values(by="ML Value", ascending=False)

                this_week_display_ou_df = merged_predictions[["Home", "Away", "O/U Bet on:", "O/U Value", "Pred Over Prob", "Book O/U", "Home Conf", 
                                                            "Away Conf", "Neutral?"]].sort_values(by="O/U Value", ascending=False)      
            

                #st.write("### Filters")
                
                # Create dropdowns for filtering options on the left column
                bet_type_options = st.selectbox(
                    "Bet Type", 
                    options=["All", "Spread", "Moneyline", "Over/Under"],  
                    index=0  
                )

                # Create dropdowns for filtering options on the left column
                team_options = st.selectbox(
                    "Team", 
                    options=["All"] + unique_teams_sorted,  
                    index=0  
                )             

                conf_options = st.selectbox(
                    "Conference", 
                    options=["All"] + unique_conf_sorted,  
                    index = 0 
                )

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
                        


                
                if bet_type_options == "All":
                    picks_df = this_week_display_df
                elif bet_type_options == "Spread":
                    picks_df = this_week_display_spread_df
                elif bet_type_options == "Moneyline":
                    picks_df = this_week_display_ml_df
                else:
                    picks_df = this_week_display_ou_df


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


            with col3:

                st.dataframe(picks_df, height = 600)
          
    
    
    else:
            
            



            unique_matchups = sorted(pd.unique(merged_predictions["matchup"]))

            st.write("")

            matchup_options = st.selectbox(
                "Select Matchup", 
                options=unique_matchups,  
                index=0,
                on_change=reset_lines
            )    

            filtered_df = merged_predictions.loc[merged_predictions["matchup"]==matchup_options]  
            

            if st.session_state.new_spread_val is not None and st.session_state.new_over_val is None:
                new_spread = st.session_state.new_spread_val
                current_home = filtered_df.iloc[0]["Home"]
                mask = (sample_data["t1_team"].eq(current_home)) | (sample_data["t2_team"].eq(current_home))
                cut_sample_data = sample_data.loc[mask].copy()
                cut_sample_data.loc[cut_sample_data["t1_team"].eq(current_home), "t1_book_spread"] = new_spread
                cut_sample_data.loc[cut_sample_data["t2_team"].eq(current_home), "t1_book_spread"] = -new_spread
                filtered_df = make_merged_predictions(cut_sample_data)
            elif st.session_state.new_spread_val is None and st.session_state.new_over_val is not None:
                new_over = st.session_state.new_over_val
                current_home = filtered_df.iloc[0]["Home"]
                mask = (sample_data["t1_team"].eq(current_home)) | (sample_data["t2_team"].eq(current_home))
                cut_sample_data = sample_data.loc[mask].copy()
                cut_sample_data["book_over_under"] = new_over
                filtered_df = make_merged_predictions(cut_sample_data)    
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
        

            # Create columns for layout
            col1, col2, col3, col4, col5 = st.columns([3, .5, 3, .5, 3]) 


            with col1:

                    
                    st.write("")

                    st.markdown("<div style='display: flex; justify-content: center;'><img src='" + filtered_df.iloc[0]["away_logo"] + "' width='150'></div>", unsafe_allow_html=True)
            
                    st.markdown(f"""
                                <div style="font-size:35px; font-weight:bold; color:{filtered_df.iloc[0]["away_color"]}; text-align:center;">
                                    {filtered_df.iloc[0]["Away"]}
                                </div>
                            """, unsafe_allow_html=True)
                    
                    st.write(" ")
                    st.write(" ")

                    st.markdown(f"""
                                <div style="font-size:20px; color: white; text-align:left;">
                                    Pred Cover Prob: {100-filtered_df.iloc[0]["Pred Home Cover Prob"]}%
                                </div>
                            """, unsafe_allow_html=True)  
                          

                    if filtered_df.iloc[0]["Book Home Spread"] < 0:
                        st.markdown(f"""
                                    <div style="font-size:20px; color: white; text-align:left;">
                                        Book Spread: +{filtered_df.iloc[0]["Book Home Spread"]*-1}
                                    </div>
                                """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                                    <div style="font-size:20px; color: white; text-align:left;">
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
                                    <div style="font-size:20px; color: white; text-align:left;">
                                        Pred ML: +{int(round(filtered_df.iloc[0]["Pred Away Moneyline"],0))}
                                    </div>
                                """, unsafe_allow_html=True)   
                    else:
                        st.markdown(f"""
                                    <div style="font-size:20px; color: white; text-align:left;">
                                        Pred ML: {int(round(filtered_df.iloc[0]["Pred Away Moneyline"],0))}
                                    </div>
                                """, unsafe_allow_html=True)                           

                    st.markdown(f"""
                                <div style="font-size:20px; color: white; text-align:left;">
                                    Pred Win Prob: {round(filtered_df.iloc[0]["Pred Away Win Prob"],2)}%
                                </div>
                            """, unsafe_allow_html=True)                         

                    if filtered_df.iloc[0]["Book Away Moneyline"]>0:
                            st.markdown(f"""
                                        <div style="font-size:20px; color: white; text-align:left;">
                                            Book ML: +{int(round(filtered_df.iloc[0]["Book Away Moneyline"],0))}
                                        </div>
                                    """, unsafe_allow_html=True)  
                    else:  
                            st.markdown(f"""
                                        <div style="font-size:20px; color: white; text-align:left;">
                                            Book ML: {int(round(filtered_df.iloc[0]["Book Away Moneyline"],0))}
                                        </div>
                                    """, unsafe_allow_html=True)                             

                    st.markdown(f"""
                                <div style="font-size:20px; color: white; text-align:left;">
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
                         

 




            with col2:
                st.write("")


            with col3:

                st.write("")


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
                            <div style="font-size:25px; color: white; font-weight:bold; text-align:center;">
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

                st.markdown(f"""
                        <div style="font-size:20px; color: white; text-align:center;">
                            Pred Over Prob: {filtered_df.iloc[0]["Pred Over Prob"]}%
                        </div>
                    """, unsafe_allow_html=True)

                st.markdown(f"""
                        <div style="font-size:20px; color: white; text-align:center;">
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
                    




                # Alternative: Streamlit button but with centering
                st.markdown(
                    """
                    <style>
                    div[data-testid="stButton"] {display: flex; justify-content: center;}
                    </style>
                    """,
                    unsafe_allow_html=True
                )
                st.button("Change book lines?", on_click=toggle_input)

                if st.session_state.show_spread_input:
                    st.text_input(
                        "Enter new book spread:",
                        key="new_spread_raw",
                        placeholder="e.g., -3.5",
                        on_change=save_new_spread,   # <- stores float into session + reruns
                    )
                    if st.session_state.new_spread_error:
                        st.warning(st.session_state.new_spread_error)

                if st.session_state.show_hml_input:
                    st.text_input(
                        "Enter new book home ml:",
                        key="new_hml_raw",
                        placeholder="e.g., -230, 300",
                        on_change=save_new_hml,   # <- stores float into session + reruns
                    )
                    if st.session_state.new_hml_error:
                        st.warning(st.session_state.new_hml_error)  

                if st.session_state.show_aml_input:
                    st.text_input(
                        "Enter new book away ml:",
                        key="new_aml_raw",
                        placeholder="e.g., -230, 300",
                        on_change=save_new_aml,   # <- stores float into session + reruns
                    )
                    if st.session_state.new_aml_error:
                        st.warning(st.session_state.new_aml_error)         



                if st.session_state.show_over_input:
                    st.text_input(
                        "Enter new book over under:",
                        key="new_over_raw",
                        placeholder="e.g., 48.5",
                        on_change=save_new_over,   # <- stores float into session + reruns
                    )
                    if st.session_state.new_over_error:
                        st.warning(st.session_state.new_over_error)  

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






            with col4:
                st.write("")


            with col5:

                st.write("") 

                st.markdown("<div style='display: flex; justify-content: center;'><img src='" + filtered_df.iloc[0]["home_logo"] + "' width='150'></div>", unsafe_allow_html=True)
                                       
                st.markdown(f"""
                            <div style="font-size:35px; font-weight:bold; color:{filtered_df.iloc[0]["home_color"]}; text-align:center;">
                                {filtered_df.iloc[0]["Home"]}
                            </div>
                        """, unsafe_allow_html=True)       

                st.write(" ")
                st.write(" ")


                st.markdown(f"""
                            <div style="font-size:20px; color: white; text-align:right;">
                                {filtered_df.iloc[0]["Pred Home Cover Prob"]}% - Pred Cover Prob
                            </div>
                        """, unsafe_allow_html=True)  
                
                                             

                if filtered_df.iloc[0]["Book Home Spread"] > 0:
                    st.markdown(f"""
                                <div style="font-size:20px; color: white; text-align:right;">
                                    +{filtered_df.iloc[0]["Book Home Spread"]} - Book Spread
                                </div>
                            """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                                <div style="font-size:20px; color: white; text-align:right;">
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
                                <div style="font-size:20px; color: white; text-align:right;">
                                    +{int(round(filtered_df.iloc[0]["Pred Home Moneyline"],0))} - Pred ML
                                </div>
                            """, unsafe_allow_html=True)   
                else:
                    st.markdown(f"""
                                <div style="font-size:20px; color: white; text-align:right;">
                                    {int(round(filtered_df.iloc[0]["Pred Home Moneyline"],0))} - Pred ML
                                </div>
                            """, unsafe_allow_html=True)                           

                st.markdown(f"""
                            <div style="font-size:20px; color: white; text-align:right;">
                                {round(filtered_df.iloc[0]["Pred Home Win Prob"],2)}% - Pred Win Prob
                            </div>
                        """, unsafe_allow_html=True)                         

                if filtered_df.iloc[0]["Book Home Moneyline"]>0:
                        st.markdown(f"""
                                    <div style="font-size:20px; color: white; text-align:right;">
                                        +{int(round(filtered_df.iloc[0]["Book Home Moneyline"],0))} - Book ML
                                    </div>
                                """, unsafe_allow_html=True)  
                else:  
                        st.markdown(f"""
                                    <div style="font-size:20px; color: white; text-align:right;">
                                        {int(round(filtered_df.iloc[0]["Book Home Moneyline"],0))} - Book ML
                                    </div>
                                """, unsafe_allow_html=True)                             

                st.markdown(f"""
                            <div style="font-size:20px; color: white; text-align:right;">
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


        
 
   


            

    
    
# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ("Home", "Betting Accuracy", "This Week's Picks", "The Czar Poll", "The Playoff", "Game Predictor"))




# Display the selected page
if page == "Home":
    welcome_page()
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
