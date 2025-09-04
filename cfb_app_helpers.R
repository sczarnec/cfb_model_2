
# for predicting book metrics for theoretical games
book_function = function(oc_var, test_data){
  
  
  # ARGS:
  #
  # oc_var = outcome variable of model
  # test_data = data used for testing
  #
  # RETURNS: predictions for the book variable 
  
  
  
  # put oc_var at end
  book_model_prep = model_prepped_omit[, c(setdiff(names(model_prepped_omit), oc_var), oc_var)]
  
  # get rid of column not oc (null for this week)
  if (oc_var == "book_over_under"){
    get_rid_col = "t1_book_spread"
  } else{
    get_rid_col = "book_over_under"
  }
  
  
  
  
  # train is from specified window
  train_book = book_model_prep %>% 
    filter(season >= 2020) %>% 
    select(-c(get_rid_col))
  
  names(train_book)[grep(oc_var, names(train_book))] = "oc_var"
  
  
  
  
  
  # variable selection
  train_v1_book = train_book %>%
    select(-c(grep("rol_off", names(train_book)), grep("rol_def", names(train_book))))
  
  train_v2_book = train_book %>%
    select(-c(grep("off_l", names(train_book)), grep("def_l", names(train_book))))        
  
  
  train_matrix_v1_book = xgb.DMatrix(data = as.matrix(train_v1_book[,1:587]), 
                                     label = as.numeric(train_v1_book$oc_var))
  
  # train_matrix_v2_book = xgb.DMatrix(data = as.matrix(train_v2_book[,1:1251]), 
  #                                    label = as.numeric(train_v2_book$oc_var))
  
  
  new_params_book = c(parameters_book, data = train_matrix_v1_book)
  model_v1_book = do.call(xgboost, new_params_book)
  imp_mat_v1_book <- xgb.importance(model = model_v1_book)
  
  # new_params_book = c(parameters_book, data = train_matrix_v2_book)
  # model_v2_book = do.call(xgboost, new_params_book)
  # imp_mat_v2_book <- xgb.importance(model = model_v2_book)
  
  
  var_list_orig_book = unique(c(imp_mat_v1_book[1:100]$Feature))
  list_matches_book = unname(sapply(var_list_orig_book, function(x) gsub("^t1", "t2", x)))
  var_list_final_book = unique(c(var_list_orig_book, list_matches_book))
  
  if ("t2_book_spread" %in% var_list_final_book){
    var_list_final_book =  var_list_final_book[-c(grep("t2_book_spread", var_list_final_book))]
  }
  
  
  
  
  # train model
  
  # new train data with selected cols
  train_book = train_book %>% select(var_list_final_book, oc_var)
  num_predictors_book = length(var_list_final_book)
  
  
  # Create training matrix
  train_matrix_book = xgb.DMatrix(data = as.matrix(train_book[,1:num_predictors_book]), 
                                  label = as.integer(train_book$oc_var))
  
  # call xgboost
  new_params_book = c(parameters_book, data = train_matrix_book)
  model_book = do.call(xgboost, new_params_book)
  
  
  
  
  
  # find preds
  
  
  test_book = test_data %>% select(var_list_final_book, t1_book_spread)
  test_matrix_book = xgb.DMatrix(data = as.matrix(test_book[,1:num_predictors_book]))
  preds_book = predict(model_book, test_matrix_book)
  
  
  
  
  
  
  return(preds_book)
  
  
}






# test_book = test_book %>% select(var_list_final_book, t1_book_spread)
# 
# test_matrix_book = xgb.DMatrix(data = as.matrix(test_book[,1:num_predictors_book]), 
#                               label = as.integer(test_book$t1_book_spread))
# 
# 
# 
# preds_book = predict(model_book, test_matrix_book)
# new_df_book = cbind.data.frame(preds_book, t1_book_spread = test_book$t1_book_spread)
# 
# 
# rmse(new_df_book$preds_book, new_df_book$t1_book_spread)
# mae(new_df_book$preds_book, new_df_book$t1_book_spread)

# MAE of 3.27 and RMSE of 4.32 when trained on 2020-23 and tested 2024








# for initial training on multiple years

model_func = function(oc_var, data, yr_length, real_yr, real_wk, initial_yr, initial_wk, params, params_tune){
  
  
  # ARGS:
  #
  # oc_var = outcome variable of model
  # data = dataframe used
  # yr_length = window size
  # real_yr = this season
  # real_wk = this week
  # initial_yr = starting year for training
  # initial_wk = starting week for training
  # params = xgboost parameters (as list)
  # params_tune = xhboost parameters for tuning (as list)
  #
  # RETURNS: dataframe of predictions from the specified period
  
  
  
  set.seed(844)
  
  
  # what years are we looking at
  year_range = c((initial_yr + yr_length):real_yr)
  
  for (yr in year_range){
    
    print(yr)
    iter = 0
    
    current_yr = yr
    start_yr = current_yr - yr_length
    
    
    
    # for each year, what weeks are available in data
    if (current_yr %in% c(2010)){
      week_range = c(4:16, 20, 22, 23)
    } else if (current_yr <= 2023) {
      week_range = c(4:15, 20, 22, 23)
    } else{
      week_range = c(4:15, 20:23)
    }
    
    # stop at current year and week when reached
    if (current_yr == real_yr){
      week_range = week_range[week_range<=real_wk]
    }
    if (current_yr == initial_yr){
      week_range = week_range[week_range>=initial_wk]
    }
    
    
    
    # for each week, run the model with rolling window
    for (wk in week_range){
      
      print(wk)
      iter = iter + 1
      
      current_wk = wk
      
      
      
      
      
      # train is from specified window
      train = data %>% 
        filter((season == start_yr & week > current_wk) | (season > start_yr & season < current_yr) | (season == current_yr & week < current_wk))
      
      
      names(train)[grep(oc_var, names(train))] = "oc_var"
      
      
      
      # run through different data assets to find important variables
      train_v1 = train %>%
        select(-c(grep("rol_off", names(train)), grep("rol_def", names(train))))
      
      # train_v2 = train %>%
      #   select(-c(grep("off_l", names(train)), grep("def_l", names(train))))        
      
      train_matrix_v1 = xgb.DMatrix(data = as.matrix(train_v1[,1:589]), 
                                    label = as.numeric(train_v1$oc_var))
      
      # train_matrix_v2 = xgb.DMatrix(data = as.matrix(train_v2[,1:1253]), 
      #                               label = as.numeric(train_v2$oc_var))
      
      set.seed(844)      
      new_params = c(params_tune, data = train_matrix_v1)
      model_v1 = do.call(xgboost, new_params)
      imp_mat_v1 <- xgb.importance(model = model_v1)
      
      # set.seed(844)      
      # new_params = c(params_tune, data = train_matrix_v2)
      # model_v2 = do.call(xgboost, new_params)
      # imp_mat_v2 <- xgb.importance(model = model_v2)
      
      
      
      # take top 100 from each variable importance
      var_list_orig = unique(c(imp_mat_v1[1:200]$Feature))
      
      
      
      # final feature list
      list_matches = unname(sapply(var_list_orig, function(x) gsub("^t1", "t2", x)))
      var_list_final = unique(c(var_list_orig, list_matches))
      if ("t2_book_spread" %in% var_list_final){
        var_list_final =  var_list_final[-c(grep("t2_book_spread", var_list_final))]
      }
      if (!"t1_book_spread" %in% var_list_final & oc_var == "actual_t1_cover"){
        var_list_final =  c(var_list_final, "t1_book_spread")
      }
      if (!"book_over_under" %in% var_list_final & oc_var == "actual_over"){
        var_list_final =  c(var_list_final, "book_over_under")
      }
      train = train %>% select(var_list_final, oc_var)
      num_predictors = length(var_list_final)
      
      
      
      
      
      
      # run actual model
      
      
      # Create training matrix
      train_matrix = xgb.DMatrix(data = as.matrix(train[,1:num_predictors]), 
                                 label = as.integer(train$oc_var))
      
      set.seed(844)         
      # call xgboost
      new_params = c(params, data = train_matrix)
      model = do.call(xgboost, new_params)
      
      
      
      
      
      
      
      
      # test is from this week only
      test = data %>% 
        filter(season == current_yr & week == current_wk) %>% 
        select(var_list_final, oc_var, game_id, t1_team, t2_team)
      
      names(test)[grep(oc_var, names(test))] = "oc_var"
      
      # Create training matrix
      test_matrix = xgb.DMatrix(data = as.matrix(test[,1:num_predictors]), 
                                label = as.numeric(test$oc_var))
      
      
      # predictions
      preds = predict(model, test_matrix)
      new_df = cbind.data.frame(preds, oc_var = test$oc_var, game_id = test$game_id, t1_team = test$t1_team, t2_team = test$t2_team)
      
      if (yr == initial_yr + yr_length & iter == 1){
        pred_df = new_df
      } else{
        pred_df = bind_rows(pred_df, new_df)
      }
      
      
      # save out models
      xgb.save(model,fname = paste0("saved_models/", yr, wk, "_",  oc_var,"_model.model"))
      write.csv(var_list_final, paste0("saved_models/", yr, wk, "_", oc_var, "model_var_list.csv"))
      
      if (real_wk == wk | real_yr == yr){ 
        
        xgb.save(model, fname = paste0("most_recent_",  oc_var,"_model.model"))
        write.csv(var_list_final, paste0("most_recent_", oc_var, "model_var_list.csv"))
        
      }       
      
      
      
      
      
      
    }
    
  }
  
  
  return(pred_df)
  
}  






# for updating the model weekly

model_func_single_year = function(oc_var, data, yr_length, real_yr, wk_range, params, params_tune){
  
  
  # ARGS:
  #
  # oc_var = outcome variable of model
  # data = dataframe used
  # yr_length = window size
  # real_yr = this season
  # wk_range = what weeks do you want preds/models for
  # params = xgboost parameters (as list)
  # params_tune = xhboost parameters for tuning (as list)
  #
  # RETURNS: dataframe of predictions from the specified period
  
  
  
  set.seed(844)
  
  
  # what years are we looking at
  current_yr = real_yr
  start_yr = current_yr - yr_length
    
    
  week_range = wk_range
  max_week = max(week_range)
    
    
  iter = 0
    
    
  # for each week, run the model with rolling window
  for (wk in week_range){
    
    print(wk)
    iter = iter + 1
    
    current_wk = wk
    
    
    
    
    
    # train is from specified window
    train = data %>% 
      filter((season == start_yr & week > current_wk) | (season > start_yr & season < current_yr) | (season == current_yr & week < current_wk))
    
    
    
    
    names(train)[grep(oc_var, names(train))] = "oc_var"
    
    
    
    # run through different data assets to find important variables
    train_v1 = train %>%
      select(-c(grep("rol_off", names(train)), grep("rol_def", names(train))))
    
    # train_v2 = train %>%
    #   select(-c(grep("off_l", names(train)), grep("def_l", names(train))))   

    

    
    train_matrix_v1 = xgb.DMatrix(data = as.matrix(train_v1[,1:589]), 
                                  label = as.numeric(train_v1$oc_var))
    
    # train_matrix_v2 = xgb.DMatrix(data = as.matrix(train_v2[,1:1253]), 
    #                               label = as.numeric(train_v2$oc_var))
    
    set.seed(844)      
    new_params = c(params_tune, data = train_matrix_v1)
    model_v1 = do.call(xgboost, new_params)
    imp_mat_v1 <- xgb.importance(model = model_v1)
    
    # set.seed(844)      
    # new_params = c(params_tune, data = train_matrix_v2)
    # model_v2 = do.call(xgboost, new_params)
    # imp_mat_v2 <- xgb.importance(model = model_v2)
    
    
    
    # take top 100 from variable importance
    var_list_orig = unique(c(imp_mat_v1[1:200]$Feature))
    
    
    
    # final feature list
    list_matches = unname(sapply(var_list_orig, function(x) gsub("^t1", "t2", x)))
    var_list_final = unique(c(var_list_orig, list_matches))
    if ("t2_book_spread" %in% var_list_final){
      var_list_final =  var_list_final[-c(grep("t2_book_spread", var_list_final))]
    }
    if (!"t1_book_spread" %in% var_list_final & oc_var == "actual_t1_cover"){
      var_list_final =  c(var_list_final, "t1_book_spread")
    }
    if (!"book_over_under" %in% var_list_final & oc_var == "actual_over"){
      var_list_final =  c(var_list_final, "book_over_under")
    }
    train = train %>% select(var_list_final, oc_var)
    num_predictors = length(var_list_final)
    
    
    
    

    
    # run actual model
    
    
    # Create training matrix
    train_matrix = xgb.DMatrix(data = as.matrix(train[,1:num_predictors]), 
                               label = as.integer(train$oc_var))
    
    set.seed(844)         
    # call xgboost
    new_params = c(params, data = train_matrix)
    model = do.call(xgboost, new_params)
    
    
    
    

    
    
    
    # test is from this week only
    test = data %>% 
      filter(season == current_yr & week == current_wk) %>% 
      select(var_list_final, oc_var, game_id, t1_team, t2_team)
    
    names(test)[grep(oc_var, names(test))] = "oc_var"
    
    # Create training matrix
    test_matrix = xgb.DMatrix(data = as.matrix(test[,1:num_predictors]))
    
    
    # predictions
    preds = predict(model, test_matrix)
    new_df = cbind.data.frame(preds, oc_var = test$oc_var, game_id = test$game_id, t1_team = test$t1_team, t2_team = test$t2_team)
    
    if (iter == 1){
      pred_df = new_df
    } else{
      pred_df = bind_rows(pred_df, new_df)
    }
    
    
    # save out models
    xgb.save(model,fname = paste0("saved_models/", current_yr, wk, "_",  oc_var,"_model.model"))
    write.csv(var_list_final, paste0("saved_models/", current_yr, wk, "_", oc_var, "model_var_list.csv"))
    
    if (wk == max(wk_range)){ 
      
      xgb.save(model, fname = paste0("most_recent_",  oc_var,"_model.model"))
      write.csv(var_list_final, paste0("most_recent_", oc_var, "model_var_list.csv"))
      
    }       
    
    
    
    
    
    
  }
  
  

  
  return(pred_df)
  
}  




