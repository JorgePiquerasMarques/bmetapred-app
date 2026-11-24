library(shiny)
library(bslib)
library(DT)
library(dplyr)
library(tidyr)
library(ggplot2)
library(bmetapred)
library(future)
library(callr)
library(e1071)
library(nnet)
library(ranger)
library(rpart)
library(glmnet)

# ==============================================================================
# 1. MANUSCRIPT SIMULATION ENGINE (EXACT M-OPEN DGP + 12 ML BASE LEARNERS)
# ==============================================================================
generar_dataset_mopen <- function(N = 2000, z = 50, h = 0.5) {
  Z_mat <- matrix(rnorm(N * z), nrow = N, ncol = z)
  colnames(Z_mat) <- paste0("Z", 1:z)
  df <- as.data.frame(Z_mat)
  
  Y_lineal <- 3.5 * df$Z1 - 3.0 * df$Z2 + 2.5 * df$Z3 - 2.0 * df$Z4 +
    1.5 * df$Z5 + 1.2 * df$Z6 - 1.0 * df$Z7
  
  Y_nolineal <- 2 * sin(df$Z8 * 1.5) + 1.5 * cos(df$Z9) + 2 * (df$Z10 * df$Z11)
  
  Y_extremo <- 1.2 * sign(df$Z12) * (abs(df$Z12)^2.5) - 0.8 * (df$Z13^3)
  
  Y_true <- Y_lineal + Y_nolineal + Y_extremo
  
  sd_vector <- 0.1 + h * abs(Y_true)
  ruido <- rnorm(N, mean = 0, sd = sd_vector)
  df$Y <- Y_true + ruido
  
  return(df)
}

entrenar_modelos_base <- function(datos) {
  vars_x        <- setdiff(names(datos), "Y")
  correlaciones <- abs(cor(datos$Y, datos[, vars_x]))
  cor_ordenada  <- sort(correlaciones[1, ], decreasing = TRUE)
  
  Z_top15  <- names(cor_ordenada)[1:15]
  num_vars <- length(cor_ordenada)
  Z_worst9 <- names(cor_ordenada)[(num_vars - 8):num_vars]
  Z_worst  <- c(names(cor_ordenada)[3], Z_worst9)
  
  top_5_nombres <- names(cor_ordenada)[1:5]
  resto_nombres <- setdiff(vars_x, top_5_nombres)
  generar_Z_mid <- function() c(sample(top_5_nombres, 4), sample(resto_nombres, 6))
  
  Y_vec <- datos$Y
  lista_modelos <- list()
  
  guardar_modelo <- function(nombre, fit_principal, predicciones_train, vars_usadas,
                             min_y = 0, rango_y = 1, centro_x = NULL, escala_x = NULL) {
    rho <- suppressWarnings(cor(predicciones_train, Y_vec))
    if (is.na(rho)) rho <- 0
    error_cuad <- (Y_vec - predicciones_train)^2
    
    # Fit LOESS to squared residuals on Oracle set (micro-jitter prevents KD-tree ties in CARTs)
    M_pred_jitter <- predicciones_train + rnorm(length(predicciones_train), 0, 1e-5)
    df_err <- data.frame(err_sq = error_cuad, M_pred = M_pred_jitter)
    fit_loess <- suppressWarnings(loess(err_sq ~ M_pred, data = df_err, span = 0.4))
    
    return(list(
      nombre        = nombre,
      fit_principal = fit_principal,
      rho           = rho,
      vars          = vars_usadas,
      min_y         = min_y,
      rango_y       = rango_y,
      centro_x      = centro_x,
      escala_x      = escala_x,
      modelo_loess  = fit_loess
    ))
  }
  
  # --- TIER 1: HIGH PERFORMANCE (5 models, Z_top15) ---
  df_top15 <- datos[, c("Y", Z_top15)]
  X_top15  <- as.matrix(datos[, Z_top15])
  
  fit_svm_r <- e1071::svm(Y ~ ., data = df_top15, kernel = "radial")
  lista_modelos[[1]] <- guardar_modelo("SVM_Radial_Top", fit_svm_r, predict(fit_svm_r, df_top15), Z_top15)
  
  X_top15_scaled <- scale(X_top15)
  centro_X <- attr(X_top15_scaled, "scaled:center")
  escala_X <- attr(X_top15_scaled, "scaled:scale")
  rango_Y  <- max(Y_vec) - min(Y_vec)
  Y_norm   <- (Y_vec - min(Y_vec)) / rango_Y
  
  fit_nnet  <- nnet::nnet(X_top15_scaled, Y_norm, size = 10, linout = TRUE, trace = FALSE, maxit = 500, decay = 0.01)
  pred_nnet <- (predict(fit_nnet, X_top15_scaled)[, 1] * rango_Y) + min(Y_vec)
  lista_modelos[[2]] <- guardar_modelo("Neural_Net_Top", fit_nnet, pred_nnet, Z_top15,
                                       min_y = min(Y_vec), rango_y = rango_Y,
                                       centro_x = centro_X, escala_x = escala_X)
  
  fit_rf <- ranger::ranger(Y ~ ., data = df_top15)
  lista_modelos[[3]] <- guardar_modelo("Random_Forest_Top", fit_rf, fit_rf$predictions, Z_top15)
  
  fit_arbol_p <- rpart::rpart(Y ~ ., data = df_top15, control = rpart::rpart.control(cp = 0.001))
  lista_modelos[[4]] <- guardar_modelo("Arbol_Profundo_Top", fit_arbol_p, predict(fit_arbol_p, df_top15), Z_top15)
  
  fit_svm_p <- e1071::svm(Y ~ ., data = df_top15, kernel = "polynomial", degree = 3)
  lista_modelos[[5]] <- guardar_modelo("SVM_Polinomico_Top", fit_svm_p, predict(fit_svm_p, df_top15), Z_top15)
  
  # --- TIER 2: INTERMEDIATE PERFORMANCE (4 models, 4 top + 6 random covariates) ---
  Z_mid_lm  <- generar_Z_mid()
  df_mid_lm <- datos[, c("Y", Z_mid_lm)]
  fit_lm    <- lm(Y ~ ., data = df_mid_lm)
  lista_modelos[[6]] <- guardar_modelo("LM_Clasico_Mid", fit_lm, predict(fit_lm, df_mid_lm), Z_mid_lm)
  
  Z_mid_ridge <- generar_Z_mid()
  X_mid_ridge <- as.matrix(datos[, Z_mid_ridge])
  fit_ridge   <- glmnet::cv.glmnet(X_mid_ridge, Y_vec, alpha = 0)
  lista_modelos[[7]] <- guardar_modelo("Ridge_L2_Mid", fit_ridge, predict(fit_ridge, newx = X_mid_ridge, s = "lambda.min")[, 1], Z_mid_ridge)
  
  Z_mid_rf   <- generar_Z_mid()
  df_mid_rf  <- datos[, c("Y", Z_mid_rf)]
  fit_rf_mid <- ranger::ranger(Y ~ ., data = df_mid_rf)
  lista_modelos[[8]] <- guardar_modelo("Random_Forest_Mid", fit_rf_mid, fit_rf_mid$predictions, Z_mid_rf)
  
  Z_mid_svm   <- generar_Z_mid()
  df_mid_svm  <- datos[, c("Y", Z_mid_svm)]
  fit_svm_mid <- e1071::svm(Y ~ ., data = df_mid_svm, kernel = "radial")
  lista_modelos[[9]] <- guardar_modelo("SVM_Radial_Mid", fit_svm_mid, predict(fit_svm_mid, df_mid_svm), Z_mid_svm)
  
  # --- TIER 3: LOW PERFORMANCE (3 models, Z_worst) ---
  df_worst <- datos[, c("Y", Z_worst)]
  
  fit_svm_l <- e1071::svm(Y ~ ., data = df_worst, kernel = "linear")
  lista_modelos[[10]] <- guardar_modelo("SVM_Lineal_Worst", fit_svm_l, predict(fit_svm_l, df_worst), Z_worst)
  
  fit_arbol_s <- rpart::rpart(Y ~ ., data = df_worst, control = rpart::rpart.control(maxdepth = 3))
  lista_modelos[[11]] <- guardar_modelo("Arbol_Simple_Worst", fit_arbol_s, predict(fit_arbol_s, df_worst), Z_worst)
  
  fit_ciego <- lm(Y ~ ., data = df_worst)
  lista_modelos[[12]] <- guardar_modelo("LM_Ciego_Worst", fit_ciego, predict(fit_ciego, df_worst), Z_worst)
  
  df_resumen <- data.frame(
    ID_Modelo = 1:12,
    Algoritmo = sapply(lista_modelos, function(x) x$nombre),
    Rho_Train = sapply(lista_modelos, function(x) x$rho),
    stringsAsFactors = FALSE
  )
  df_resumen <- df_resumen[order(-df_resumen$Rho_Train), ]
  rownames(df_resumen) <- NULL
  
  return(list(modelos = lista_modelos, resumen_rendimiento = df_resumen))
}

predict_loess_robusto <- function(modelo_loess, nuevos_M) {
  rango_M <- modelo_loess$x
  min_M   <- min(rango_M, na.rm = TRUE)
  max_M   <- max(rango_M, na.rm = TRUE)
  media_M <- mean(rango_M, na.rm = TRUE)
  
  malos <- is.na(nuevos_M) | is.infinite(nuevos_M)
  if (any(malos)) nuevos_M[malos] <- media_M
  
  nuevos_M_seguros <- pmax(min_M, pmin(max_M, nuevos_M))
  df_new    <- data.frame(M_pred = nuevos_M_seguros)
  preds_var <- suppressWarnings(predict(modelo_loess, newdata = df_new))
  
  malos_pred <- is.na(preds_var) | is.nan(preds_var) | is.infinite(preds_var)
  if (any(malos_pred)) {
    val_rep <- median(preds_var[!malos_pred], na.rm = TRUE)
    if (is.na(val_rep) || is.nan(val_rep)) val_rep <- 0.04
    preds_var[malos_pred] <- val_rep
  }
  if (any(malos)) preds_var[malos] <- max(preds_var, na.rm = TRUE) * 10
  
  return(pmax(preds_var, 0.0004))
}

# Exact Scenario 2 (Oracle LOESS) prediction generator matching the manuscript
generar_X <- function(datos_nuevos, modelos_preentrenados, p = c(5, 4, 3)) {
  if (length(p) != 3) stop("Parameter 'p' must be a vector of length 3: c(Top, Mid, Worst).")
  p_total <- sum(p)
  
  resumen   <- modelos_preentrenados$resumen_rendimiento
  res_top   <- resumen[grepl("_Top", resumen$Algoritmo), ]
  res_mid   <- resumen[grepl("_Mid", resumen$Algoritmo), ]
  res_worst <- resumen[grepl("_Worst", resumen$Algoritmo), ]
  
  ids_seleccionados <- c()
  if (p[1] > 0) ids_seleccionados <- c(ids_seleccionados, res_top$ID_Modelo[1:p[1]])
  if (p[2] > 0) ids_seleccionados <- c(ids_seleccionados, res_mid$ID_Modelo[1:p[2]])
  if (p[3] > 0) ids_seleccionados <- c(ids_seleccionados, res_worst$ID_Modelo[1:p[3]])
  
  df_meta <- data.frame(Y = datos_nuevos$Y)
  
  for (i in 1:p_total) {
    id_modelo       <- ids_seleccionados[i]
    modelo          <- modelos_preentrenados$modelos[[id_modelo]]
    nombre_alg      <- modelo$nombre
    vars_del_modelo <- modelo$vars
    
    # 1. Exact base learner point prediction (no artificial jitter added to M_i)
    if (nombre_alg == "Neural_Net_Top") {
      X_new_scaled <- scale(datos_nuevos[, vars_del_modelo],
                            center = modelo$centro_x, scale = modelo$escala_x)
      M_pred_raw <- as.numeric(predict(modelo$fit_principal, newdata = X_new_scaled))
    } else if (grepl("Ridge|Lasso|ElasticNet", nombre_alg)) {
      M_pred_raw <- as.numeric(predict(modelo$fit_principal,
                                       newx = as.matrix(datos_nuevos[, vars_del_modelo]),
                                       s = "lambda.min"))
    } else if (grepl("Random_Forest", nombre_alg)) {
      M_pred_raw <- predict(modelo$fit_principal, data = datos_nuevos[, c("Y", vars_del_modelo)])$predictions
    } else {
      M_pred_raw <- as.numeric(predict(modelo$fit_principal, newdata = datos_nuevos[, c("Y", vars_del_modelo)]))
    }
    
    M_pred <- (M_pred_raw * modelo$rango_y) + modelo$min_y
    malos  <- is.na(M_pred) | is.infinite(M_pred)
    if (any(malos)) M_pred[malos] <- mean(modelo$modelo_loess$x, na.rm = TRUE)
    
    df_meta[[paste0("M", i)]] <- M_pred
    
    # 2. Exact Scenario 2: Oracle LOESS error-scale proxy
    var_oracle <- predict_loess_robusto(modelo$modelo_loess, M_pred)
    df_meta[[paste0("sd_m", i)]] <- pmax(sqrt(var_oracle), 0.02)
  }
  
  return(df_meta)
}

repartir_tiers_p <- function(p_total) {
  p_total <- max(4L, min(12L, as.integer(p_total)))
  mapa <- list(
    "4"  = c(2L, 1L, 1L),
    "5"  = c(2L, 2L, 1L),
    "6"  = c(3L, 2L, 1L),
    "7"  = c(3L, 2L, 2L),
    "8"  = c(3L, 3L, 2L),
    "9"  = c(4L, 3L, 2L),
    "10" = c(4L, 3L, 3L),
    "11" = c(5L, 3L, 3L),
    "12" = c(5L, 4L, 3L)
  )
  mapa[[as.character(p_total)]]
}

# ==============================================================================
# 2. MINIMAL TIE-BREAKING HELPER FOR LOESS & EXACT SCORING METRICS
# ==============================================================================
# Only adds micro-jitter (1e-5) on a temporary calibration copy when a predictor
# has massive exact ties (e.g., CART trees) to prevent stats::loess KD-tree NaN.
# Never alters the actual M columns in df_train or df_test.
preparar_df_loess_ties <- function(df_cal, m_cols) {
  df_out <- df_cal
  for (mc in m_cols) {
    x <- df_out[[mc]]
    if (length(unique(x)) < length(x) * 0.95) {
      escala_x <- sd(x, na.rm = TRUE)
      if (is.na(escala_x) || escala_x == 0) escala_x <- 1
      df_out[[mc]] <- x + rnorm(length(x), mean = 0, sd = escala_x * 1e-5)
    }
  }
  df_out
}

# Exact metrics matching evaluate_predictions() from the manuscript:
# - For bmetapred: uses raw posterior predictive samples matrix via scoringRules::crps_sample()
# - For base models with s_ik: uses exact Normal CRPS via scoringRules::crps_norm()
calcular_metricas_exactas <- function(y_true, mu_pred, lower_ci, upper_ci, alpha = 0.05,
                                      raw_samples = NULL, sd_norm = NULL) {
  ok <- !is.na(y_true) & !is.na(mu_pred)
  y_ok  <- y_true[ok]
  mu_ok <- mu_pred[ok]
  
  err_i  <- y_ok - mu_ok
  ss_res <- sum(err_i^2)
  ss_tot <- sum((y_ok - mean(y_ok))^2)
  
  width_i <- upper_ci[ok] - lower_ci[ok]
  pen_low <- (2 / alpha) * (lower_ci[ok] - y_ok) * as.numeric(y_ok < lower_ci[ok])
  pen_up  <- (2 / alpha) * (y_ok - upper_ci[ok]) * as.numeric(y_ok > upper_ci[ok])
  wis_i   <- width_i + pen_low + pen_up
  cov_i   <- as.numeric(y_ok >= lower_ci[ok] & y_ok <= upper_ci[ok])
  
  crps_val <- NA_real_
  if (!is.null(raw_samples)) {
    if (requireNamespace("scoringRules", quietly = TRUE)) {
      # raw_samples is [n_draws x n_obs]; scoringRules::crps_sample expects [n_obs x n_draws]
      mat_obs_rows <- t(raw_samples[, ok, drop = FALSE])
      crps_val <- mean(scoringRules::crps_sample(y = y_ok, dat = mat_obs_rows), na.rm = TRUE)
    }
  } else if (!is.null(sd_norm)) {
    sd_ok <- pmax(sd_norm[ok], 1e-6)
    if (requireNamespace("scoringRules", quietly = TRUE)) {
      crps_val <- mean(scoringRules::crps_norm(y = y_ok, mean = mu_ok, sd = sd_ok), na.rm = TRUE)
    } else {
      z <- (y_ok - mu_ok) / sd_ok
      crps_val <- mean(sd_ok * (z * (2 * pnorm(z) - 1) + 2 * dnorm(z) - 1 / sqrt(pi)), na.rm = TRUE)
    }
  }
  
  list(
    MWIS       = mean(wis_i, na.rm = TRUE),
    CRPS       = crps_val,
    Cov_Pct    = mean(cov_i, na.rm = TRUE) * 100,
    Mean_Width = mean(width_i, na.rm = TRUE),
    RMSE       = sqrt(mean(err_i^2, na.rm = TRUE)),
    MAE        = mean(abs(err_i), na.rm = TRUE),
    Mean_Bias  = mean(err_i, na.rm = TRUE),
    R2         = if (ss_tot > 0) 1 - (ss_res / ss_tot) else NA_real_
  )
}

# ==============================================================================
# 3. USER INTERFACE (UI)
# ==============================================================================
ui <- page_navbar(
  window_title = "bmetapred: Bayesian Distributional Meta-Prediction",
  title = div(
    class = "bm-brand-container",
    div(
      class = "bm-brand-badge",
      icon("layer-group")
    ),
    div(
      class = "bm-brand-text",
      div(class = "bm-brand-main", "bmetapred"),
      div(class = "bm-brand-sub", "Bayesian Distributional Meta-Prediction")
    )
  ),
  theme = bs_theme(version = 5, bootswatch = "flatly", primary = "#2C3E50"),
  
  header = tags$head(
    tags$style(HTML("
      /* --- TOP NAVBAR PERFECT VERTICAL CENTERING & HORIZONTAL SPACING --- */
      .navbar {
        padding-top: 0.55rem !important;
        padding-bottom: 0.55rem !important;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.15);
      }
      .navbar > .container-fluid,
      .navbar > .container {
        display: flex !important;
        align-items: center !important;
      }
      .navbar-brand {
        display: flex !important;
        align-items: center !important;
        padding: 0 !important;
        margin-top: 0 !important;
        margin-bottom: 0 !important;
        margin-right: 28px !important;
      }
      .bm-brand-container {
        display: flex;
        align-items: center;
        gap: 12px;
        background-color: rgba(0, 0, 0, 0.22);
        padding: 7px 18px 7px 14px;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.18);
        border-left: 4px solid #18bc9c;
        margin: 0;
      }
      .bm-brand-badge {
        background-color: #18bc9c;
        color: #ffffff;
        width: 36px;
        height: 36px;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.15rem;
        flex-shrink: 0;
      }
      .bm-brand-text {
        display: flex;
        flex-direction: column;
        justify-content: center;
        line-height: 1.15;
      }
      .bm-brand-main {
        font-size: 1.16rem;
        font-weight: 800;
        color: #ffffff;
        letter-spacing: 0.3px;
      }
      .bm-brand-sub {
        font-size: 0.74rem;
        font-weight: 400;
        color: #cfe2f3;
        margin-top: 2px;
      }
      .navbar-collapse {
        display: flex !important;
        align-items: center !important;
      }
      .navbar-nav {
        display: flex !important;
        align-items: center !important;
        gap: 14px !important;
        margin-left: 8px !important;
        padding-left: 28px !important;
        padding-top: 4px !important;
        padding-bottom: 4px !important;
        border-left: 2px solid rgba(255, 255, 255, 0.28);
      }
      .navbar-nav .nav-item {
        display: flex;
        align-items: center;
      }
      .navbar-nav .nav-item .nav-link {
        background-color: rgba(255, 255, 255, 0.10) !important;
        color: rgba(255, 255, 255, 0.90) !important;
        border: 1px solid rgba(255, 255, 255, 0.28) !important;
        border-radius: 8px !important;
        padding: 9px 18px !important;
        font-weight: 600 !important;
        font-size: 0.93rem !important;
        transition: all 0.2s ease-in-out !important;
      }
      .navbar-nav .nav-item .nav-link:hover {
        background-color: rgba(255, 255, 255, 0.22) !important;
        color: #ffffff !important;
        transform: translateY(-1px);
        border-color: rgba(255, 255, 255, 0.55) !important;
      }
      .navbar-nav .nav-item .nav-link.active {
        background-color: #ffffff !important;
        color: #2C3E50 !important;
        border: 2px solid #18bc9c !important;
        box-shadow: 0 3px 10px rgba(0, 0, 0, 0.22) !important;
        font-weight: 700 !important;
      }

      /* --- 4-BAR PROGRESS MONITOR STYLES --- */
      .phase-step {
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 12px 12px;
        background-color: #f8f9fa;
        color: #6c757d;
        transition: all 0.35s ease;
        height: 100%;
      }
      .phase-step.active {
        border: 2px solid #2980b9;
        background-color: #ebf5fb;
        color: #1b4f72;
        box-shadow: 0 3px 10px rgba(41, 128, 185, 0.18);
      }
      .phase-step.done {
        border-color: #18bc9c;
        background-color: #eafaf1;
        color: #117864;
      }
      .milestone-feed {
        background-color: #ffffff;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 12px 16px;
        max-height: 240px;
        overflow-y: auto;
        font-size: 0.92rem;
      }
      .milestone-item {
        padding: 6px 0;
        border-bottom: 1px solid #f1f3f5;
        display: flex;
        justify-content: space-between;
        align-items: center;
      }
      .milestone-item:last-child {
        border-bottom: none;
        font-weight: 600;
        color: #2C3E50;
      }
      /* --- LOGO Y TEXTO FUNDED BY: A LA DERECHA DE LA BARRA --- */
      .navbar-nav {
        flex: 1 !important;
      }
      .bslib-nav-spacer {
        flex-grow: 1 !important;
      }
      .bm-partner-wrapper {
        display: flex;
        align-items: center;
        gap: 10px;
      }
      .bm-funded-label {
        font-size: 0.78rem;
        font-weight: 500;
        color: #cfe2f3;
        white-space: nowrap;
        letter-spacing: 0.2px;
      }
      .bm-partner-logo {
        display: flex;
        align-items: center;
        background-color: rgba(255, 255, 255, 0.92); /* Borra esta línea si tu logo ya es blanco/transparente */
        padding: 4px 10px;
        border-radius: 8px;
      }
      .bm-partner-logo img {
        height: 36px;
        width: auto;
        display: block;
        object-fit: contain;
      }
    ")),
    tags$script(HTML("
      Shiny.addCustomMessageHandler('bm_prog_update', function(msg) {
        var elTime = document.getElementById('bm_elapsed_badge');
        if (elTime && msg.elapsed) elTime.textContent = msg.elapsed;

        if (msg.new_event) {
          var elTitle = document.getElementById('bm_phase_title');
          if (elTitle && msg.etapa) elTitle.textContent = msg.etapa;

          var elDetail = document.getElementById('bm_sub_detail');
          if (elDetail && msg.detalle) elDetail.textContent = msg.detalle;

          for (var i = 0; i <= 3; i++) {
            var st  = document.getElementById('bm_step_' + i);
            var bar = document.getElementById('bm_bar_' + i);
            var lbl = document.getElementById('bm_pct_lbl_' + i);
            var pctVal = (msg.pcts && msg.pcts[i] !== undefined) ? msg.pcts[i] : 0;

            if (lbl) lbl.textContent = pctVal + '%';

            if (st && bar) {
              bar.style.width = pctVal + '%';
              bar.textContent = pctVal > 8 ? (pctVal + '%') : '';

              if (i < msg.phase_idx) {
                st.className = 'phase-step done';
                bar.className = 'progress-bar bg-success fw-bold';
              } else if (i === msg.phase_idx) {
                st.className = 'phase-step active';
                bar.className = 'progress-bar progress-bar-striped progress-bar-animated bg-primary fw-bold';
              } else {
                st.className = 'phase-step';
                bar.className = 'progress-bar bg-secondary';
              }
            }
          }

          var elFeed = document.getElementById('bm_milestones_list');
          if (elFeed && msg.milestones_html) {
            elFeed.innerHTML = msg.milestones_html;
            elFeed.scrollTop = elFeed.scrollHeight;
          }
        }
      });
    "))
  ),
  
  # ----------------------------------------------------------------------------
  # TAB 1 (BLOCK A): DATA, PARTITIONING, AND STANDARDIZATION
  # ----------------------------------------------------------------------------
  nav_panel(
    title = "1. Data & Preprocessing",
    layout_sidebar(
      sidebar = sidebar(
        width = 405,
        title = "Step 1: Source & Structure",
        
        radioButtons(
          inputId = "origen_datos",
          label   = "Select data source:",
          choices = c(
            "Demo 1: Heteroscedastic Simulation"            = "demo_sim",
            "Demo 2: Real Meteorology (srft - ensembleBMA)" = "demo_srft",
            "Upload custom dataset (.csv or .rds)"          = "archivo"
          ),
          selected = "demo_sim"
        ),
        
        conditionalPanel(
          condition = "input.origen_datos == 'demo_sim'",
          numericInput(
            "sim_N_oracle",
            "External Oracle sample size for base learners (N_oracle, max 20,000):",
            value = 2000, min = 500, max = 20000, step = 500
          ),
          helpText("Independent historical sample used to train the 12 ML base learners (and Oracle LOESS curves). Use 20,000 to match the exact paper setup."),
          numericInput("sim_N", "Target study sample size (N):", value = 1500, min = 200, max = 10000, step = 100),
          sliderInput("sim_p", "Number of base models (p):", min = 4, max = 12, value = 6, step = 1),
          uiOutput("ui_info_tiers_sim"),
          sliderInput("sim_h", "Heteroscedasticity intensity (h):", min = 0, max = 3, value = 0.5, step = 0.25)
        ),
        
        conditionalPanel(
          condition = "input.origen_datos == 'archivo'",
          fileInput("file_upload", "Upload main dataset (.csv or .rds):", accept = c(".csv", ".rds")),
          checkboxInput("csv_header", "CSV has header", value = TRUE),
          selectInput("csv_sep", "CSV separator:", choices = c("Comma (,)" = ",", "Semicolon (;)" = ";", "Tab" = "\t"))
        ),
        
        actionButton("btn_cargar", "Load Raw Data", class = "btn-primary w-100 mb-3"),
        hr(),
        
        h6("Variable Assignment"),
        selectInput("col_y", "Response Variable (Y):", choices = NULL),
        selectInput("cols_m", "Base Model Predictions (M1...Mp):", choices = NULL, multiple = TRUE),
        
        hr(),
        h6("Prediction Error Specification (sd_cols)"),
        selectInput(
          inputId = "error_spec_mode",
          label   = "How are base model prediction errors specified?",
          choices = c(
            "1. Calibrate via fit_LOESS (All Base Models)"                 = "all_loess",
            "2. Pre-computed Error Columns in Dataset (All Base Models)"   = "all_precomputed",
            "3. No Prediction Errors — Homoscedastic (Skip fit_LOESS)"     = "none",
            "4. Mixed / Partial Error Specification (Per-Model Assignment)" = "mixed"
          ),
          selected = "all_loess"
        ),
        
        conditionalPanel(
          condition = "input.origen_datos == 'demo_sim'",
          uiOutput("ui_nota_oraculo_sds")
        ),
        
        conditionalPanel(
          condition = "input.error_spec_mode == 'all_precomputed'",
          selectInput("cols_sd", "Error Columns (same order as M1...Mp):", choices = NULL, multiple = TRUE)
        ),
        
        conditionalPanel(
          condition = "input.error_spec_mode == 'mixed'",
          div(
            class = "p-2 border rounded bg-light mb-2",
            style = "max-height: 290px; overflow-y: auto;",
            helpText(class = "mb-2", style = "font-size: 0.80rem;",
                     "Assign an error source for each base model: choose an existing column, estimate via fit_LOESS in Tab 2, or leave without error (Homoscedastic):"),
            uiOutput("ui_mixed_sd_selectors")
          )
        ),
        
        hr(),
        h6("Error Calibration Strategy & Partitioning"),
        
        conditionalPanel(
          condition = "input.error_spec_mode == 'all_loess' || input.error_spec_mode == 'mixed'",
          radioButtons(
            inputId = "loess_strategy",
            label   = "How should fit_LOESS calibrate errors?",
            choices = c(
              "K-Fold Cross-Fitting on Training (Recommended for small N)" = "kfold",
              "Disjoint Pre-training Split (3-way split)"                  = "split",
              "Upload External Calibration Dataset (.csv / .rds)"          = "external"
            ),
            selected = "kfold"
          ),
          conditionalPanel(
            condition = "input.loess_strategy == 'external'",
            fileInput("file_ext_loess", "Upload external calibration file:", accept = c(".csv", ".rds"))
          ),
          conditionalPanel(
            condition = "input.loess_strategy == 'split'",
            numericInput("prop_pretrain", "% Pre-training (for fit_LOESS):", value = 45, min = 10, max = 80, step = 5)
          )
        ),
        
        numericInput("prop_train", "% Training (for fit_bmetapred):", value = 75, min = 10, max = 95, step = 5),
        helpText("The remaining % is automatically reserved for Out-of-Sample Testing."),
        
        numericInput("seed_part", "Random seed (Reproducibility):", value = 2026, min = 1),
        checkboxInput("aplicar_std", "Standardize data in Y: (x - mu_Y) / sd_Y", value = TRUE),
        
        actionButton("btn_procesar", "Apply Partition & Standardize", class = "btn-success w-100 mt-2")
      ),
      
      layout_columns(
        col_widths = c(12, 12),
        card(
          card_header("Partition Summary & Data Status"),
          verbatimTextOutput("texto_resumen_datos")
        ),
        card(
          card_header("Descriptive Statistics & Base Model Performance (Training Set)"),
          DTOutput("tabla_descriptivos")
        )
      )
    )
  ),
  
  # ----------------------------------------------------------------------------
  # TAB 2 (BLOCK B): ERROR CALIBRATION (fit_LOESS)
  # ----------------------------------------------------------------------------
  nav_panel(
    title = "2. Error Calibration (fit_LOESS)",
    layout_sidebar(
      sidebar = sidebar(
        width = 360,
        title = "Step 2: fit_LOESS Parameters",
        uiOutput("ui_info_loess_mode"),
        
        sliderInput("loess_span", "Smoothing parameter (span):", min = 0.15, max = 1.0, value = 0.40, step = 0.05),
        
        conditionalPanel(
          condition = "(input.error_spec_mode == 'all_loess' || input.error_spec_mode == 'mixed') && input.loess_strategy == 'kfold'",
          numericInput("loess_k", "Number of Cross-Fitting Folds (K):", value = 5, min = 2, max = 20, step = 1),
          uiOutput("ui_recomendacion_k")
        ),
        
        actionButton("btn_fit_loess", "Fit fit_LOESS & Project Errors", class = "btn-primary w-100 mt-2"),
        hr(),
        h6("Visualization Options"),
        radioButtons(
          "loess_tipo_grafico",
          "Multipanel plot type:",
          choices = c(
            "Residual Funnel vs Prediction"            = "residuos",
            "Calibration: Observation Y vs Prediction" = "calibracion"
          ),
          selected = "residuos"
        ),
        downloadButton("descargar_fig_loess", "Download Figure (PNG 300 DPI)", class = "btn-outline-secondary w-100 mt-2")
      ),
      
      layout_columns(
        col_widths = c(12, 12),
        card(
          card_header("Projected Heteroscedasticity Diagnostics (s_ik in Training Set)"),
          DTOutput("tabla_cv_sd")
        ),
        card(
          card_header(
            class = "d-flex justify-content-between align-items-center",
            span("Nominal 95% Error Bounds Plot (±1.96 s_ik)"),
            span(class = "text-muted", style = "font-size: 0.82rem;", "Scroll vertically inside the window to inspect all base models")
          ),
          card_body(
            fill = FALSE,
            div(
              style = "max-height: 650px; overflow-y: auto; overflow-x: hidden; width: 100%; padding-right: 6px;",
              plotOutput("plot_loess_grid", width = "100%", height = "auto", fill = FALSE)
            )
          )
        )
      )
    )
  ),
  
  # ----------------------------------------------------------------------------
  # TAB 3 (BLOCK C): META-MODEL FITTING (fit_bmetapred)
  # ----------------------------------------------------------------------------
  nav_panel(
    title = "3. Meta-Model Fitting (fit_bmetapred)",
    layout_sidebar(
      sidebar = sidebar(
        width = 390,
        title = "Step 3: fit_bmetapred Configuration",
        helpText("Configure Bayesian distributional meta-regression, priors, model space search, and Stan MCMC settings."),
        
        selectInput(
          "b_method", 
          "Combination Method (method):", 
          choices = c(
            "Bayesian Stacking" = "stacking",
            "Pseudo-BMA"        = "pbma"
          ),
          selected = "stacking"
        ),
        
        h6("Prior Hyperparameters"),
        layout_columns(
          col_widths = c(6, 6),
          numericInput("b_prisd", "Slope SD (pri_sd):", value = 2.5, min = 0.1, max = 20, step = 0.5),
          numericInput("b_prisd_int", "Intercept SD (pri_sd_int):", value = 5.0, min = 0.5, max = 50, step = 0.5)
        ),
        layout_columns(
          col_widths = c(6, 6),
          numericInput("b_prisd_sigma", "Sigma Slope SD (pri_sd_sigma):", value = 0.5, min = 0.05, max = 10, step = 0.1),
          numericInput("b_prisd_int_sigma", "Sigma Int. SD (pri_sd_int_sigma):", value = 5.0, min = 0.5, max = 50, step = 0.5)
        ),
        
        hr(),
        h6("Model Space Search & Selection"),
        
        uiOutput("ui_search_mode_selector"),
        
        conditionalPanel(
          condition = "input.b_search_mode == 'heuristic'",
          numericInput("b_mc3", "Heuristic search iterations (n_iter_mc3):", value = 1500, min = 200, max = 20000, step = 250)
        ),
        
        conditionalPanel(
          condition = "input.b_method == 'pbma' && input.b_search_mode == 'heuristic'",
          numericInput("b_burnin", "p-BMA Burn-in iterations (0 = default 20%):", value = 0, min = 0, max = 10000, step = 100)
        ),
        
        conditionalPanel(
          condition = "input.b_method == 'stacking'",
          layout_columns(
            col_widths = c(6, 6),
            numericInput("b_minK", "Min selected models (min_K):", value = 10, min = 2, max = 50, step = 1),
            numericInput("b_maxK", "Max selected models (max_K):", value = 50, min = 5, max = 300, step = 5)
          )
        ),
        
        hr(),
        h6("Stan & Parallelism Settings"),
        selectInput("b_backend", "MCMC Backend (backend):", choices = c("rstan", "cmdstanr"), selected = "rstan"),
        numericInput("b_iterstan", "Stan MCMC Iterations (iter_stan):", value = 1500, min = 200, max = 6000, step = 250),
        layout_columns(
          col_widths = c(6, 6),
          numericInput("b_coresstan", "Cores/Stan (cores_stan):", value = 1, min = 1, max = 8, step = 1),
          numericInput("b_workers", "Workers (parallel_workers):", value = 1, min = 1, max = 16, step = 1)
        ),
        
        actionButton("btn_fit_bmetapred", "Run fit_bmetapred (Fit Model Space)", class = "btn-danger w-100 mt-3")
      ),
      
      uiOutput("panel_central_bmetapred")
    )
  ),
  
  # ----------------------------------------------------------------------------
  # TAB 4 (BLOCK D): META-PREDICTION, EVALUATION & REPRODUCIBLE EXPORT
  # ----------------------------------------------------------------------------
  nav_panel(
    title = "4. Meta-Prediction & Results",
    layout_sidebar(
      sidebar = sidebar(
        width = 370,
        title = "Step 4: Meta-Prediction & Export",
        helpText("Generate out-of-sample distributional meta-predictions from posterior predictive draws and compare against base models."),
        
        radioButtons(
          "pred_source",
          "Evaluation dataset:",
          choices = c(
            "Reserved Testing Partition (from Tab 1)"      = "reserved_test",
            "Upload New External Test Dataset (.csv/.rds)" = "external_test"
          ),
          selected = "reserved_test"
        ),
        
        conditionalPanel(
          condition = "input.pred_source == 'external_test'",
          fileInput("file_ext_test", "Upload external test file:", accept = c(".csv", ".rds"))
        ),
        
        numericInput("pred_nsamples", "Posterior predictive samples (n_samples_totales):", value = 2000, min = 200, max = 10000, step = 200),
        sliderInput("pred_level", "Credible Interval Level (1 - alpha):", min = 0.80, max = 0.99, value = 0.95, step = 0.01),
        checkboxInput("pred_destd", "Report results on Original Physical Scale (De-standardize)", value = TRUE),
        
        actionButton("btn_run_predict", "Run Meta-Prediction (predict)", class = "btn-success w-100 mt-2"),
        hr(),
        
        h6("Visualization Controls"),
        checkboxGroupInput(
          "pred_plot_types",
          "Select diagnostic figures to enable:",
          choices = c(
            "1. Ordered Predictive Intervals & Coverage"  = "intervals",
            "2. Base Model PIPs & Selected Model Weights" = "weights",
            "3. Calibration (Observed vs. Predicted)"     = "calibration"
          ),
          selected = c("intervals", "weights", "calibration")
        ),
        conditionalPanel(
          condition = "input.pred_plot_types.indexOf('intervals') !== -1",
          sliderInput("pred_n_plot", "Max test observations shown in interval plot:", min = 30, max = 500, value = 120, step = 10)
        ),
        
        hr(),
        h6("Export Deliverables"),
        downloadButton("dl_pred_csv", "Download Test Predictions (.csv)", class = "btn-outline-primary w-100 mb-2"),
        downloadButton("dl_metrics_csv", "Download Metrics Table (.csv)", class = "btn-outline-primary w-100 mb-2"),
        downloadButton("dl_model_rds", "Download bmetapred Object (.rds)", class = "btn-outline-secondary w-100")
      ),
      
      navset_card_tab(
        id = "tab4_subtabs",
        
        nav_panel(
          title = "4.1 Out-of-Sample Metrics",
          card_body(
            uiOutput("ui_banner_pred"),
            DTOutput("tabla_metricas_test")
          )
        ),
        
        nav_panel(
          title = "4.2 Diagnostic & Comparative Plots",
          card_body(
            uiOutput("ui_contenedor_plots_d")
          )
        ),
        
        nav_panel(
          title = "4.3 Observation-Level Predictions",
          card_body(
            DTOutput("tabla_pred_individuales")
          )
        ),
        
        nav_panel(
          title = "4.4 Reproducible R Script",
          card_body(
            div(
              class = "d-flex justify-content-between align-items-center mb-2",
              p(class = "mb-0 text-muted", "Self-contained R script reproducing your exact workflow, partitioning, priors, and hyperparameters:"),
              downloadButton("dl_r_script", "Download R Script (.R)", class = "btn-primary btn-sm")
            ),
            verbatimTextOutput("codigo_r_reproducible")
          )
        )
      )
    )
  ),
  
  # --- LOGO A LA DERECHA DEL TODO ---
  nav_spacer(),
  nav_item(
    div(
      class = "bm-partner-wrapper",
      span(class = "bm-funded-label", "Funded by:"),
      div(
        class = "bm-partner-logo",
        tags$img(
          src = "Fundación_BBVA_idgg2t__lM_0.png",
          alt = "Funding Institution Logo"
        )
      )
    )
  )
)

# ==============================================================================
# 4. SERVER LOGIC
# ==============================================================================
server <- function(input, output, session) {
  
  estado <- reactiveValues(
    df_raw              = NULL,
    df_pretrain         = NULL,
    df_train            = NULL,
    df_test             = NULL,
    escala_Y            = NULL,
    y_col               = NULL,
    m_cols              = NULL,
    sd_cols             = NULL,
    sd_is_homoscedastic = logical(0),
    sd_ready            = FALSE,
    error_spec_used     = NULL,
    loess_mode_used     = NULL,
    models_need_loess   = character(0),
    n_oracle_used       = NULL,
    loess_obj           = NULL,
    plot_loess          = NULL,
    modelo_bm           = NULL,
    tiempo_bm           = NULL,
    pips_bm             = NULL,
    top_mods_bm         = NULL,
    # Background process state (Block C - 4-bar flicker-free monitor)
    bg_running          = FALSE,
    bg_process          = NULL,
    bg_log_file         = NULL,
    bg_start_time       = NULL,
    bg_file_size        = -1,
    bg_logs             = character(0),
    bg_error_msg        = NULL,
    # Prediction & evaluation state (Block D)
    pred_raw_std        = NULL,
    df_eval_std         = NULL,
    df_pred_final       = NULL,
    df_metrics_comp     = NULL,
    # Single-plot viewer active plot key
    active_plot_d       = "intervals"
  )
  
  output$ui_info_tiers_sim <- renderUI({
    req(input$sim_p)
    tiers <- repartir_tiers_p(input$sim_p)
    helpText(sprintf(
      "Manuscript 3-tier pool for p = %d: %d High-perf. (Top) + %d Intermediate (Mid) + %d Low-perf. (Worst) ML models.",
      sum(tiers), tiers[1], tiers[2], tiers[3]
    ))
  })
  
  output$ui_nota_oraculo_sds <- renderUI({
    n_orac_val <- if (!is.null(input$sim_N_oracle) && !is.na(input$sim_N_oracle)) as.integer(input$sim_N_oracle) else 2000L
    if (identical(input$error_spec_mode, "all_precomputed")) {
      div(
        class = "alert alert-info py-2 px-3 mb-2",
        style = "font-size: 0.84rem;",
        strong("Exact Manuscript Reproduction Mode: "),
        sprintf(
          "By selecting Pre-computed Error Columns in Demo 1, the error scales (sd_m1...sd_mp) are calibrated via LOESS directly on the external Oracle dataset (N_oracle = %s). Step 2 (fit_LOESS in Tab 2) is NOT needed—you can proceed directly to Tab 3 after partitioning.",
          format(n_orac_val, big.mark = ",")
        )
      )
    } else if (identical(input$error_spec_mode, "none")) {
      div(
        class = "alert alert-secondary py-2 px-3 mb-2",
        style = "font-size: 0.82rem;",
        strong("Homoscedastic Mode Active: "),
        "No prediction error scales s_ik will be used (all base models use the homoscedastic Stan template). Step 2 (fit_LOESS) is skipped—proceed directly to Tab 3 after partitioning."
      )
    } else if (identical(input$error_spec_mode, "mixed")) {
      div(
        class = "alert alert-info py-2 px-3 mb-2",
        style = "font-size: 0.82rem;",
        strong("Mixed / Partial Error Mode: "),
        "Configure below which models use Oracle/native sd_ columns, which ones are calibrated via fit_LOESS in Tab 2, and which ones have no error (Homoscedastic)."
      )
    } else {
      div(
        class = "alert alert-light border py-2 px-3 mb-2",
        style = "font-size: 0.82rem;",
        strong("Note for Demo 1: "),
        "Select '2. Pre-computed Error Columns' above if you want LOESS error scales trained directly on the external Oracle dataset (reproducing the exact paper pipeline without needing Step 2)."
      )
    }
  })
  
  output$ui_mixed_sd_selectors <- renderUI({
    req(estado$df_raw, input$cols_m)
    m_cols    <- input$cols_m
    all_cols  <- names(estado$df_raw)
    cand_cols <- setdiff(all_cols, c(input$col_y, m_cols))
    
    opciones_base <- c(
      "[No Error — Homoscedastic]"        = "__NONE__",
      "[Estimate via fit_LOESS in Tab 2]" = "__LOESS__"
    )
    if (length(cand_cols) > 0) {
      names(cand_cols) <- paste0("Column: ", cand_cols)
      opciones_tot <- c(opciones_base, cand_cols)
    } else {
      opciones_tot <- opciones_base
    }
    
    lapply(seq_along(m_cols), function(i) {
      mc <- m_cols[i]
      posibles_match <- c(paste0("sd_", mc), paste0("sd_", tolower(mc)), paste0("SD_", mc))
      hit <- intersect(posibles_match, all_cols)
      sel_def <- if (length(hit) > 0) hit[1] else "__LOESS__"
      
      selectInput(
        inputId  = paste0("mixed_sd_", mc),
        label    = sprintf("Error source for %s:", mc),
        choices  = opciones_tot,
        selected = sel_def,
        width    = "100%"
      )
    })
  })
  
  output$ui_search_mode_selector <- renderUI({
    p_val <- if (!is.null(estado$m_cols) && length(estado$m_cols) > 0) {
      length(estado$m_cols)
    } else if (!is.null(input$cols_m) && length(input$cols_m) > 0) {
      length(input$cols_m)
    } else {
      6L
    }
    
    es_pequeno <- p_val <= 9
    rel_txt    <- if (es_pequeno) "p <= 9" else "p > 9"
    sug_txt    <- if (es_pequeno) "Exhaustive exploration is suggested" else "Heuristic exploration is suggested"
    sel_def    <- if (es_pequeno) "exhaustive" else "heuristic"
    
    n_comb_fmt <- format(2^p_val, big.mark = ",", scientific = FALSE)
    choices_vec <- c("exhaustive", "heuristic")
    names(choices_vec) <- c(
      sprintf("Exhaustive Search (2^%d = %s models)", p_val, n_comb_fmt),
      "Heuristic Search"
    )
    
    tagList(
      div(
        class = "alert alert-light border py-2 px-3 mb-2",
        style = "font-size: 0.86rem;",
        sprintf("The number of base models is p = %d (%s). %s.", p_val, rel_txt, sug_txt)
      ),
      radioButtons(
        inputId  = "b_search_mode",
        label    = "How would you like to explore the model space?",
        choices  = choices_vec,
        selected = sel_def
      )
    )
  })
  
  calcular_pmax_interno <- function(p_val, modo_sel) {
    modo <- if (!is.null(modo_sel)) modo_sel else if (p_val <= 9) "exhaustive" else "heuristic"
    if (modo == "exhaustive") {
      if (p_val <= 9) return(9L) else return(as.integer(p_val + 1L))
    } else {
      if (p_val > 9) return(9L) else return(as.integer(max(1L, p_val - 1L)))
    }
  }
  
  observeEvent(input$loess_strategy, {
    if (input$loess_strategy %in% c("kfold", "external")) {
      if (input$prop_train < 60) updateNumericInput(session, "prop_train", value = 75)
    } else if (input$loess_strategy == "split") {
      if (input$prop_pretrain + input$prop_train >= 95) {
        updateNumericInput(session, "prop_pretrain", value = 45)
        updateNumericInput(session, "prop_train", value = 35)
      }
    }
  })
  
  # ----------------------------------------------------------------------------
  # EVENT 1: LOAD RAW DATA (EXACT MANUSCRIPT DEMO 1 & FULL DEMO 2 srft)
  # ----------------------------------------------------------------------------
  observeEvent(input$btn_cargar, {
    if (input$origen_datos == "demo_sim") {
      n_orac <- if (!is.null(input$sim_N_oracle) && !is.na(input$sim_N_oracle)) {
        min(20000L, max(500L, as.integer(input$sim_N_oracle)))
      } else {
        2000L
      }
      if (!is.null(input$sim_N_oracle) && !is.na(input$sim_N_oracle) && input$sim_N_oracle > 20000) {
        updateNumericInput(session, "sim_N_oracle", value = 20000)
        showNotification("Oracle dataset size capped at the maximum of 20,000 observations.", type = "warning")
      }
      
      withProgress(
        message = sprintf("Simulating M-open DGP & training 12 ML base learners on Oracle set (N_oracle = %s)...", format(n_orac, big.mark = ",")),
        value = 0.15,
        {
          set.seed(input$seed_part)
          tiers_vec     <- repartir_tiers_p(input$sim_p)
          datos_oraculo <- generar_dataset_mopen(N = n_orac, z = 50, h = input$sim_h)
          incProgress(0.45, detail = "Training SVM, NNet, Random Forest, CART, Ridge & LM learners + Oracle LOESS...")
          modelos_pre   <- entrenar_modelos_base(datos_oraculo)
          incProgress(0.25, detail = "Generating out-of-sample study predictions (M1..Mp) & Oracle LOESS proxies...")
          datos_estudio <- generar_dataset_mopen(N = input$sim_N, z = 50, h = input$sim_h)
          estado$df_raw <- generar_X(datos_estudio, modelos_pre, p = tiers_vec)
          estado$n_oracle_used <- n_orac
          incProgress(0.15)
        }
      )
      showNotification(
        sprintf("Manuscript simulation generated (N_oracle = %s, study N = %d, p = %d, h = %.2f).",
                format(n_orac, big.mark = ","), input$sim_N, input$sim_p, input$sim_h),
        type = "message"
      )
      
    } else if (input$origen_datos == "demo_srft") {
      if (!requireNamespace("ensembleBMA", quietly = TRUE)) {
        showNotification("The 'ensembleBMA' package is not installed.", type = "error")
        return()
      }
      data("srft", package = "ensembleBMA", envir = environment())
      modelos_orig   <- c("CMCG", "ETA", "GASP", "GFS", "JMA", "NGPS", "TCWB", "UKMO")
      cols_nec       <- c("observation", modelos_orig)
      df_temp        <- srft[complete.cases(srft[, cols_nec]), cols_nec]
      names(df_temp) <- c("Y", paste0("M", 1:8))
      estado$df_raw  <- df_temp
      estado$n_oracle_used <- NULL
      showNotification(sprintf("Full meteorological dataset 'srft' loaded (%s complete observations).", format(nrow(df_temp), big.mark = ",")), type = "message")
      
    } else if (input$origen_datos == "archivo") {
      req(input$file_upload)
      ext <- tools::file_ext(input$file_upload$name)
      if (tolower(ext) == "csv") {
        df_subido <- read.csv(
          input$file_upload$datapath,
          header           = input$csv_header,
          sep              = input$csv_sep,
          stringsAsFactors = FALSE,
          check.names      = FALSE
        )
      } else if (tolower(ext) == "rds") {
        df_subido <- as.data.frame(readRDS(input$file_upload$datapath))
      } else {
        showNotification("Unsupported format.", type = "error")
        return()
      }
      
      # Sanitize column names so R formulas in bmetapred never fail
      nombres_limpios <- gsub("[^[:alnum:]_]", "_", trimws(names(df_subido)))
      nombres_limpios <- gsub("_+", "_", nombres_limpios)
      nombres_limpios <- gsub("^_|_$", "", nombres_limpios)
      nombres_limpios <- make.names(nombres_limpios, unique = TRUE)
      names(df_subido) <- nombres_limpios
      
      estado$df_raw <- df_subido
      estado$n_oracle_used <- NULL
      showNotification("User file loaded successfully.", type = "message")
    }
    
    cols_disponibles <- names(estado$df_raw)
    es_num   <- sapply(estado$df_raw, function(col) is.numeric(col) || is.integer(col))
    cols_num <- cols_disponibles[es_num]
    if (length(cols_num) == 0) cols_num <- cols_disponibles
    
    col_y_def <- if ("Y" %in% cols_num) "Y" else if ("y" %in% cols_num) "y" else cols_num[1]
    updateSelectInput(session, "col_y", choices = cols_disponibles, selected = col_y_def)
    
    posibles_m  <- grep("^[Mm][0-9]+$", cols_num, value = TRUE)
    posibles_sd <- grep("^sd_", cols_num, value = TRUE, ignore.case = TRUE)
    if (length(posibles_m) == 0) {
      posibles_m <- setdiff(cols_num, c(col_y_def, posibles_sd))
    }
    updateSelectInput(session, "cols_m", choices = cols_disponibles, selected = posibles_m)
    updateSelectInput(session, "cols_sd", choices = cols_disponibles, selected = posibles_sd)
  })
  
  # ----------------------------------------------------------------------------
  # EVENT 2: RANDOM PARTITIONING, STANDARDIZATION & 4-MODE ERROR SPECIFICATION
  # ----------------------------------------------------------------------------
  observeEvent(input$btn_procesar, {
    req(estado$df_raw, input$col_y, input$cols_m)
    
    df       <- estado$df_raw
    y_col    <- input$col_y
    m_cols   <- input$cols_m
    err_mode <- input$error_spec_mode
    
    sd_cols_plan      <- character(length(m_cols))
    is_homo_vec       <- rep(FALSE, length(m_cols))
    models_need_loess <- character(0)
    cols_sd_exist     <- character(0)
    
    if (err_mode == "all_loess") {
      models_need_loess <- m_cols
      sd_cols_plan      <- paste0("sd_", m_cols)
      
    } else if (err_mode == "all_precomputed") {
      if (length(input$cols_sd) != length(m_cols)) {
        showNotification(
          sprintf("You selected %d base models but %d error columns. For partial error columns, choose '4. Mixed / Partial Error Specification'.",
                  length(m_cols), length(input$cols_sd)),
          type = "error", duration = 10
        )
        return()
      }
      sd_cols_plan  <- input$cols_sd
      cols_sd_exist <- input$cols_sd
      
    } else if (err_mode == "none") {
      sd_cols_plan      <- paste0("sd_none_", m_cols)
      is_homo_vec       <- rep(TRUE, length(m_cols))
      models_need_loess <- character(0)
      
    } else if (err_mode == "mixed") {
      for (i in seq_along(m_cols)) {
        mc  <- m_cols[i]
        sel <- input[[paste0("mixed_sd_", mc)]]
        if (is.null(sel) || sel == "__NONE__") {
          sd_cols_plan[i] <- paste0("sd_none_", mc)
          is_homo_vec[i]  <- TRUE
        } else if (sel == "__LOESS__") {
          models_need_loess <- c(models_need_loess, mc)
          sd_cols_plan[i]   <- paste0("sd_", mc)
        } else {
          sd_cols_plan[i] <- sel
          cols_sd_exist   <- c(cols_sd_exist, sel)
        }
      }
    }
    
    cols_usar <- unique(c(y_col, m_cols, cols_sd_exist))
    
    for (cn in cols_usar) {
      if (!is.numeric(df[[cn]])) {
        df[[cn]] <- suppressWarnings(as.numeric(gsub(",", ".", as.character(df[[cn]]))))
      }
      if (sum(!is.na(df[[cn]])) < 20) {
        showNotification(
          sprintf("Column '%s' is not numeric (or contains mostly NAs/text like dates). Please remove it from Base Model Predictions.", cn),
          type = "error", duration = 10
        )
        return()
      }
    }
    
    df <- df[complete.cases(df[, cols_usar, drop = FALSE]), cols_usar, drop = FALSE]
    N_total <- nrow(df)
    
    set.seed(input$seed_part)
    idx_perm <- sample(seq_len(N_total), size = N_total, replace = FALSE)
    
    needs_loess <- length(models_need_loess) > 0
    strat       <- if (!needs_loess) {
      if (err_mode == "none" || all(is_homo_vec)) "none" else "precomputed"
    } else {
      input$loess_strategy
    }
    
    if (strat == "split") {
      if (input$prop_pretrain + input$prop_train >= 95) {
        showNotification("Sum of % Pre-training and % Training must be below 95% to leave data for Testing.", type = "error")
        return()
      }
      n_pre      <- round(N_total * (input$prop_pretrain / 100))
      n_tr       <- round(N_total * (input$prop_train / 100))
      idx_pre    <- idx_perm[1:n_pre]
      idx_tr     <- idx_perm[(n_pre + 1):(n_pre + n_tr)]
      idx_te     <- idx_perm[(n_pre + n_tr + 1):N_total]
      df_pre_raw <- df[idx_pre, , drop = FALSE]
      df_tr_raw  <- df[idx_tr, , drop = FALSE]
      df_te_raw  <- df[idx_te, , drop = FALSE]
      
    } else if (strat == "external") {
      if (is.null(input$file_ext_loess)) {
        showNotification("Please upload the External Calibration file for fit_LOESS.", type = "error")
        return()
      }
      ext <- tools::file_ext(input$file_ext_loess$name)
      df_ext <- if (tolower(ext) == "csv") {
        read.csv(input$file_ext_loess$datapath, header = input$csv_header, sep = input$csv_sep, stringsAsFactors = FALSE, check.names = FALSE)
      } else {
        as.data.frame(readRDS(input$file_ext_loess$datapath))
      }
      nombres_ext <- gsub("[^[:alnum:]_]", "_", trimws(names(df_ext)))
      nombres_ext <- gsub("_+", "_", nombres_ext)
      nombres_ext <- gsub("^_|_$", "", nombres_ext)
      names(df_ext) <- make.names(nombres_ext, unique = TRUE)
      
      if (!all(c(y_col, models_need_loess) %in% names(df_ext))) {
        showNotification("External calibration file must contain the response (Y) and all predictor columns requiring fit_LOESS.", type = "error")
        return()
      }
      df_pre_raw <- df_ext[complete.cases(df_ext[, c(y_col, models_need_loess)]), , drop = FALSE]
      n_tr       <- round(N_total * (input$prop_train / 100))
      df_tr_raw  <- df[idx_perm[1:n_tr], , drop = FALSE]
      df_te_raw  <- df[idx_perm[(n_tr + 1):N_total], , drop = FALSE]
      
    } else {
      if (input$prop_train >= 95) {
        showNotification("% Training must be below 95% to leave observations for Out-of-Sample Testing.", type = "error")
        return()
      }
      df_pre_raw <- NULL
      n_tr       <- round(N_total * (input$prop_train / 100))
      df_tr_raw  <- df[idx_perm[1:n_tr], , drop = FALSE]
      df_te_raw  <- df[idx_perm[(n_tr + 1):N_total], , drop = FALSE]
      
      k_rec <- if (nrow(df_tr_raw) < 200) 10 else 5
      updateNumericInput(session, "loess_k", value = k_rec)
    }
    
    df_ancla <- if (!is.null(df_pre_raw) && nrow(df_pre_raw) > 10) df_pre_raw else df_tr_raw
    mu_Y <- mean(df_ancla[[y_col]])
    sd_Y <- sd(df_ancla[[y_col]])
    
    if (input$aplicar_std) {
      estandarizar <- function(d) {
        if (is.null(d)) return(NULL)
        d_out <- d
        for (c_name in intersect(c(y_col, m_cols), names(d_out))) {
          d_out[[c_name]] <- (d[[c_name]] - mu_Y) / sd_Y
        }
        for (s_name in intersect(cols_sd_exist, names(d_out))) {
          d_out[[s_name]] <- d[[s_name]] / sd_Y
        }
        return(d_out)
      }
      estado$df_pretrain <- estandarizar(df_pre_raw)
      estado$df_train    <- estandarizar(df_tr_raw)
      estado$df_test     <- estandarizar(df_te_raw)
      estado$escala_Y    <- list(mu_Y = mu_Y, sd_Y = sd_Y, estandarizado = TRUE)
    } else {
      estado$df_pretrain <- df_pre_raw
      estado$df_train    <- df_tr_raw
      estado$df_test     <- df_te_raw
      estado$escala_Y    <- list(mu_Y = 0, sd_Y = 1, estandarizado = FALSE)
    }
    
    # Zero-valued columns trigger bmetapred Phase 0.1 audit_and_clean_errors -> is_clean = TRUE (sigma ~ 1)
    if (any(is_homo_vec)) {
      cols_zero <- sd_cols_plan[is_homo_vec]
      for (cz in cols_zero) {
        if (!is.null(estado$df_pretrain)) estado$df_pretrain[[cz]] <- 0.0
        estado$df_train[[cz]] <- 0.0
        estado$df_test[[cz]]  <- 0.0
      }
    }
    
    estado$y_col               <- y_col
    estado$m_cols              <- m_cols
    estado$sd_cols             <- sd_cols_plan
    estado$sd_is_homoscedastic <- is_homo_vec
    estado$models_need_loess   <- models_need_loess
    estado$sd_ready            <- !needs_loess
    estado$error_spec_used     <- err_mode
    estado$loess_mode_used     <- strat
    estado$loess_obj           <- NULL
    estado$modelo_bm           <- NULL
    estado$pred_raw_std        <- NULL
    estado$df_pred_final       <- NULL
    estado$df_metrics_comp     <- NULL
    estado$bg_error_msg        <- NULL
    
    if (!needs_loess) {
      showNotification("Partitioning completed! Step 2 (fit_LOESS) is not required—you can proceed directly to Tab 3.", type = "message", duration = 6)
    } else {
      showNotification(sprintf("Partitioning completed! Proceed to Tab 2 to calibrate fit_LOESS for %d model(s).", length(models_need_loess)), type = "message")
    }
  })
  
  output$texto_resumen_datos <- renderPrint({
    validate(need(!is.null(estado$df_train), "1. Click 'Load Raw Data' and then 'Apply Partition & Standardize'."))
    if (!is.null(estado$n_oracle_used) && input$origen_datos == "demo_sim") {
      cat(sprintf("Demo 1 External Oracle Dataset: N_oracle = %s observations (used to train base ML learners)\n",
                  format(estado$n_oracle_used, big.mark = ",")))
    }
    cat(sprintf("Error Specification Mode: %s | Calibration Strategy: %s\n",
                toupper(estado$error_spec_used), toupper(estado$loess_mode_used)))
    cat(sprintf("  - Pre-training (fit_LOESS): N_pre   = %d rows %s\n", 
                if (!is.null(estado$df_pretrain)) nrow(estado$df_pretrain) else 0,
                if (estado$loess_mode_used == "kfold") {
                  "(0 rows lost: using Out-of-Fold Cross-Fitting on Training)"
                } else if (estado$loess_mode_used == "none") {
                  "(Homoscedastic mode: no error calibration needed)"
                } else if (estado$loess_mode_used == "precomputed") {
                  "(Pre-computed sd_ columns ready — Step 2 fit_LOESS not needed)"
                } else {
                  ""
                }))
    cat(sprintf("  - Training (fit_bmetapred): N_train = %d rows\n", nrow(estado$df_train)))
    cat(sprintf("  - Testing (Out-of-sample):  N_test  = %d rows\n", nrow(estado$df_test)))
    cat(sprintf("\nResponse: %s | Base Models (p = %d): %s\n", 
                estado$y_col, length(estado$m_cols), paste(estado$m_cols, collapse = ", ")))
    
    etiquetas_sd <- ifelse(estado$sd_is_homoscedastic, "None (Homoscedastic)", estado$sd_cols)
    map_str <- paste(sprintf("%s -> %s", estado$m_cols, etiquetas_sd), collapse = "  |  ")
    cat(sprintf("Per-Model Error Mapping: %s\n", map_str))
    
    if (isTRUE(estado$sd_ready)) {
      cat("Status: READY FOR TAB 3 (fit_bmetapred). Step 2 is not needed.\n")
    } else {
      cat(sprintf("Status: Pending Step 2 (fit_LOESS) for: %s\n", paste(estado$models_need_loess, collapse = ", ")))
    }
  })
  
  output$tabla_descriptivos <- renderDT({
    req(estado$df_train, estado$y_col, estado$m_cols)
    df <- estado$df_train
    y_vec <- df[[estado$y_col]]
    todas_vars <- c(estado$y_col, estado$m_cols)
    
    tabla <- do.call(rbind, lapply(seq_along(todas_vars), function(idx) {
      v <- todas_vars[idx]
      x <- df[[v]]
      es_mod <- v != estado$y_col
      err <- if (es_mod) (y_vec - x) else rep(NA, length(x))
      sd_assigned <- if (es_mod) {
        if (isTRUE(estado$sd_is_homoscedastic[idx - 1])) "None (Homoscedastic)" else estado$sd_cols[idx - 1]
      } else {
        "—"
      }
      data.frame(
        Variable     = v,
        Role         = if (es_mod) "Base Model" else "Response (Target)",
        Error_Scale  = sd_assigned,
        Mean         = round(mean(x, na.rm = TRUE), 3),
        SD           = round(sd(x, na.rm = TRUE), 3),
        Min          = round(min(x, na.rm = TRUE), 3),
        Median       = round(median(x, na.rm = TRUE), 3),
        Max          = round(max(x, na.rm = TRUE), 3),
        Mean_Bias    = if (es_mod) round(mean(err, na.rm = TRUE), 4) else NA,
        Base_RMSE    = if (es_mod) round(sqrt(mean(err^2, na.rm = TRUE)), 4) else NA,
        Base_MAE     = if (es_mod) round(mean(abs(err), na.rm = TRUE), 4) else NA
      )
    }))
    datatable(tabla, options = list(pageLength = 15, dom = "t"), rownames = FALSE)
  })
  
  # ----------------------------------------------------------------------------
  # EVENT 3 (BLOCK B): NATIVE bmetapred::fit_LOESS & predict.bmetapred_loess
  # ----------------------------------------------------------------------------
  output$ui_info_loess_mode <- renderUI({
    req(estado$loess_mode_used)
    n_need <- length(estado$models_need_loess)
    
    if (estado$loess_mode_used == "none") {
      div(class = "alert alert-secondary py-2 px-3 mb-2", style = "font-size: 0.88rem;",
          strong("No Prediction Errors Mode Active (Homoscedastic): "),
          "All base models are configured without prediction error scales. Step 2 (fit_LOESS) is skipped—proceed directly to Tab 3.")
    } else if (estado$loess_mode_used == "precomputed") {
      div(class = "alert alert-success py-2 px-3 mb-2", style = "font-size: 0.88rem;",
          strong("Pre-computed Error Scales Ready: "),
          "No models require fit_LOESS estimation. You can inspect the diagnostic plots on the right or proceed directly to Tab 3.")
    } else if (estado$loess_mode_used == "kfold") {
      div(class = "alert alert-info py-2 px-3 mb-2", style = "font-size: 0.88rem;",
          strong(sprintf("Active Mode: K-Fold Cross-Fitting (%d model(s) to calibrate).", n_need)),
          sprintf(" Target models for fit_LOESS: %s.", paste(estado$models_need_loess, collapse = ", ")))
    } else if (estado$loess_mode_used == "split") {
      div(class = "alert alert-secondary py-2 px-3 mb-2", style = "font-size: 0.88rem;",
          strong(sprintf("Active Mode: Disjoint Pre-training Split (%d model(s) to calibrate).", n_need)),
          sprintf(" Using N_pre = %d dedicated observations.", if (!is.null(estado$df_pretrain)) nrow(estado$df_pretrain) else 0))
    } else {
      div(class = "alert alert-secondary py-2 px-3 mb-2", style = "font-size: 0.88rem;",
          strong(sprintf("Active Mode: External Calibration Dataset (%d model(s) to calibrate).", n_need)))
    }
  })
  
  output$ui_recomendacion_k <- renderUI({
    req(estado$df_train)
    n_tr <- nrow(estado$df_train)
    if (n_tr < 200) {
      helpText(sprintf("Sample size detected: N_train = %d (under 200). Recommended: K = 10 (retains 90%% of data per fold for tail stability).", n_tr))
    } else {
      helpText(sprintf("Sample size detected: N_train = %d (200 or more). Recommended: K = 5 (optimal bias-variance balance; 80%% data per fold).", n_tr))
    }
  })
  
  observeEvent(input$btn_fit_loess, {
    req(estado$df_train, estado$y_col, estado$m_cols)
    strat       <- estado$loess_mode_used
    mods_target <- estado$models_need_loess
    
    if (strat == "none" || length(mods_target) == 0) {
      showNotification("No base models require fit_LOESS calibration under your current Error Specification. Proceed directly to Tab 3!", type = "message")
      return()
    }
    
    if (strat == "kfold") {
      K <- as.integer(input$loess_k)
      if (is.na(K) || K < 2 || K > nrow(estado$df_train) / 10) {
        showNotification("Please choose a valid number of folds K (with at least 10 observations per fold).", type = "error")
        return()
      }
      
      err_flag <- FALSE
      withProgress(message = sprintf("Running %d-Fold Cross-Fitting for %d model(s)...", K, length(mods_target)), value = 0.1, {
        df_tr_sub <- estado$df_train[, c(estado$y_col, mods_target), drop = FALSE]
        df_te_sub <- estado$df_test[, c(estado$y_col, mods_target), drop = FALSE]
        N_tr      <- nrow(df_tr_sub)
        
        set.seed(input$seed_part)
        fold_id <- sample(rep(seq_len(K), length.out = N_tr))
        
        # Micro-jitter only on temporary calibration copy if ties exist (e.g. CART trees)
        df_tr_cal_safe <- preparar_df_loess_ties(df_tr_sub, mods_target)
        sd_names_targ  <- paste0("sd_", mods_target)
        df_tr_sds_tmp  <- estado$df_train
        for (sc in sd_names_targ) df_tr_sds_tmp[[sc]] <- NA_real_
        
        tryCatch({
          for (k in seq_len(K)) {
            idx_train_k <- which(fold_id != k)
            idx_val_k   <- which(fold_id == k)
            
            mod_k <- bmetapred::fit_LOESS(
              data       = df_tr_cal_safe[idx_train_k, , drop = FALSE],
              y          = estado$y_col,
              predictors = mods_target,
              span       = input$loess_span
            )
            pred_val_k <- predict(mod_k, newdata = df_tr_sub[idx_val_k, , drop = FALSE], prefix = "sd_")
            
            for (sc in sd_names_targ) {
              df_tr_sds_tmp[idx_val_k, sc] <- pred_val_k[[sc]]
            }
            incProgress(0.7 / K)
          }
          
          loess_full <- bmetapred::fit_LOESS(
            data       = df_tr_cal_safe,
            y          = estado$y_col,
            predictors = mods_target,
            span       = input$loess_span
          )
          df_te_out <- predict(loess_full, newdata = df_te_sub, prefix = "sd_")
          
          estado$df_train <- df_tr_sds_tmp
          for (sc in sd_names_targ) {
            estado$df_test[[sc]] <- df_te_out[[sc]]
          }
          estado$loess_obj <- loess_full
          estado$sd_ready  <- TRUE
        }, error = function(e) {
          err_flag <<- TRUE
          showNotification(paste("Error in bmetapred::fit_LOESS / predict:", e$message), type = "error", duration = 12)
        })
        incProgress(0.2)
      })
      if (!err_flag) {
        showNotification(sprintf("%d-Fold Cross-Fitting completed for %s!", K, paste(mods_target, collapse = ", ")), type = "message")
      }
      
    } else {
      if (is.null(estado$df_pretrain) || nrow(estado$df_pretrain) < 20) {
        showNotification("At least 20 Pre-training observations are required for split/external fit_LOESS.", type = "error")
        return()
      }
      
      err_flag <- FALSE
      withProgress(message = "Fitting bmetapred::fit_LOESS and projecting error scales s_ik...", value = 0.4, {
        df_pre_sub <- estado$df_pretrain[, c(estado$y_col, mods_target), drop = FALSE]
        df_tr_sub  <- estado$df_train[, c(estado$y_col, mods_target), drop = FALSE]
        df_te_sub  <- estado$df_test[, c(estado$y_col, mods_target), drop = FALSE]
        
        df_pre_cal_safe <- preparar_df_loess_ties(df_pre_sub, mods_target)
        sd_names_targ   <- paste0("sd_", mods_target)
        
        tryCatch({
          loess_mod <- bmetapred::fit_LOESS(
            data       = df_pre_cal_safe,
            y          = estado$y_col,
            predictors = mods_target,
            span       = input$loess_span
          )
          incProgress(0.4)
          
          df_pre_out <- predict(loess_mod, newdata = df_pre_sub, prefix = "sd_")
          df_tr_out  <- predict(loess_mod, newdata = df_tr_sub,  prefix = "sd_")
          df_te_out  <- predict(loess_mod, newdata = df_te_sub,  prefix = "sd_")
          
          for (sc in sd_names_targ) {
            estado$df_pretrain[[sc]] <- df_pre_out[[sc]]
            estado$df_train[[sc]]    <- df_tr_out[[sc]]
            estado$df_test[[sc]]     <- df_te_out[[sc]]
          }
          
          estado$loess_obj <- loess_mod
          estado$sd_ready  <- TRUE
        }, error = function(e) {
          err_flag <<- TRUE
          showNotification(paste("Error in bmetapred::fit_LOESS / predict:", e$message), type = "error", duration = 12)
        })
        incProgress(0.2)
      })
      if (!err_flag) {
        showNotification("fit_LOESS successfully fitted and errors projected.", type = "message")
      }
    }
  })
  
  obtener_indices_con_sd <- reactive({
    req(estado$df_train, estado$m_cols)
    if (is.null(estado$sd_cols)) return(integer(0))
    which(!estado$sd_is_homoscedastic & (estado$sd_cols %in% names(estado$df_train)))
  })
  
  output$tabla_cv_sd <- renderDT({
    req(estado$df_train, estado$m_cols)
    if (identical(estado$loess_mode_used, "none")) {
      validate(need(FALSE, "No Prediction Errors Mode (Homoscedastic) is active: no s_ik columns are used. Proceed directly to Tab 3."))
    }
    idx_valid <- obtener_indices_con_sd()
    validate(need(length(idx_valid) > 0, "Click 'Fit fit_LOESS & Project Errors' on the left sidebar to generate s_ik diagnostics."))
    
    tabla_cv <- do.call(rbind, lapply(seq_along(estado$m_cols), function(i) {
      col_m  <- estado$m_cols[i]
      col_sd <- estado$sd_cols[i]
      if (isTRUE(estado$sd_is_homoscedastic[i])) {
        return(data.frame(
          Model            = col_m,
          Error_Column     = "None (Homoscedastic)",
          SD_Min           = NA,
          SD_Mean          = NA,
          SD_Max           = NA,
          Max_Min_Ratio    = NA,
          CV_SD_Percentage = NA,
          Empirical_Cov_95 = NA
        ))
      }
      if (!(col_sd %in% names(estado$df_train))) {
        return(data.frame(
          Model            = col_m,
          Error_Column     = paste0(col_sd, " (Pending fit_LOESS)"),
          SD_Min           = NA,
          SD_Mean          = NA,
          SD_Max           = NA,
          Max_Min_Ratio    = NA,
          CV_SD_Percentage = NA,
          Empirical_Cov_95 = NA
        ))
      }
      sds    <- estado$df_train[[col_sd]]
      y_val  <- estado$df_train[[estado$y_col]]
      pred   <- estado$df_train[[col_m]]
      cov_95 <- mean(y_val >= (pred - 1.96 * sds) & y_val <= (pred + 1.96 * sds), na.rm = TRUE) * 100
      
      data.frame(
        Model            = col_m,
        Error_Column     = col_sd,
        SD_Min           = round(min(sds, na.rm = TRUE), 4),
        SD_Mean          = round(mean(sds, na.rm = TRUE), 4),
        SD_Max           = round(max(sds, na.rm = TRUE), 4),
        Max_Min_Ratio    = round(max(sds, na.rm = TRUE) / min(sds, na.rm = TRUE), 2),
        CV_SD_Percentage = round((sd(sds, na.rm = TRUE) / mean(sds, na.rm = TRUE)) * 100, 2),
        Empirical_Cov_95 = round(cov_95, 1)
      )
    }))
    datatable(tabla_cv, options = list(pageLength = 15, dom = "t"), rownames = FALSE)
  })
  
  output$plot_loess_grid <- renderPlot(
    height = function() {
      idx_v   <- obtener_indices_con_sd()
      p_num   <- max(1L, length(idx_v))
      n_filas <- ceiling(p_num / 2)
      w_px    <- session$clientData$output_plot_loess_grid_width
      if (is.null(w_px) || is.na(w_px) || w_px < 300) w_px <- 900
      alto_fila <- round((w_px / 2) * 0.78)
      max(420L, as.integer(n_filas * alto_fila))
    },
    res = 96,
    {
      req(estado$df_train)
      idx_valid <- obtener_indices_con_sd()
      validate(need(length(idx_valid) > 0, "No error columns (s_ik) available to plot. Fit fit_LOESS or proceed to Tab 3 if using Homoscedastic mode."))
      
      df_tr <- estado$df_train
      y_val <- df_tr[[estado$y_col]]
      
      lista_paneles <- lapply(idx_valid, function(i) {
        mod_id <- estado$m_cols[i]
        sd_id  <- estado$sd_cols[i]
        pred   <- df_tr[[mod_id]]
        sd_e   <- df_tr[[sd_id]]
        res    <- y_val - pred
        
        ic_low_y <- pred - 1.96 * sd_e
        ic_up_y  <- pred + 1.96 * sd_e
        cov_emp  <- mean(y_val >= ic_low_y & y_val <= ic_up_y, na.rm = TRUE) * 100
        
        df_m <- data.frame(
          panel_label = sprintf("%s (%s)  |  95%% Band Cov: %.1f%%", mod_id, sd_id, cov_emp),
          pred        = pred,
          Y           = y_val,
          residuo     = res,
          ic_low_y    = ic_low_y,
          ic_up_y     = ic_up_y,
          ic_low_r    = -1.96 * sd_e,
          ic_up_r     = +1.96 * sd_e
        )
        df_m[order(df_m$pred), ]
      })
      
      df_grid <- do.call(rbind, lista_paneles)
      df_grid$panel_label <- factor(df_grid$panel_label, levels = unique(df_grid$panel_label))
      
      if (input$loess_tipo_grafico == "residuos") {
        p <- ggplot(df_grid, aes(x = pred, y = residuo)) +
          geom_point(alpha = 0.25, size = 0.9, color = "steelblue4") +
          geom_ribbon(aes(ymin = ic_low_r, ymax = ic_up_r), fill = "#D55E00", alpha = 0.20) +
          geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
          geom_line(aes(y = ic_up_r), color = "#D55E00", linewidth = 0.8) +
          geom_line(aes(y = ic_low_r), color = "#D55E00", linewidth = 0.8) +
          facet_wrap(~ panel_label, ncol = 2, scales = "fixed") +
          labs(x = "Base model prediction", y = "Prediction residual (Y - M_k)") +
          theme_bw(base_size = 13) +
          theme(strip.text = element_text(face = "bold"))
      } else {
        p <- ggplot(df_grid, aes(x = pred, y = Y)) +
          geom_point(alpha = 0.25, size = 0.9, color = "grey30") +
          geom_ribbon(aes(ymin = ic_low_y, ymax = ic_up_y), fill = "#D55E00", alpha = 0.20) +
          geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "black") +
          geom_line(aes(y = ic_up_y), color = "#D55E00", linewidth = 0.8) +
          geom_line(aes(y = ic_low_y), color = "#D55E00", linewidth = 0.8) +
          facet_wrap(~ panel_label, ncol = 2, scales = "fixed") +
          labs(x = "Base model prediction", y = "Verifying observation (Y)") +
          theme_bw(base_size = 13) +
          theme(strip.text = element_text(face = "bold"))
      }
      
      estado$plot_loess <- p
      return(p)
    }
  )
  
  output$descargar_fig_loess <- downloadHandler(
    filename = function() {
      paste0("bmetapred_loess_", input$loess_tipo_grafico, ".png")
    },
    content = function(file) {
      req(estado$plot_loess)
      idx_v   <- obtener_indices_con_sd()
      p_num   <- max(1L, length(idx_v))
      n_filas <- ceiling(p_num / 2)
      ggsave(file, plot = estado$plot_loess, width = 8.8, height = max(5.5, n_filas * 3.4), dpi = 300)
    }
  )
  
  # ----------------------------------------------------------------------------
  # EVENT 4 (BLOCK C): 4-BAR FLICKER-FREE ASYNCHRONOUS MONITOR
  # ----------------------------------------------------------------------------
  observeEvent(input$btn_fit_bmetapred, {
    if (is.null(estado$df_train)) {
      showNotification("Please complete Step 1 (Data Partition) first.", type = "warning")
      return()
    }
    if (!isTRUE(estado$sd_ready)) {
      showNotification(
        sprintf("Please run Step 2 (fit_LOESS in Tab 2) first to calibrate error scales for: %s.",
                paste(estado$models_need_loess, collapse = ", ")),
        type = "warning", duration = 8
      )
      return()
    }
    if (isTRUE(estado$bg_running)) {
      showNotification("A model fitting process is already running.", type = "warning")
      return()
    }
    
    p_actual   <- length(estado$m_cols)
    pmax_usado <- calcular_pmax_interno(p_actual, input$b_search_mode)
    burn_val   <- if (!is.null(input$b_burnin) && input$b_burnin > 0) as.integer(input$b_burnin) else NULL
    n_mc3_val  <- if (!is.null(input$b_mc3) && !is.na(input$b_mc3)) as.integer(input$b_mc3) else 1500L
    min_k_val  <- if (!is.null(input$b_minK) && !is.na(input$b_minK)) as.integer(input$b_minK) else 10L
    max_k_val  <- if (!is.null(input$b_maxK) && !is.na(input$b_maxK)) as.integer(input$b_maxK) else 50L
    
    df_tr_fit <- estado$df_train
    for (i in seq_along(estado$sd_cols)) {
      sc <- estado$sd_cols[i]
      if (isTRUE(estado$sd_is_homoscedastic[i]) || !(sc %in% names(df_tr_fit))) {
        df_tr_fit[[sc]] <- 0.0
      }
    }
    
    log_tmp <- tempfile(pattern = "bmetapred_live_log_", fileext = ".txt")
    writeLines(c("Initializing bmetapred asynchronous Stan worker..."), log_tmp)
    
    args_fit <- list(
      data_train       = df_tr_fit,
      y                = estado$y_col,
      predictors       = estado$m_cols,
      sd_cols          = as.character(estado$sd_cols),
      method           = input$b_method,
      p_max_exhaust    = pmax_usado,
      n_iter_mc3       = n_mc3_val,
      burnin_pbma      = burn_val,
      min_K            = min_k_val,
      max_K            = max_k_val,
      pri_sd           = as.numeric(input$b_prisd),
      pri_sd_int       = as.numeric(input$b_prisd_int),
      pri_sd_sigma     = as.numeric(input$b_prisd_sigma),
      pri_sd_int_sigma = as.numeric(input$b_prisd_int_sigma),
      parallel_workers = as.integer(input$b_workers),
      cores_stan       = as.integer(input$b_coresstan),
      iter_stan        = as.integer(input$b_iterstan),
      verbosity        = 1,
      backend          = input$b_backend
    )
    
    estado$modelo_bm     <- NULL
    estado$bg_error_msg  <- NULL
    estado$bg_log_file   <- log_tmp
    estado$bg_start_time <- Sys.time()
    estado$bg_file_size  <- -1
    estado$bg_logs       <- character(0)
    estado$bg_running    <- TRUE
    
    estado$bg_process <- callr::r_bg(
      func = function(a) {
        do.call(bmetapred::fit_bmetapred, a)
      },
      args      = list(a = args_fit),
      stdout    = log_tmp,
      stderr    = log_tmp,
      supervise = TRUE
    )
  })
  
  observeEvent(input$btn_cancel_bmetapred, {
    if (!is.null(estado$bg_process) && estado$bg_process$is_alive()) {
      estado$bg_process$kill()
    }
    estado$bg_running   <- FALSE
    estado$bg_error_msg <- "Execution was manually cancelled by the user."
    showNotification("fit_bmetapred execution cancelled.", type = "warning")
  })
  
  parsear_telemetria_logs <- function(raw_lines, max_K_val, p_val, modo_busq, mc3_val) {
    fragmentos     <- unlist(strsplit(raw_lines, "\r", fixed = TRUE))
    lineas_limpias <- trimws(fragmentos)
    lineas_limpias <- lineas_limpias[nzchar(lineas_limpias)]
    texto          <- paste(lineas_limpias, collapse = "\n")
    
    es_exh <- identical(modo_busq, "exhaustive")
    n_cand <- if (es_exh) (2^p_val) else mc3_val
    
    pct_0 <- 20
    pct_1 <- 0
    pct_2 <- 0
    pct_3 <- 0
    
    phase_idx <- 0
    etapa     <- "Subphase 0: Auditing Errors & Compiling Base Stan Templates"
    detalle   <- if (length(lineas_limpias) > 0) tail(lineas_limpias, 1) else "Starting worker..."
    
    if (grepl("Compiling base models", texto, ignore.case = TRUE)) {
      pct_0     <- 65
      phase_idx <- 0
      etapa     <- "Subphase 0: Pre-compiling Stan C++ Distributional Templates"
    }
    
    idx_p1_start <- grep("Phase 1|Exhaustive|MC3|Tabu|Evaluating|Discovery|Searching|Exploring", lineas_limpias, ignore.case = TRUE)
    if (length(idx_p1_start) > 0) {
      pct_0     <- 100
      phase_idx <- 1
      lineas_p1 <- lineas_limpias[idx_p1_start[1]:length(lineas_limpias)]
      
      idx_p2_in_p1 <- grep("Phase 2|Building Stacking Matrix|Extracting pointwise LOO|Calculating predictive p-BMA", lineas_p1, ignore.case = TRUE)
      if (length(idx_p2_in_p1) > 0) {
        lineas_p1 <- if (idx_p2_in_p1[1] > 1) lineas_p1[1:(idx_p2_in_p1[1] - 1)] else character(0)
      }
      texto_p1 <- paste(lineas_p1, collapse = "\n")
      
      m_pct  <- regmatches(texto_p1, gregexpr("[0-9]+(\\.[0-9]+)?\\s*%", texto_p1))[[1]]
      m_frac <- regmatches(texto_p1, gregexpr("[0-9]+\\s*/\\s*[0-9]+", texto_p1))[[1]]
      m_bars <- regmatches(texto_p1, gregexpr("\\|={1,}\\s*\\|", texto_p1))[[1]]
      
      if (length(m_pct) > 0) {
        ult_val <- as.numeric(gsub("[^0-9\\.]", "", tail(m_pct, 1)))
        pct_1   <- min(99, max(2, round(ult_val)))
      } else if (length(m_frac) > 0) {
        partes  <- as.numeric(strsplit(gsub("\\s+", "", tail(m_frac, 1)), "/")[[1]])
        if (length(partes) == 2 && partes[2] > 0) {
          pct_1 <- min(99, max(2, round(100 * partes[1] / partes[2])))
        }
      } else if (length(m_bars) > 0) {
        ult_bar <- tail(m_bars, 1)
        n_eq    <- nchar(gsub("[^=]", "", ult_bar))
        n_tot   <- max(1, nchar(ult_bar) - 2)
        pct_1   <- min(99, max(2, round(100 * n_eq / n_tot)))
      } else {
        n_ev  <- length(grep("Model|Iter|ELPD|Step|Candidate|fit", lineas_p1, ignore.case = TRUE))
        pct_1 <- min(95, max(5, round(100 * n_ev / max(1, n_cand))))
      }
      etapa <- sprintf("Subphase 1: Exploring Meta-Model Space (%d%% completed)", pct_1)
    }
    
    if (grepl("Phase 2: Building Stacking Matrix|Extracting pointwise LOO", texto, ignore.case = TRUE)) {
      pct_0     <- 100
      pct_1     <- 100
      phase_idx <- 2
      ranks     <- regmatches(texto, gregexpr("Rank [0-9]+", texto))[[1]]
      ult_r     <- if (length(ranks) > 0) as.numeric(sub("Rank ", "", tail(ranks, 1))) else 1
      denom_k   <- max(1, min(max_K_val, n_cand))
      pct_2     <- min(98, max(5, round(100 * ult_r / denom_k)))
      if (grepl("Early stop|max_K Limit reached", texto, ignore.case = TRUE)) {
        pct_2 <- 100
      }
      etapa <- sprintf("Subphase 2: Extracting Pointwise LOO for Top Models (Model Rank %d)", ult_r)
    }
    
    if (grepl("Phase 2/3: Calculating predictive p-BMA|Phase 3: Optimizing stacking weights", texto, ignore.case = TRUE)) {
      pct_0     <- 100
      pct_1     <- 100
      pct_2     <- 100
      phase_idx <- 3
      pct_3     <- 85
      etapa     <- "Subphase 3: Optimizing Final Bayesian Ensemble Weights"
    }
    
    lineas_hitos  <- lineas_limpias[!grepl("^Warning|^In |^\\s*$", lineas_limpias)]
    ultimos_hitos <- tail(lineas_hitos, 10)
    html_items <- paste(
      vapply(ultimos_hitos, function(l) {
        sprintf("<div class='milestone-item'><span>%s</span><span class='badge bg-secondary-subtle text-secondary'>OK</span></div>",
                htmltools::htmlEscape(l))
      }, character(1)),
      collapse = ""
    )
    
    list(
      pcts            = c(pct_0, pct_1, pct_2, pct_3),
      phase_idx       = phase_idx,
      etapa           = etapa,
      detalle         = detalle,
      milestones_html = html_items
    )
  }
  
  temporizador_bg <- reactiveTimer(500)
  
  observe({
    temporizador_bg()
    if (!isTRUE(isolate(estado$bg_running))) return()
    
    elapsed_sec <- as.numeric(difftime(Sys.time(), isolate(estado$bg_start_time), units = "secs"))
    mins        <- floor(elapsed_sec / 60)
    secs        <- round(elapsed_sec %% 60)
    str_elapsed <- sprintf("Elapsed Time: %02dm %02ds", mins, secs)
    
    log_f  <- isolate(estado$bg_log_file)
    f_size <- if (!is.null(log_f) && file.exists(log_f)) file.info(log_f)$size else -1
    if (is.na(f_size)) f_size <- -1
    
    hay_cambio_log <- f_size > isolate(estado$bg_file_size)
    
    if (hay_cambio_log) {
      estado$bg_file_size <- f_size
      lineas_actuales     <- readLines(log_f, warn = FALSE)
      estado$bg_logs      <- lineas_actuales
      
      p_actual  <- if (!is.null(isolate(estado$m_cols))) length(isolate(estado$m_cols)) else 6
      max_k_use <- if (!is.null(isolate(input$b_maxK)) && !is.na(isolate(input$b_maxK))) isolate(input$b_maxK) else 50
      info      <- parsear_telemetria_logs(
        raw_lines = lineas_actuales,
        max_K_val = max_k_use,
        p_val     = p_actual,
        modo_busq = isolate(input$b_search_mode),
        mc3_val   = if (!is.null(isolate(input$b_mc3))) isolate(input$b_mc3) else 1500
      )
      
      session$sendCustomMessage("bm_prog_update", list(
        elapsed         = str_elapsed,
        new_event       = TRUE,
        pcts            = info$pcts,
        phase_idx       = info$phase_idx,
        etapa           = info$etapa,
        detalle         = info$detalle,
        milestones_html = info$milestones_html
      ))
    } else {
      session$sendCustomMessage("bm_prog_update", list(
        elapsed   = str_elapsed,
        new_event = FALSE
      ))
    }
    
    proc <- isolate(estado$bg_process)
    if (!is.null(proc) && !proc$is_alive()) {
      estado$bg_running <- FALSE
      
      if (!is.null(log_f) && file.exists(log_f)) {
        estado$bg_logs <- readLines(log_f, warn = FALSE)
      }
      
      res_try <- tryCatch({
        proc$get_result()
      }, error = function(e) {
        msg_real <- if (!is.null(e$parent) && !is.null(e$parent$message)) {
          e$parent$message
        } else {
          e$message
        }
        cola_log <- if (length(estado$bg_logs) > 0) {
          paste(tail(estado$bg_logs, 6), collapse = " | ")
        } else {
          ""
        }
        estado$bg_error_msg <- paste0(msg_real, if (nzchar(cola_log)) paste0(" — Log: ", cola_log) else "")
        showNotification(paste("Error in fit_bmetapred:", msg_real), type = "error", duration = 12)
        return(NULL)
      })
      
      if (!is.null(res_try)) {
        estado$modelo_bm <- res_try
        estado$tiempo_bm <- elapsed_sec
        
        # Use native bmetapred package functions for PIPs and model summary table
        estado$pips_bm <- bmetapred::get_pip_bmetapred(res_try)
        
        invisible(capture.output(df_sum <- summary(res_try)))
        estado$top_mods_bm <- df_sum[df_sum$Weight > 1e-5, , drop = FALSE]
        showNotification("fit_bmetapred completed successfully! Proceed to Tab 4 for Meta-Prediction.", type = "message")
      }
    }
  })
  
  output$panel_central_bmetapred <- renderUI({
    if (isTRUE(estado$bg_running)) {
      return(
        card(
          card_header(
            class = "bg-primary text-white d-flex justify-content-between align-items-center",
            span(icon("gears"), strong(" Live Algorithm Progress — Bayesian Distributional Meta-Model")),
            span(id = "bm_elapsed_badge", class = "badge bg-light text-dark fs-6", "Elapsed Time: 00m 00s")
          ),
          card_body(
            h5(id = "bm_phase_title", class = "mb-3 text-primary", "Subphase 0: Auditing Errors & Compiling Base Stan Templates"),
            
            layout_columns(
              col_widths = c(2, 5, 3, 2),
              
              div(
                id = "bm_step_0", class = "phase-step active",
                div(class = "d-flex justify-content-between align-items-center mb-1",
                    span(style = "font-size: 0.75rem; text-transform: uppercase; font-weight: 700;", "0. Compile Base"),
                    span(id = "bm_pct_lbl_0", class = "badge bg-secondary-subtle text-dark", "20%")),
                div(class = "progress", style = "height: 20px;",
                    div(id = "bm_bar_0",
                        class = "progress-bar progress-bar-striped progress-bar-animated bg-primary fw-bold",
                        role = "progressbar", style = "width: 20%; transition: width 0.4s ease;", "20%"))
              ),
              
              div(
                id = "bm_step_1", class = "phase-step",
                div(class = "d-flex justify-content-between align-items-center mb-1",
                    span(style = "font-size: 0.82rem; text-transform: uppercase; font-weight: 700;", "1. Model Space Exploration"),
                    span(id = "bm_pct_lbl_1", class = "badge bg-secondary-subtle text-dark", "0%")),
                div(class = "progress", style = "height: 24px;",
                    div(id = "bm_bar_1",
                        class = "progress-bar bg-secondary",
                        role = "progressbar", style = "width: 0%; transition: width 0.4s ease;", ""))
              ),
              
              div(
                id = "bm_step_2", class = "phase-step",
                div(class = "d-flex justify-content-between align-items-center mb-1",
                    span(style = "font-size: 0.75rem; text-transform: uppercase; font-weight: 700;", "2. Top Models LOO"),
                    span(id = "bm_pct_lbl_2", class = "badge bg-secondary-subtle text-dark", "0%")),
                div(class = "progress", style = "height: 20px;",
                    div(id = "bm_bar_2",
                        class = "progress-bar bg-secondary",
                        role = "progressbar", style = "width: 0%; transition: width 0.4s ease;", ""))
              ),
              
              div(
                id = "bm_step_3", class = "phase-step",
                div(class = "d-flex justify-content-between align-items-center mb-1",
                    span(style = "font-size: 0.75rem; text-transform: uppercase; font-weight: 700;", "3. Ensemble Weights"),
                    span(id = "bm_pct_lbl_3", class = "badge bg-secondary-subtle text-dark", "0%")),
                div(class = "progress", style = "height: 20px;",
                    div(id = "bm_bar_3",
                        class = "progress-bar bg-secondary",
                        role = "progressbar", style = "width: 0%; transition: width 0.4s ease;", ""))
              )
            ),
            
            div(
              class = "alert alert-light border py-2 px-3 mt-3 mb-3",
              div(class = "text-muted", style = "font-size: 0.8rem; text-transform: uppercase;", "Latest Model Event:"),
              div(id = "bm_sub_detail", class = "fw-semibold text-dark", "Initializing Stan worker...")
            ),
            
            p(class = "text-muted mb-1", style = "font-size: 0.85rem;", "Completed Algorithm Milestones:"),
            div(
              id    = "bm_milestones_list",
              class = "milestone-feed",
              div(class = "milestone-item", span("Worker initialized"), span(class = "badge bg-secondary-subtle text-secondary", "OK"))
            ),
            
            div(
              class = "d-flex justify-content-end mt-3",
              actionButton("btn_cancel_bmetapred", "Cancel / Stop Execution", class = "btn-outline-danger")
            )
          )
        )
      )
    }
    
    if (!is.null(estado$modelo_bm)) {
      return(
        layout_columns(
          col_widths = c(12, 12),
          card(
            card_header(
              class = "bg-success text-white d-flex justify-content-between align-items-center",
              span("Model Fitting Summary & Posterior Inclusion Probabilities (PIPs)"),
              span(class = "badge bg-light text-dark", sprintf("Completed in %.1f s", isolate(estado$tiempo_bm)))
            ),
            verbatimTextOutput("texto_resumen_bmetapred")
          ),
          card(
            card_header("Selected Top Meta-Models and Ensemble Weights"),
            DTOutput("tabla_top_modelos")
          )
        )
      )
    }
    
    p_len <- if (!is.null(estado$m_cols)) length(estado$m_cols) else 0
    modo_sel <- if (!is.null(input$b_search_mode)) input$b_search_mode else if (p_len <= 9) "exhaustive" else "heuristic"
    modo_busqueda <- if (p_len > 0 && modo_sel == "exhaustive") {
      sprintf("Exhaustive Search (2^%d = %s candidate distributional meta-models)", p_len, format(2^p_len, big.mark = ","))
    } else if (p_len > 0) {
      sprintf("Heuristic Search (%d iterations over 2^%d = %s candidate models)",
              if (!is.null(input$b_mc3)) input$b_mc3 else 1500, p_len, format(2^p_len, big.mark = ","))
    } else {
      "Load data in Tab 1 first"
    }
    
    sd_status_txt <- if (is.null(estado$df_train)) {
      "Not loaded yet"
    } else if (all(estado$sd_is_homoscedastic)) {
      "Homoscedastic Mode (No s_ik errors used; all models use sigma ~ 1 template)"
    } else if (isTRUE(estado$sd_ready)) {
      etiquetas <- ifelse(estado$sd_is_homoscedastic, "None (Homoscedastic)", estado$sd_cols)
      sprintf("Ready (%s)", paste(etiquetas, collapse = ", "))
    } else {
      sprintf("Pending Tab 2 (fit_LOESS) for: %s", paste(estado$models_need_loess, collapse = ", "))
    }
    
    card(
      card_header("Pre-Execution Summary & Computational Preview"),
      card_body(
        if (!is.null(estado$bg_error_msg)) {
          div(
            class = "alert alert-danger",
            strong("Last Execution Stopped / Error: "), estado$bg_error_msg
          )
        },
        h5("Ready to Fit Bayesian Distributional Meta-Model"),
        tags$ul(
          tags$li(strong("Training Sample Size: "), if (!is.null(estado$df_train)) sprintf("N_train = %d observations", nrow(estado$df_train)) else "Not loaded yet"),
          tags$li(strong("Base Predictors (p): "), if (p_len > 0) sprintf("%d models (%s)", p_len, paste(estado$m_cols, collapse = ", ")) else "None"),
          tags$li(strong("Error Scales (s_ik) Status: "), sd_status_txt),
          tags$li(strong("Search Strategy: "), modo_busqueda),
          tags$li(strong("MCMC Configuration: "), sprintf("Backend = %s | %d iterations | %d parallel workers", input$b_backend, input$b_iterstan, input$b_workers))
        ),
        p(class = "text-muted mt-2", "Click 'Run fit_bmetapred (Fit Model Space)' on the left sidebar to start.")
      )
    )
  })
  
  output$texto_resumen_bmetapred <- renderPrint({
    req(estado$modelo_bm)
    cat(sprintf("Methodology: %s | Backend: %s\n", toupper(input$b_method), toupper(input$b_backend)))
    cat(sprintf("Total fitting execution time: %.1f seconds\n", estado$tiempo_bm))
    cat(sprintf("Selected top meta-models: %d\n\n", nrow(estado$top_mods_bm)))
    cat("Posterior Inclusion Probabilities (PIPs) — bmetapred::get_pip_bmetapred():\n")
    print(round(estado$pips_bm, 4))
    cat("\n--- Package summary(model) output ---\n")
    print(summary(estado$modelo_bm))
  })
  
  output$tabla_top_modelos <- renderDT({
    req(estado$top_mods_bm)
    datatable(estado$top_mods_bm, options = list(pageLength = 10, dom = "tp"), rownames = FALSE)
  })
  
  # ----------------------------------------------------------------------------
  # EVENT 5 (BLOCK D): EXACT OUT-OF-SAMPLE META-PREDICTION & EVALUATION
  # ----------------------------------------------------------------------------
  observeEvent(input$btn_run_predict, {
    if (is.null(estado$modelo_bm)) {
      showNotification("Please fit the meta-model in Tab 3 (fit_bmetapred) before running Meta-Prediction.", type = "warning")
      return()
    }
    
    if (input$pred_source == "reserved_test") {
      req(estado$df_test)
      df_eval <- estado$df_test
    } else {
      if (is.null(input$file_ext_test)) {
        showNotification("Please upload an external test dataset (.csv or .rds).", type = "error")
        return()
      }
      ext <- tools::file_ext(input$file_ext_test$name)
      df_ext_te <- if (tolower(ext) == "csv") {
        read.csv(input$file_ext_test$datapath, header = input$csv_header, sep = input$csv_sep, stringsAsFactors = FALSE, check.names = FALSE)
      } else {
        as.data.frame(readRDS(input$file_ext_test$datapath))
      }
      nombres_te <- gsub("[^[:alnum:]_]", "_", trimws(names(df_ext_te)))
      nombres_te <- gsub("_+", "_", nombres_te)
      nombres_te <- gsub("^_|_$", "", nombres_te)
      names(df_ext_te) <- make.names(nombres_te, unique = TRUE)
      
      if (!all(estado$m_cols %in% names(df_ext_te))) {
        showNotification("External test dataset must contain all base predictor columns (M1...Mp).", type = "error")
        return()
      }
      
      df_eval <- df_ext_te
      sd_hetero <- estado$sd_cols[!estado$sd_is_homoscedastic]
      
      if (isTRUE(estado$escala_Y$estandarizado)) {
        mu_Y <- estado$escala_Y$mu_Y
        sd_Y <- estado$escala_Y$sd_Y
        if (estado$y_col %in% names(df_eval)) df_eval[[estado$y_col]] <- (df_eval[[estado$y_col]] - mu_Y) / sd_Y
        for (mc in estado$m_cols) df_eval[[mc]] <- (df_eval[[mc]] - mu_Y) / sd_Y
        for (sc in sd_hetero) {
          if (sc %in% names(df_eval)) df_eval[[sc]] <- df_eval[[sc]] / sd_Y
        }
      }
      
      if (length(estado$models_need_loess) > 0) {
        sd_needed <- paste0("sd_", estado$models_need_loess)
        if (!all(sd_needed %in% names(df_eval))) {
          if (is.null(estado$loess_obj)) {
            showNotification("External test file is missing LOESS sd_ columns and no fit_LOESS object is available.", type = "error")
            return()
          }
          df_proj <- predict(estado$loess_obj, newdata = df_eval[, estado$models_need_loess, drop = FALSE], prefix = "sd_")
          for (sc in sd_needed) df_eval[[sc]] <- df_proj[[sc]]
        }
      }
    }
    
    for (i in seq_along(estado$sd_cols)) {
      sc <- estado$sd_cols[i]
      if (isTRUE(estado$sd_is_homoscedastic[i]) || !(sc %in% names(df_eval))) {
        df_eval[[sc]] <- 0.0
      }
    }
    
    n_samp <- if (!is.null(input$pred_nsamples) && !is.na(input$pred_nsamples)) as.integer(input$pred_nsamples) else 2000L
    
    withProgress(message = sprintf("Sampling %d posterior predictive draws on Test Set...", n_samp), value = 0.3, {
      res_pred <- tryCatch({
        predict(
          estado$modelo_bm,
          newdata           = df_eval,
          n_samples_totales = n_samp,
          estimator         = "ensemble",
          verbosity         = 0
        )
      }, error = function(e) {
        showNotification(paste("Error during predict():", e$message), type = "error", duration = 10)
        return(NULL)
      })
      
      incProgress(0.5)
      if (!is.null(res_pred)) {
        estado$pred_raw_std <- res_pred
        estado$df_eval_std  <- df_eval
        recalcular_tablas_pred()
        showNotification("Out-of-sample meta-prediction completed!", type = "message")
      }
      incProgress(0.2)
    })
  })
  
  observeEvent(list(input$pred_level, input$pred_destd), {
    if (!is.null(estado$pred_raw_std) && !is.null(estado$df_eval_std)) {
      recalcular_tablas_pred()
    }
  }, ignoreInit = TRUE)
  
  recalcular_tablas_pred <- function() {
    req(estado$pred_raw_std, estado$df_eval_std)
    df_ev <- estado$df_eval_std
    n_obs <- nrow(df_ev)
    lvl   <- as.numeric(input$pred_level)
    alpha <- 1 - lvl
    z_val <- qnorm(1 - alpha / 2)
    
    usar_orig <- isTRUE(input$pred_destd) && isTRUE(estado$escala_Y$estandarizado)
    mu_anchor <- if (usar_orig) estado$escala_Y$mu_Y else 0
    sd_anchor <- if (usar_orig) estado$escala_Y$sd_Y else 1
    
    # Extract exact raw posterior predictive draws matrix [n_samples x n_obs] from predict.bmetapred_model()
    S_raw <- attr(estado$pred_raw_std, "raw_samples")
    if (is.null(S_raw) || !is.matrix(S_raw)) {
      showNotification("The predict() object did not return the 'raw_samples' posterior matrix.", type = "error")
      return()
    }
    
    # Transform raw posterior draws to target physical/standardized scale and compute exact empirical quantiles
    S_final <- mu_anchor + sd_anchor * S_raw
    mu_bm   <- colMeans(S_final, na.rm = TRUE)
    sd_bm   <- apply(S_final, 2, sd, na.rm = TRUE)
    low_bm  <- apply(S_final, 2, quantile, probs = alpha / 2, na.rm = TRUE)
    up_bm   <- apply(S_final, 2, quantile, probs = 1 - alpha / 2, na.rm = TRUE)
    
    tiene_y_test <- estado$y_col %in% names(df_ev) && !all(is.na(df_ev[[estado$y_col]]))
    y_test_vec   <- if (tiene_y_test) (mu_anchor + sd_anchor * df_ev[[estado$y_col]]) else rep(NA_real_, n_obs)
    
    df_out <- data.frame(
      Obs_ID         = seq_len(n_obs),
      Y_Observed     = round(y_test_vec, 4),
      Pred_Mean      = round(mu_bm, 4),
      Pred_SD        = round(sd_bm, 4),
      CI_Lower       = round(low_bm, 4),
      CI_Upper       = round(up_bm, 4),
      Interval_Width = round(up_bm - low_bm, 4),
      Residual       = if (tiene_y_test) round(y_test_vec - mu_bm, 4) else NA_real_,
      Covered        = if (tiene_y_test) ifelse(y_test_vec >= low_bm & y_test_vec <= up_bm, "Yes", "No") else NA_character_
    )
    estado$df_pred_final <- df_out
    
    if (tiene_y_test) {
      # 1. Exact bmetapred evaluation using S_final (exact quantiles + scoringRules::crps_sample)
      m_bm <- calcular_metricas_exactas(
        y_true      = y_test_vec,
        mu_pred     = mu_bm,
        lower_ci    = low_bm,
        upper_ci    = up_bm,
        alpha       = alpha,
        raw_samples = S_final
      )
      
      # 2. Individual Base Models evaluation (only point metrics when no s_ik error scale is provided)
      filas_base <- lapply(seq_along(estado$m_cols), function(k) {
        mk_pred         <- mu_anchor + sd_anchor * df_ev[[estado$m_cols[k]]]
        tiene_sd_propio <- !isTRUE(estado$sd_is_homoscedastic[k])
        
        if (tiene_sd_propio && estado$sd_cols[k] %in% names(df_ev)) {
          sk_pred <- sd_anchor * df_ev[[estado$sd_cols[k]]]
          mk_met  <- calcular_metricas_exactas(
            y_true   = y_test_vec,
            mu_pred  = mk_pred,
            lower_ci = mk_pred - z_val * sk_pred,
            upper_ci = mk_pred + z_val * sk_pred,
            alpha    = alpha,
            sd_norm  = sk_pred
          )
          data.frame(
            Model             = estado$m_cols[k],
            Type              = "Base Model (with s_ik)",
            MWIS              = round(mk_met$MWIS, 4),
            CRPS              = round(mk_met$CRPS, 4),
            Empirical_Cov_Pct = round(mk_met$Cov_Pct, 1),
            Mean_Width        = round(mk_met$Mean_Width, 4),
            RMSE              = round(mk_met$RMSE, 4),
            MAE               = round(mk_met$MAE, 4),
            Mean_Bias         = round(mk_met$Mean_Bias, 4),
            R2                = round(mk_met$R2, 4)
          )
        } else {
          ok     <- !is.na(y_test_vec) & !is.na(mk_pred)
          err_k  <- y_test_vec[ok] - mk_pred[ok]
          ss_res <- sum(err_k^2)
          ss_tot <- sum((y_test_vec[ok] - mean(y_test_vec[ok]))^2)
          data.frame(
            Model             = estado$m_cols[k],
            Type              = "Base Model (Point Only / No s_ik)",
            MWIS              = NA_real_,
            CRPS              = NA_real_,
            Empirical_Cov_Pct = NA_real_,
            Mean_Width        = NA_real_,
            RMSE              = round(sqrt(mean(err_k^2)), 4),
            MAE               = round(mean(abs(err_k)), 4),
            Mean_Bias         = round(mean(err_k), 4),
            R2                = round(if (ss_tot > 0) 1 - (ss_res / ss_tot) else NA_real_, 4)
          )
        }
      })
      
      df_comp <- rbind(
        data.frame(
          Model             = sprintf("bmetapred (%s)", toupper(input$b_method)),
          Type              = "Bayesian Distributional Ensemble",
          MWIS              = round(m_bm$MWIS, 4),
          CRPS              = round(m_bm$CRPS, 4),
          Empirical_Cov_Pct = round(m_bm$Cov_Pct, 1),
          Mean_Width        = round(m_bm$Mean_Width, 4),
          RMSE              = round(m_bm$RMSE, 4),
          MAE               = round(m_bm$MAE, 4),
          Mean_Bias         = round(m_bm$Mean_Bias, 4),
          R2                = round(m_bm$R2, 4)
        ),
        do.call(rbind, filas_base)
      )
      estado$df_metrics_comp <- df_comp
    } else {
      estado$df_metrics_comp <- NULL
    }
  }
  
  output$ui_banner_pred <- renderUI({
    validate(need(!is.null(estado$df_pred_final), "4. Click 'Run Meta-Prediction (predict)' on the left sidebar to evaluate out-of-sample performance."))
    escala_txt <- if (isTRUE(input$pred_destd) && isTRUE(estado$escala_Y$estandarizado)) {
      sprintf("Original Physical Scale (de-standardized with mu_Y = %.3f, sd_Y = %.3f)", estado$escala_Y$mu_Y, estado$escala_Y$sd_Y)
    } else if (isTRUE(estado$escala_Y$estandarizado)) {
      "Standardized Scale (mean 0, SD 1)"
    } else {
      "Original Unscaled Units"
    }
    crps_note <- if (!requireNamespace("scoringRules", quietly = TRUE)) {
      " | Warning: install 'scoringRules' to compute exact sample CRPS."
    } else {
      ""
    }
    div(
      class = "alert alert-light border mb-3 py-2",
      strong("Evaluation Summary: "),
      sprintf("N_test = %d observations | Exact Empirical Credible Interval: %.0f%% | Posterior Draws: %d | Scale: %s%s",
              nrow(estado$df_pred_final), input$pred_level * 100, nrow(attr(estado$pred_raw_std, "raw_samples")), escala_txt, crps_note)
    )
  })
  
  output$tabla_metricas_test <- renderDT({
    req(estado$df_metrics_comp)
    datatable(
      estado$df_metrics_comp,
      options  = list(pageLength = 15, dom = "t"),
      rownames = FALSE
    ) %>%
      formatStyle(
        "Model",
        target          = "row",
        fontWeight      = styleEqual(sprintf("bmetapred (%s)", toupper(input$b_method)), "bold"),
        backgroundColor = styleEqual(sprintf("bmetapred (%s)", toupper(input$b_method)), "#eafaf1")
      )
  })
  
  output$tabla_pred_individuales <- renderDT({
    validate(need(!is.null(estado$df_pred_final), "Run Meta-Prediction first to inspect individual test predictions."))
    datatable(estado$df_pred_final, options = list(pageLength = 12, scrollX = TRUE), rownames = FALSE)
  })
  
  # ----------------------------------------------------------------------------
  # SINGLE-PLOT FULL-SCREEN VIEWER WITH PREV/NEXT & DYNAMIC DESCRIPTIVE TITLE
  # ----------------------------------------------------------------------------
  construir_plot_intervals <- reactive({
    req(estado$df_pred_final)
    df_p   <- estado$df_pred_final
    n_show <- min(nrow(df_p), as.integer(input$pred_n_plot))
    df_sub <- df_p[seq_len(n_show), ]
    df_sub <- df_sub[order(df_sub$Pred_Mean), ]
    df_sub$Rank_Ord <- seq_len(nrow(df_sub))
    
    cov_pct <- mean(df_p$Covered == "Yes", na.rm = TRUE) * 100
    
    ggplot(df_sub, aes(x = Rank_Ord)) +
      geom_ribbon(aes(ymin = CI_Lower, ymax = CI_Upper), fill = "#3498db", alpha = 0.25) +
      geom_errorbar(aes(ymin = CI_Lower, ymax = CI_Upper), color = "#2980b9", alpha = 0.45, width = 0.2) +
      geom_line(aes(y = Pred_Mean), color = "#1f618d", linewidth = 0.9) +
      geom_point(aes(y = Y_Observed, color = Covered), size = 2.0, alpha = 0.9) +
      scale_color_manual(values = c("Yes" = "#27ae60", "No" = "#e74c3c")) +
      labs(
        title    = sprintf("Out-of-Sample %.0f%% Predictive Intervals (Ordered by Predicted Magnitude)", input$pred_level * 100),
        subtitle = sprintf("Overall Empirical Coverage on Test Set: %.1f%% (Nominal: %.0f%%) | Showing %d observations",
                           cov_pct, input$pred_level * 100, nrow(df_sub)),
        x        = "Ordered Test Observation Index",
        y        = "Response Value (Y)",
        color    = "Inside CI?"
      ) +
      theme_bw(base_size = 14) +
      theme(legend.position = "top")
  })
  
  construir_plot_weights <- reactive({
    req(estado$pips_bm, estado$top_mods_bm)
    df_pips <- data.frame(
      Model = factor(names(estado$pips_bm), levels = names(estado$pips_bm)),
      PIP   = as.numeric(estado$pips_bm)
    )
    ggplot(df_pips, aes(x = reorder(Model, PIP), y = PIP)) +
      geom_col(fill = "#2C3E50", width = 0.65) +
      geom_text(aes(label = sprintf("%.3f", PIP)), hjust = -0.15, size = 4.2, fontface = "bold") +
      coord_flip(ylim = c(0, 1.12)) +
      labs(
        title    = sprintf("Posterior Inclusion Probabilities (PIPs) / Marginal Ensemble Weights (%s)", toupper(input$b_method)),
        subtitle = sprintf("Selected top meta-models: %d | Highest-weight model: %s (w = %.3f)",
                           nrow(estado$top_mods_bm), estado$top_mods_bm$Structure[1], estado$top_mods_bm$Weight[1]),
        x        = "Base Predictor Model",
        y        = "Marginal Inclusion Probability / Weight"
      ) +
      theme_bw(base_size = 14)
  })
  
  construir_plot_calibration <- reactive({
    req(estado$df_pred_final)
    df_p <- estado$df_pred_final
    ggplot(df_p, aes(x = Pred_Mean, y = Y_Observed)) +
      geom_errorbar(aes(ymin = CI_Lower, ymax = CI_Upper), color = "steelblue", alpha = 0.25) +
      geom_point(aes(color = Covered), alpha = 0.7, size = 1.8) +
      geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "black", linewidth = 0.9) +
      scale_color_manual(values = c("Yes" = "#2C3E50", "No" = "#e74c3c")) +
      labs(
        title    = "Out-of-Sample Calibration: Verifying Observation (Y) vs. bmetapred Prediction",
        subtitle = sprintf("Vertical bars denote observation-specific %.0f%% predictive credible intervals", input$pred_level * 100),
        x        = "bmetapred Posterior Predictive Mean",
        y        = "Observed Response (Y)",
        color    = "Covered?"
      ) +
      theme_bw(base_size = 14) +
      theme(legend.position = "top")
  })
  
  observeEvent(input$pred_plot_types, {
    sel <- input$pred_plot_types
    if (!is.null(sel) && length(sel) > 0) {
      if (!(estado$active_plot_d %in% sel)) {
        estado$active_plot_d <- sel[1]
      }
    }
  }, ignoreNULL = FALSE)
  
  observeEvent(input$btn_prev_plot_d, {
    sel <- input$pred_plot_types
    req(length(sel) > 1)
    idx <- match(estado$active_plot_d, sel)
    if (is.na(idx) || idx <= 1) {
      estado$active_plot_d <- sel[length(sel)]
    } else {
      estado$active_plot_d <- sel[idx - 1]
    }
  })
  
  observeEvent(input$btn_next_plot_d, {
    sel <- input$pred_plot_types
    req(length(sel) > 1)
    idx <- match(estado$active_plot_d, sel)
    if (is.na(idx) || idx >= length(sel)) {
      estado$active_plot_d <- sel[1]
    } else {
      estado$active_plot_d <- sel[idx + 1]
    }
  })
  
  output$plot_d_single_view <- renderPlot({
    sel <- input$pred_plot_types
    req(length(sel) > 0)
    curr <- if (estado$active_plot_d %in% sel) estado$active_plot_d else sel[1]
    
    if (curr == "intervals") {
      validate(need(!is.null(estado$df_pred_final), "Click 'Run Meta-Prediction (predict)' on the left sidebar to display the Predictive Intervals figure."))
      construir_plot_intervals()
    } else if (curr == "weights") {
      validate(need(!is.null(estado$pips_bm), "Fit the meta-model in Tab 3 first to display PIPs and model weights."))
      construir_plot_weights()
    } else if (curr == "calibration") {
      validate(need(!is.null(estado$df_pred_final), "Click 'Run Meta-Prediction (predict)' on the left sidebar to display the Calibration figure."))
      construir_plot_calibration()
    }
  })
  
  output$dl_active_plot_d <- downloadHandler(
    filename = function() {
      paste0("bmetapred_figure_", estado$active_plot_d, "_", Sys.Date(), ".png")
    },
    content = function(file) {
      curr <- estado$active_plot_d
      p_obj <- if (curr == "intervals") {
        construir_plot_intervals()
      } else if (curr == "weights") {
        construir_plot_weights()
      } else {
        construir_plot_calibration()
      }
      ggsave(file, plot = p_obj, width = 9.5, height = 6.2, dpi = 300)
    }
  )
  
  output$ui_contenedor_plots_d <- renderUI({
    seleccionados <- input$pred_plot_types
    if (is.null(seleccionados) || length(seleccionados) == 0) {
      return(
        div(
          class = "alert alert-secondary my-3",
          "No diagnostic figures are currently enabled. Check one, two, or all three options under 'Visualization Controls' on the left sidebar to view them."
        )
      )
    }
    
    catalogo_nombres <- c(
      "intervals"   = "Ordered Predictive Intervals & Empirical Coverage",
      "weights"     = "Base Model PIPs & Selected Meta-Model Weights",
      "calibration" = "Out-of-Sample Calibration (Verifying Observation vs. Prediction)"
    )
    
    curr     <- if (estado$active_plot_d %in% seleccionados) estado$active_plot_d else seleccionados[1]
    idx_curr <- match(curr, seleccionados)
    n_sel    <- length(seleccionados)
    
    card(
      card_header(
        class = "bg-light py-2 px-3",
        div(
          class = "d-flex flex-wrap justify-content-between align-items-center gap-2",
          
          div(
            class = "d-flex align-items-center gap-2",
            if (n_sel > 1) {
              actionButton("btn_prev_plot_d", label = "◀ Previous", class = "btn-outline-primary btn-sm fw-semibold")
            },
            span(
              class = "badge bg-primary px-3 py-2",
              style = "font-size: 0.86rem;",
              sprintf("Figure %d of %d", idx_curr, n_sel)
            ),
            if (n_sel > 1) {
              actionButton("btn_next_plot_d", label = "Next ▶", class = "btn-outline-primary btn-sm fw-semibold")
            }
          ),
          
          h5(
            class = "mb-0 fw-bold text-primary text-center",
            style = "font-size: 1.05rem;",
            catalogo_nombres[[curr]]
          ),
          
          downloadButton("dl_active_plot_d", "Download This Figure (PNG 300 DPI)", class = "btn-outline-secondary btn-sm")
        )
      ),
      card_body(
        plotOutput("plot_d_single_view", height = "600px")
      )
    )
  })
  
  # ----------------------------------------------------------------------------
  # REPRODUCIBLE R SCRIPT GENERATOR (SUB-TAB 4.4)
  # ----------------------------------------------------------------------------
  generar_script_r <- reactive({
    y_c   <- if (!is.null(estado$y_col)) estado$y_col else "Y"
    p_len <- if (!is.null(estado$m_cols)) length(estado$m_cols) else 6L
    m_c   <- if (!is.null(estado$m_cols)) paste0('"', estado$m_cols, '"', collapse = ", ") else '"M1", "M2", "M3"'
    
    sd_c <- if (is.null(estado$sd_cols)) {
      'paste0("sd_", predictors)'
    } else {
      paste0("c(", paste0('"', estado$sd_cols, '"', collapse = ", "), ")")
    }
    
    strat_loess <- if (!is.null(estado$loess_mode_used)) estado$loess_mode_used else "kfold"
    mods_loess  <- if (!is.null(estado$models_need_loess) && length(estado$models_need_loess) > 0) {
      paste0('c(', paste0('"', estado$models_need_loess, '"', collapse = ", "), ')')
    } else {
      "predictors"
    }
    
    cols_homo <- if (!is.null(estado$sd_cols) && any(estado$sd_is_homoscedastic)) {
      estado$sd_cols[estado$sd_is_homoscedastic]
    } else {
      character(0)
    }
    bloque_zero_homo <- if (length(cols_homo) > 0) {
      paste0(
        "\n# Zero-valued error columns trigger bmetapred's homoscedastic template (sigma ~ 1):\n",
        "for (cz in c(", paste0('"', cols_homo, '"', collapse = ", "), ")) {\n",
        "  df_train[[cz]] <- 0.0\n  df_test[[cz]]  <- 0.0\n}\n"
      )
    } else {
      ""
    }
    
    pmax_script <- calcular_pmax_interno(p_len, input$b_search_mode)
    mc3_script  <- if (!is.null(input$b_mc3) && !is.na(input$b_mc3)) as.integer(input$b_mc3) else 1500L
    n_samp_scr  <- if (!is.null(input$pred_nsamples) && !is.na(input$pred_nsamples)) as.integer(input$pred_nsamples) else 2000L
    
    bloque_loess <- if (strat_loess == "none") {
      paste0('# --- 2. Homoscedastic Mode: No prediction error scales s_ik used ---', bloque_zero_homo)
    } else if (strat_loess == "precomputed") {
      paste0('# --- 2. Using Pre-computed Error Scales (Step 2 fit_LOESS not needed) ---', bloque_zero_homo)
    } else if (strat_loess == "kfold") {
      paste0(
        sprintf(
          '# --- 2. Error Calibration via %d-Fold Cross-Fitting (fit_LOESS) ---
mods_loess <- %s
sd_loess   <- paste0("sd_", mods_loess)
K <- %d
set.seed(%d)
fold_id <- sample(rep(seq_len(K), length.out = nrow(df_train)))
for (sc in sd_loess) df_train[[sc]] <- NA_real_

for (k in seq_len(K)) {
  mod_k  <- fit_LOESS(data = df_train[fold_id != k, ], y = y_col, predictors = mods_loess, span = %.2f)
  pred_k <- predict(mod_k, newdata = df_train[fold_id == k, ], prefix = "sd_")
  for (sc in sd_loess) df_train[fold_id == k, sc] <- pred_k[[sc]]
}

loess_full <- fit_LOESS(data = df_train[, c(y_col, mods_loess)], y = y_col, predictors = mods_loess, span = %.2f)
df_test    <- predict(loess_full, newdata = df_test, prefix = "sd_")',
    as.integer(input$loess_k), mods_loess, as.integer(input$loess_k), as.integer(input$seed_part),
    as.numeric(input$loess_span), as.numeric(input$loess_span)
        ),
    bloque_zero_homo
      )
    } else {
      paste0(
        sprintf(
          '# --- 2. Error Calibration on Disjoint Pre-training Set (fit_LOESS) ---
mods_loess <- %s
loess_mod  <- fit_LOESS(data = df_pretrain, y = y_col, predictors = mods_loess, span = %.2f)
df_train   <- predict(loess_mod, newdata = df_train, prefix = "sd_")
df_test    <- predict(loess_mod, newdata = df_test,  prefix = "sd_")',
          mods_loess, as.numeric(input$loess_span)
        ),
        bloque_zero_homo
      )
    }
    
    lineas_k <- if (input$b_method == "stacking") {
      sprintf("  min_K            = %d,\n  max_K            = %d,\n",
              if (!is.null(input$b_minK) && !is.na(input$b_minK)) as.integer(input$b_minK) else 10L,
              if (!is.null(input$b_maxK) && !is.na(input$b_maxK)) as.integer(input$b_maxK) else 50L)
    } else {
      ""
    }
    
    paste0(
      '# ==============================================================================
# REPRODUCIBLE WORKFLOW GENERATED BY bmetapred
# Date: ', format(Sys.Date(), "%Y-%m-%d"), '
# ==============================================================================
library(bmetapred)
library(scoringRules)

# --- 1. Configuration & Variable Specification ---
y_col      <- "', y_c, '"
predictors <- c(', m_c, ')
sd_cols    <- ', sd_c, '
set.seed(', as.integer(input$seed_part), ')

# (Assume df_train and df_test are loaded and standardized in Y with mu_Y, sd_Y)
', bloque_loess, '

# --- 3. Fit Bayesian Distributional Meta-Model (fit_bmetapred) ---
fit_bm <- fit_bmetapred(
  data_train       = df_train,
  y                = y_col,
  predictors       = predictors,
  sd_cols          = sd_cols,
  method           = "', input$b_method, '",
  p_max_exhaust    = ', pmax_script, ',
  n_iter_mc3       = ', mc3_script, ',
', lineas_k, '  pri_sd           = ', as.numeric(input$b_prisd), ',
  pri_sd_int       = ', as.numeric(input$b_prisd_int), ',
  pri_sd_sigma     = ', as.numeric(input$b_prisd_sigma), ',
  pri_sd_int_sigma = ', as.numeric(input$b_prisd_int_sigma), ',
  parallel_workers = ', as.integer(input$b_workers), ',
  cores_stan       = ', as.integer(input$b_coresstan), ',
  iter_stan        = ', as.integer(input$b_iterstan), ',
  backend          = "', input$b_backend, '",
  verbosity        = 1
)

get_pip_bmetapred(fit_bm)
summary(fit_bm)

# --- 4. Out-of-Sample Meta-Prediction ---
pred_test <- predict(
  fit_bm,
  newdata           = df_test,
  n_samples_totales = ', n_samp_scr, ',
  estimator         = "ensemble"
)
head(pred_test)

# Exact sample-based CRPS over posterior predictive draws
raw_draws <- attr(pred_test, "raw_samples")
mean(scoringRules::crps_sample(y = df_test[[y_col]], dat = t(raw_draws)))
'
    )
  })

output$codigo_r_reproducible <- renderText({
  generar_script_r()
})

# ----------------------------------------------------------------------------
# DOWNLOAD HANDLERS (BLOCK D)
# ----------------------------------------------------------------------------
output$dl_pred_csv <- downloadHandler(
  filename = function() paste0("bmetapred_test_predictions_", Sys.Date(), ".csv"),
  content  = function(file) {
    req(estado$df_pred_final)
    write.csv(estado$df_pred_final, file, row.names = FALSE)
  }
)

output$dl_metrics_csv <- downloadHandler(
  filename = function() paste0("bmetapred_test_metrics_", Sys.Date(), ".csv"),
  content  = function(file) {
    req(estado$df_metrics_comp)
    write.csv(estado$df_metrics_comp, file, row.names = FALSE)
  }
)

output$dl_model_rds <- downloadHandler(
  filename = function() paste0("bmetapred_fitted_model_", Sys.Date(), ".rds"),
  content  = function(file) {
    req(estado$modelo_bm)
    saveRDS(estado$modelo_bm, file)
  }
)

output$dl_r_script <- downloadHandler(
  filename = function() paste0("bmetapred_reproducible_workflow_", Sys.Date(), ".R"),
  content  = function(file) {
    writeLines(generar_script_r(), file)
  }
)
}

# ==============================================================================
# 5. APPLICATION LAUNCHER
# ==============================================================================
shinyApp(ui = ui, server = server)
