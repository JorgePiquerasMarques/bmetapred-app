# ==============================================================================
# STANDALONE DESKTOP LAUNCHER FOR THE bmetapred SHINY APPLICATION
# ==============================================================================
options(repos = c(CRAN = "https://cloud.r-project.org"))

required_packages <- c(
  "bmetapred", "shiny", "bslib", "DT", "dplyr", "tidyr", "ggplot2",
  "future", "callr", "scoringRules", "ensembleBMA",
  "e1071", "nnet", "ranger", "rpart", "glmnet"
)

# Use installed.packages() instead of requireNamespace() so Windows does not
# lock rstan/StanHeaders DLLs in memory before a potential version sync
installed_pkgs <- rownames(installed.packages())
missing_pkgs   <- required_packages[!(required_packages %in% installed_pkgs)]

if (length(missing_pkgs) > 0) {
  cat("\nInstalling missing packages (first-time setup only):\n  ",
      paste(missing_pkgs, collapse = ", "), "\n\n", sep = "")
  install.packages(missing_pkgs)
}

# Verify that rstan and StanHeaders share the same major.minor version.
# Only triggers Stan R-universe when CRAN has an out-of-sync pair.
get_ver_mm <- function(pkg) {
  v <- tryCatch(as.character(utils::packageVersion(pkg)), error = function(e) NA_character_)
  if (is.na(v)) return(NA_character_)
  parts <- strsplit(v, "\\.")[[1]]
  paste(parts[1:min(2, length(parts))], collapse = ".")
}

v_rstan   <- get_ver_mm("rstan")
v_headers <- get_ver_mm("StanHeaders")

if (!is.na(v_rstan) && !is.na(v_headers) && !identical(v_rstan, v_headers)) {
  cat(sprintf(
    "\nDetected version mismatch: rstan (%s) vs StanHeaders (%s).\n",
    as.character(utils::packageVersion("rstan")),
    as.character(utils::packageVersion("StanHeaders"))
  ))
  cat("Installing matched rstan & StanHeaders from https://stan-dev.r-universe.dev ...\n\n")
  
  install.packages(
    "rstan",
    repos = c("https://stan-dev.r-universe.dev", "https://cloud.r-project.org")
  )
  install.packages(
    "StanHeaders",
    repos = "https://stan-dev.r-universe.dev"
  )
}

# Open in a clean standalone app window (Chrome/Edge --app mode) when available
desktop_browser <- function(url) {
  sys_os <- Sys.info()[["sysname"]]
  
  if (sys_os == "Darwin") {
    chrome_mac <- "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    edge_mac   <- "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"
    if (file.exists(chrome_mac)) {
      system2(chrome_mac, args = paste0("--app=", url), wait = FALSE)
    } else if (file.exists(edge_mac)) {
      system2(edge_mac, args = paste0("--app=", url), wait = FALSE)
    } else {
      utils::browseURL(url)
    }
  } else if (sys_os == "Windows") {
    chrome_win <- "C:/Program Files/Google/Chrome/Application/chrome.exe"
    edge_win   <- "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"
    if (file.exists(chrome_win)) {
      system2(chrome_win, args = paste0("--app=", url), wait = FALSE)
    } else if (file.exists(edge_win)) {
      system2(edge_win, args = paste0("--app=", url), wait = FALSE)
    } else {
      utils::browseURL(url)
    }
  } else {
    utils::browseURL(url)
  }
}

cat("\n============================================================\n")
cat("  bmetapred Desktop Application is running.\n")
cat("  Keep this window open while using the app.\n")
cat("  Close this window (or press Ctrl+C) to stop the server.\n")
cat("============================================================\n\n")

shiny::runApp(
  appDir         = ".",
  launch.browser = desktop_browser
)