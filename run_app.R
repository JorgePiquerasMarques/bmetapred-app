# ==============================================================================
# STANDALONE DESKTOP LAUNCHER FOR THE bmetapred SHINY APPLICATION
# ==============================================================================
options(repos = c(CRAN = "https://cloud.r-project.org"))

required_packages <- c(
  "bmetapred", "shiny", "bslib", "DT", "dplyr", "tidyr", "ggplot2",
  "future", "callr", "scoringRules", "ensembleBMA",
  "e1071", "nnet", "ranger", "rpart", "glmnet"
)

missing_pkgs <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]

if (length(missing_pkgs) > 0) {
  cat("\nInstalling missing packages (first-time setup only):\n  ",
      paste(missing_pkgs, collapse = ", "), "\n\n", sep = "")
  install.packages(missing_pkgs)
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