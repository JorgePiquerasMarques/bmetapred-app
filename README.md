# `bmetapred` Desktop Application

Interactive graphical user interface for the [**`bmetapred`**](https://CRAN.R-project.org/package=bmetapred) R package, implementing **Bayesian Stacking of Linear Distributional Meta-Models under Heteroscedastic Prediction Errors**.

This standalone application allows researchers and practitioners to run the complete `bmetapred` workflow without writing R code or opening RStudio, while also providing built-in presets to reproduce the numerical studies and meteorological application from the accompanying manuscript.

---

## Features

The interface is organized into four sequential modules:

1. **Data & Preprocessing:** Load custom `.csv` / `.rds` datasets. Also, loads the manuscript's heteroscedastic simulation (Demo 1) or the full Pacific Northwest numerical weather prediction dataset `srft` (Demo 2). Supports four prediction-error specifications: full `fit_LOESS` calibration, pre-computed error columns, homoscedastic mode (no prediction errors), and mixed per-model assignment.
2. **Error Calibration (`fit_LOESS`):** Estimate observation-specific error-scale proxies ($s_{ik}$) when base models do not incluve native prediction errors. Can be implemented via $K$-fold cross-fitting, a disjoint pre-training split, or an external calibration file, with interactive residual funnel and calibration diagnostics.
3. **Meta-Model Fitting (`fit_bmetapred`):** Configure Bayesian Stacking or Pseudo-BMA, prior hyperparameters, exhaustive ($p \le 9$) or parallel-tempering Tabu heuristic search ($p > 9$), Stan backend (`rstan` or `cmdstanr`), and monitor progress in real time without freezing the interface.
4. **Meta-Prediction, Evaluation & Reproducible Export:** Generate out-of-sample posterior predictive draws, compute exact sample-based proper scoring rules (**CRPS** via `scoringRules` and **MWIS**), inspect interactive diagnostic figures, export predictions/models, and download a self-contained reproducible R script.

---

## Prerequisites

**RStudio is not required.** However, because `bmetapred` compiles distributional regression templates in Stan (`brms`) at runtime, your computer needs:

1. **R** (version 4.1.0 or higher): [https://cran.r-project.org/](https://cran.r-project.org/)
2. **A C++ toolchain** (required once to compile Stan models):
   - **Windows:** Install [Rtools](https://cran.r-project.org/bin/windows/Rtools/) matching your R version.
   - **macOS:** Open Terminal and run `xcode-select --install` (follow the prompt to install Apple Command Line Tools).
   - **Linux:** Ensure `build-essential` / `g++` is installed.

> **Optinal but recommended:** For faster MCMC sampling in high-dimensional ensembles ($p > 9$), we recommend installing [`cmdstanr`](https://mc-stan.org/cmdstanr/) and selecting `cmdstanr` as the backend in Tab 3.

---

## Quick Start

### Option 1: Standalone Desktop Launcher (No RStudio Required)

1. Click the green **`<> Code`** button at the top of this page and select **Download ZIP**.
2. Unzip the downloaded folder (`bmetapred-app-main`).
3. Launch the app by double-clicking the script for your operating system:
   - **Windows:** Double-click **`Launch_bmetapred_Windows.bat`**.
   - **macOS:** Double-click **`Launch_bmetapred_Mac.command`**.

On the **first launch only**, `run_app.R` will automatically install `bmetapred` and any missing GUI dependencies from CRAN. Subsequent launches start immediately in a standalone application window.

#### First-Time Security Note for macOS Users
Because `Launch_bmetapred_Mac.command` is downloaded from the internet, macOS Gatekeeper may block it on the first double-click:
- **Right-click** (or `Control` + click) on `Launch_bmetapred_Mac.command` and choose **Open → Open**.
- If macOS reports a permission error, open Terminal inside the unzipped folder and run once:
  ```bash
  chmod +x Launch_bmetapred_Mac.command
  ```

### Option 2: Run Directly from R

If you already have R open, you can launch the app directly from GitHub without manually downloading the repository:

```r
install.packages("shiny") # if not already installed
shiny::runGitHub("bmetapred-app", "JorgePiquerasMarques")
```

---

## Reproducing the Manuscript Examples

- **Demo 1 (Heteroscedastic Simulation):** In Tab 1, select *Demo 1: Heteroscedastic Simulation*, set `N_oracle = 20000`, and choose *2. Pre-computed Error Columns in Dataset* to use the exact Oracle LOESS error scales from the manuscript (Step 2 is skipped automatically).
- **Demo 2 (Meteorology — `srft`):** In Tab 1, select *Demo 2: Real Meteorology (`srft` - `ensembleBMA`)* to load all complete forecast–observation pairs across the $p = 8$ UWME ensemble members, then calibrate `fit_LOESS` (`span = 0.40`) via a disjoint pre-training split.

---

## Citation

If you use `bmetapred` or this application in your research, please cite:

- **Manuscript:** Piqueras-Marqués, J., Martínez-Minaya, J., & Forte Deltell, A. (2026). *Bayesian Stacking of Linear Distributional Meta-Models under Heteroscedastic Prediction Errors: The `bmetapred` Method*. Manuscript submitted for publication.
- **R Package:** Piqueras-Marqués, J., Martínez-Minaya, J., & Forte Deltell, A. `bmetapred`: Bayesian Stacking of Linear Distributional Meta-Models with Prediction Errors. Available on CRAN: [https://CRAN.R-project.org/package=bmetapred](https://CRAN.R-project.org/package=bmetapred).

---

## Funding

This work was supported by **Fundación BBVA**.