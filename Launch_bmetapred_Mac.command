#!/bin/bash
cd "$(dirname "$0")"

if command -v Rscript >/dev/null 2>&1; then
  RSCRIPT_BIN="Rscript"
elif [ -x "/Library/Frameworks/R.framework/Resources/bin/Rscript" ]; then
  RSCRIPT_BIN="/Library/Frameworks/R.framework/Resources/bin/Rscript"
else
  echo "ERROR: R is not installed on this computer."
  echo "Please install R from https://cran.r-project.org/ and try again."
  read -p "Press Enter to close..."
  exit 1
fi

"$RSCRIPT_BIN" run_app.R
