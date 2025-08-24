# Index
- `requirements.txt` \- List of required libraries and dependencies for the SSL project.
- Used Python 3.12 on a Macbook Air M2
- One note is that pymer4, which interfaces with R, is finicky. It's best to make sure 
the following R packages are installed:

```r
install.packages(c(
  "tibble",
  "lme4",
  "lmerTest",
  "emmeans",
  "broom",
  "broom.mixed",
  "report"
))


