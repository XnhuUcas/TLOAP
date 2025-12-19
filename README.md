# Model aggregation-assisted transfer learning framework for generalized semiparametric models

This repository is the implementation of "*Model aggregation-assisted transfer learning framework for generalized semiparametric models*". This repository provides a multi-source transfer learning procedure for prediction via frequentist model averaging under generalized partially linear varying-coefficient models. 

Any questions or comments, please don’t hesitate to contact with me any time.

**File Structure:**

- **`paper_codes`**  
  codes for reproduction of all numerical results in our paper:
  - main_Figure2_Figure3.R: results of Fig.2 and Fig.3 in main text;
  - main_Figure4_supp_FigureS6_FigureS7_TableS12.R: results of Fig.4 in main text and Fig.S.6, Fig.S.7, and Table.S.12 in the Supplementary Materials;
  - main_Figure5_supp_FigureS8_TableS12.R: results of Fig.5 in main text and Fig.S.8 and Table.S.12 in the Supplementary Materials;
  - main_Table1_Table2_Table3_supp_FigureS2_TableS11.R: results of Table 1-3 in main text and Fig.S.2 and Table.S.11 in the Supplementary Materials;
  - supp_FigureS9_FigureS11.R: results of Fig.S.9 and Fig.S.11 in the Supplementary Materials;
  - supp_FigureS10_FigureS12.R: results of Fig.S.10 and Fig.S.12 in the Supplementary Materials;
  - supp_TableS1_TableS2_TableS3_FigureS4.R: results of Table S.1-S.3 and Fig.S.4 in the Supplementary Materials;
  - supp_TableS4_TableS5_TableS6_TableS11_FigureS3.R: results of Table S.4-S.6, Table S.11 and Fig.S.3 in the Supplementary Materials;
  - supp_TableS7_TableS8_TableS9_FigureS5.R: results of Table S.7-S.9 and Fig.S.5 in the Supplementary Materials;
  - supp_TableS10.R: results of Table S.10 in the Supplementary Materials.

- **`simdata.gen.R`**
- **`pred.tloap.R`**
- **`example.R`**  


## Requirements

First of all, make sure you have installed the R language environment (it is recommended to use R version 4.1 or higher).

In R, run the following command to install the required packages:

```r
install.packages(c("glasso", "Matrix", "igraph"))
```

## Data Generation

## Training

To train the model in the paper, run the entire **`pred.tloap.R`** script. 

## Evaluation

To evaluate my model, run:

```eval

```

