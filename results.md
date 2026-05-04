# Patho-bench results

For Virchow2 on 11 CPTAC mutation tasks. Metric: Macro OvR AUC


## Tasks


| Dataset     | Organ     | Mutation | Atlas2 paper | Reproduced (no aug.) | Scanner-transfer aug. | HistAug     | Scanner-transfer + HistAug | HistAug + Scanner-transfer |
|-------------|-----------|----------|--------------|----------------------|-----------------------|-------------|----------------------------|----------------------------|
| CPTAC CCRCC | Kidney    | BAP1     | 67.4         | 62.9 ± 01.8          | 66.5 ± 02.1           | 67.3 ± 02.2 | 65.1 ± 02.6                | 62.0 ± 02.7                |
| CPTAC CCRCC | Kidney    | PBRM1    | 49.4         | 50.5 ± 01.8          | 52.2 ± 01.7           | 51.0 ± 01.6 | 52.7 ± 01.8                | 51.6 ± 01.6                |
| CPTAC LUAD  | Lung      | EGFR     | 81.7         | 81.0 ± 01.4          | 80.3 ± 01.5           | 77.8 ± 01.7 | 66.5 ± 01.8                | 67.2 ± 02.0                |
| CPTAC LUAD  | Lung      | STK11    | 82.4         | 82.3 ± 01.6          | 83.6 ± 01.6           | 79.5 ± 01.7 | 70.6 ± 02.1                | 71.3 ± 02.1                |
| CPTAC LUAD  | Lung      | TP53     | 77.3         | 77.2 ± 01.5          | 77.5 ± 01.5           | 75.9 ± 01.8 | 66.1 ± 01.8                | 67.3 ± 01.7                |
| CPTAC LSCC  | Lung      | KEAP1    | 63.1         | 64.5 ± 02.1          | 65.2 ± 02.2           | 59.6 ± 02.0 | 54.3 ± 02.2                | 51.7 ± 02.4                |
| CPTAC LSCC  | Lung      | ARID1A   | 41.9         | 38.5 ± 02.4          | 43.3 ± 02.2           | 46.7 ± 02.0 | 44.3 ± 02.1                | 45.6 ± 02.1                |
| CPTAC HNSC  | Head&Neck | CASP8    | 56.6         | 58.4 ± 02.6          | 58.4 ± 02.8           | 57.2 ± 03.0 | 61.2 ± 03.1                | 61.1 ± 02.9                |
| CPTAC GBM   | Brain     | EGFR     | 62.1         | 64.0 ± 01.7          | 63.8 ± 01.8           | 58.6 ± 01.9 | 56.8 ± 01.9                | 56.4 ± 02.1                |
| CPTAC GBM   | Brain     | TP53     | 74.8         | 70.2 ± 02.2          | 73.8 ± 01.9           | 65.1 ± 02.0 | 57.5 ± 02.3                | 57.7 ± 02.1                |
| CPTAC PDA   | Pancreas  | SMAD4    | 44.6         | 50.3 ± 02.0          | 51.4 ± 02.3           | 47.3 ± 02.1 | 46.9 ± 02.0                | 46.4 ± 02.0                |


