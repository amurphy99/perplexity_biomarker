# Perplexity Speech Biomarker


### Transformer-Based Results
| Model      | Win  | Step | Time | Aggr   | MinTok | \| | Score  | Outliers | \|  | Welch p | M-W u   | Hedges g | Cliff's δ |
|-----------:|:----:|:----:|:----:|:------:|:--:|:--:|:----:|:----:|:--:|:-------:|:-------:|:--------:|:--------|
| gpt2-medium | 256 | 16 | 00:46  | winsor | 3 | \| | aNLL | none | \| |  0.1142 |  0.0252 | -0.2317 | -0.1899 | 
| gpt2-medium | 256 | 16 | 00:46  | winsor | 3 | \| | PPL | none | \| |  0.4121 |  0.0252 | -0.1242 | -0.1899 | 
| gpt2-medium | 256 | 16 | 00:46  | winsor | 3 | \| | aNLL | trim | \| |  0.0050 |  0.0070 | -0.4396 | -0.2436 | 
| gpt2-medium | 256 | 16 | 00:46  | winsor | 3 | \| | PPL | trim | \| |  0.0065 |  0.0070 | -0.4140 | -0.2436 | 



