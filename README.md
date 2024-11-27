# Analysis of ML and DL models for the task of stock prediction

# Stock Prediction Models

## Overview
This repository explores and implements various **Machine Learning (ML)** and **Deep Learning (DL)** models for predicting stock prices. The project focuses on evaluating the performance of these models across different indices and stock sectors, using a combination of standard and novel hybrid approaches.

## Key Features
- **Indices Analyzed**:
  - Dow Jones
  - S&P 500
  - Nasdaq-100 (IXIC)
  - Wilshire 5000 (W5000)
  - XLK (Technology Select Sector ETF)

- **Sectors Included**:  
  - **Healthcare**: AMGN, ABBV, ABT  
  - **Technology**: INTC, AMZN, META  
  - **Automotive**: TSLA, GM, F  
  - **Industrial**: GE, BA, RTX  

- **Models Implemented**:  
  - LSTM (Long Short-Term Memory)  
  - GRU (Gated Recurrent Unit)  
  - Bidirectional LSTM (BiLSTM)  
  - LSTM-GRU Hybrid  
  - BiLSTM-GRU Hybrid  
  - CNN-LSTM Hybrid  
  - GRU-CNN Hybrid  
  - BiLSTM-CNN Hybrid  

---

## Methodology
1. **Time-Series Forecasting**:  
   - Predicting closing prices using sequential models.  
   - Models are trained using historical input window sizes (\(k\) = 10, 30, 60).  

2. **Hybrid Model Innovations**:  
   - Combining CNN with RNN-based architectures (LSTM and GRU) for feature extraction and sequential pattern learning.  
   - Integrating Bidirectional LSTMs for richer contextual understanding.  

3. **Performance Metrics**:  
   - **Mean Absolute Error (MAE)** to evaluate accuracy.  
   - Sector-specific trends and optimal model selection are analyzed.  

---

## Results

### Indices Performance
- **Best Overall Model**: **Bidirectional LSTM (BiLSTM)** achieved the lowest MAE values across most indices, particularly with shorter windows (\(k = 10\)).  
- **Notable Contender**: **CNN-LSTM Hybrid** performed competitively for indices like IXIC and XLK, benefiting from longer windows (\(k = 60\)).  

### Sector-Specific Insights
#### Healthcare
- **BiLSTM** consistently outperformed others with the lowest MAE values for AMGN, ABBV, and ABT (\(k = 10\)).  
- **CNN-LSTM Hybrid** excelled for ABT with an MAE of 1.42 (\(k = 30\)).  

#### Technology
- **BiLSTM** achieved the best performance for INTC and AMZN (\(k = 10\)).  
- For META, **CNN-LSTM Hybrid** showed promise (\(k = 30\)).  

#### Automotive
- **BiLSTM** excelled across TSLA, GM, and F with the best MAE values (\(k = 10\)).  
- **CNN-LSTM Hybrid** showed competitive performance for TSLA and F (\(k = 30\)).  

#### Industrial
- **BiLSTM** and **CNN-LSTM Hybrid** performed well across GE, BA, and RTX.  
- BA showed a preference for longer windows (\(k = 30\)) with hybrid models.  

---

## Future Work
1. **Incorporating Attention Mechanisms**:  
   - Attention layers can help the model focus on critical time steps for better accuracy.  

2. **Advanced Hyperparameter Tuning**:  
   - Use Bayesian optimization or genetic algorithms to refine configurations.  

3. **Transformer Architectures**:  
   - Explore LSTM-Transformer hybrids for capturing long-term dependencies.  

4. **Incorporating Large Language Models (LLMs)**:  
   - Include financial news sentiment analysis to complement numerical predictions.  

5. **Sector-Specific Fine-Tuning**:  
   - Tailor models to sector-specific characteristics.  

6. **Real-Time Systems**:  
   - Build scalable prediction pipelines for high-frequency trading with tools like Kafka or Spark.  

7. **Explainability**:  
   - Use SHAP or LIME for interpretable predictions in financial contexts.  

8. **Global Markets**:  
   - Expand analysis to global indices and multi-currency datasets.  

---

## Repository Structure
- `models/`: Implementations of ML/DL models.  
- `data/`: Preprocessed stock data and historical datasets.  
- `notebooks/`: Jupyter notebooks for experiments and visualizations.  
- `results/`: Tables and prediction plots comparing MAE across models and sectors.  


