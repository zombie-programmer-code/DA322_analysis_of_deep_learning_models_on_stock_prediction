#Models suitable for predicting given 10, 30 previous days data. There are 4 models for each type-the average of these 4 gives the prediction.
def bilstm_gru_hybrid_stock_model(X_train, y_train, X_dev, y_dev, epochs=2, batch_size=128, first_layer=40):
    model = Sequential()
    model.add(Bidirectional(LSTM(units=first_layer, activation = 'tanh', return_sequences=True, input_shape=(X_train.shape[1], 1))))
    model.add(Dropout(0.2))
    model.add(GRU(units=60, activation = 'relu', return_sequences=False))
    model.add(Dropout(0.5))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['MAE'])

    model.fit(X_train, y_train, validation_data = (X_dev, y_dev), epochs=epochs)
    
    return model
