#Models suitable for predicting given 10, 30, 60 previous days data. There are 4 models for each type-the average of these 4 gives the prediction. 
def lstm_stock_model(X_train, y_train, X_dev, y_dev, epochs=2, batch_size=64, first_layer=60):
    model = Sequential()
    model.add(LSTM(units=first_layer, activation = 'relu', return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=60, activation = 'relu', return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(units=80, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['MAE'])

    # Train the model
    model.fit(X_train, y_train, validation_data = (X_dev, y_dev), epochs=epochs)
    
    return model
