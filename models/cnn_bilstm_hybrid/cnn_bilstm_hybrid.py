#Models suitable for predicting given 10, 30, 60 previous days data. There are 4 models for each type-the average of these 4 gives the prediction.
def cnn_bilstm_stock_model(X_train, y_train, X_dev, y_dev, epochs=2, batch_size=128, filters=32, kernel_size=3, pool_size=2, lstm_units=50):
    model = Sequential()
    
    # Convolutional Layer for feature extraction
    model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(MaxPooling1D(pool_size=pool_size))
    
    # LSTM Layer for sequential modeling
    model.add(Bidirectional(LSTM(units=lstm_units, activation='tanh', return_sequences=False)))
    model.add(Dropout(0.2))
    
    # Dense Output Layer
    model.add(Dense(units=1, activation='linear'))  # Regression output

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

    model.fit(X_train, y_train, validation_data=(X_dev, y_dev), epochs=epochs, batch_size=batch_size)

    return model
