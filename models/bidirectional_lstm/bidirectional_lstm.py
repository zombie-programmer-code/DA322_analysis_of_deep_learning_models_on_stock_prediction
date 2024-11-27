
def bi_lstm(X_train, y_train, X_dev, y_dev, epochs=2, batch_size=128, first_layer=60):
    model = Sequential()
    model.add(Bidirectional(LSTM(units=first_layer, activation = 'tanh', input_shape=(X_train.shape[1], 1))))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['MAE'])

    # Train the model
    model.fit(X_train, y_train, validation_data = (X_dev, y_dev), epochs=epochs)
    
    return model
