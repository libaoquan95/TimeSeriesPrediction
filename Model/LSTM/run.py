import lstm1 as lstm1
import time
import matplotlib.pyplot as plt
import lstm

def plot_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()

def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    #Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
        plt.legend()
    plt.show()

#Main Run Thread
if __name__=='__main__':
    global_start_time = time.time()
    epochs  = 10
    seq_len = 15
    filename = "data/indicator_org_id_day_xxsfjye_958_table_prod.csv"
    lstm = lstm.lstm()
    print('> Loading data... ')

    X_train, y_train, X_test, y_test = lstm.load_data(filename, seq_len, True)
    X_train1, y_train1, X_test1, y_test1 = lstm.load_data(filename, seq_len, False)
    
    print('> Data Loaded. Compiling...')

    model = lstm.buildModel([1, seq_len, 100, 1])

    model.fit(
        X_train,
        y_train,
        batch_size=512,
        nb_epoch=epochs,
        validation_split=0.05)

    #predictions = lstm.predict_sequences_multiple(model, X_test, seq_len, 50)
    #predicted = lstm.predict_sequence_full(model, X_test, seq_len)
    predicted = lstm.predictPointByPoint(model, X_test)        

    print('Training duration (s) : ', time.time() - global_start_time)
    #plot_results_multiple(predictions, y_test, 50)
    x, y_de_real = lstm.de_normalise_windows(X_test1, y_test)
    x, y_de_pred = lstm.de_normalise_windows(X_test1, predicted)
    plot_results(y_de_pred, y_de_real)