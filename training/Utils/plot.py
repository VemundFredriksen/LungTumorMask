import csv
import matplotlib.pyplot as plt
import numpy as np

def _decode_train_csv(csv_path):
    
    epochs = []
    train_loss = []
    val_loss = []
    dice = []

    with open(csv_path) as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            epochs.append(row['step'])
            train_loss.append(row['train_loss'])
            val_loss.append(row['val_loss'])
            dice.append(row['dice_score'])
    
    return (np.array(epochs, dtype=np.uint), np.array(train_loss, dtype=np.float32), 
    np.array(val_loss, dtype=np.float32), np.array(dice, dtype=np.float32))

def plot_train_data(csv_path, store = None, show=True, steps_in_epoch = -1):
    data = _decode_train_csv(csv_path)

    plt.plot(data[0], data[1], label = 'Training Loss')
    plt.plot(data[0], data[2], label = 'Validation loss')
    plt.plot(data[0], data[3], label = 'Dice Score')

    
    if(steps_in_epoch > 0):
        vlines = [x for x in range(0, data[0][-1]) if x % steps_in_epoch == 0]
        plt.vlines(vlines, ymin = -0.2, ymax = -0.05)

    plt.ylim(-0.1, 1.1)
    plt.ylabel('Training loss')
    plt.xlabel('Train Step')
    plt.legend(loc="upper left")
    if(store):
        plt.savefig(store)
    if(show):
        plt.show()

def plot_multiple_val_losses(names, csvs):
    for name, csv in zip(names, csvs):
        data = _decode_train_csv(csv)
        plt.plot(data[0], data[2], label = name)

    plt.ylim(-0.1, 1.1)
    plt.xlim(0, 7000)
    plt.ylabel('Validation Loss')
    plt.xlabel('Train Step')
    plt.legend(loc="upper left")
    plt.show()

if __name__ == "__main__":
    #path = "D:\\Repos\\LungTumorSegmentation\\models\\metrics.csv"
    #plot_train_data(path)
    names = ['Base 16: Multiplier: 2x', 'Base 64: Multiplier: 2x', 'Base 128: Multiplier: 2x', 'Base 64: Multiplier: 3.5x', 'Base 192: Multiplier: 1.5x']
    csvs = ["C:\\Users\\vemun\\Desktop\\Plots\\16_2.csv", "C:\\Users\\vemun\\Desktop\\Plots\\64_2.csv", "C:\\Users\\vemun\\Desktop\\Plots\\128_2.csv", "C:\\Users\\vemun\\Desktop\\Plots\\64_3_5.csv", "C:\\Users\\vemun\\Desktop\\Plots\\192_1_5.csv"]
    plot_multiple_val_losses(names, csvs)