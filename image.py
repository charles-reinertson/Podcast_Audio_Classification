import os
import numpy as np
import matplotlib.pyplot as plt




def main():
    plt.ion()
    fig, axes = plt.subplots(1,2, figsize=(10,5))
    plt.suptitle('Validation')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')

    epoch = 40
    epoch1 = 9


    stats_audio_features = np.loadtxt('stats_audio_features.csv', delimiter=',')
    stats_emotion_features = np.loadtxt('stats_emotion_features.csv', delimiter=',')
    stats_bag_of_words_features = np.loadtxt('stats_bag_of_words_features.csv', delimiter=',')


    valid_acc0 = [s[0] for s in stats_audio_features]
    valid_loss0 = [s[1] for s in stats_audio_features]
    valid_acc1 = [s[0] for s in stats_emotion_features]
    valid_loss1 = [s[1] for s in stats_emotion_features]
    valid_acc2 = [s[0] for s in stats_bag_of_words_features]
    valid_loss2 = [s[1] for s in stats_bag_of_words_features]

    axes[0].plot(range(epoch - len(stats_audio_features) + 1, epoch + 1), valid_acc0,
        linestyle='--', marker='o', color='b')
    axes[0].plot(range(epoch - len(stats_emotion_features) + 1, epoch + 1), valid_acc1,
        linestyle='--', marker='o', color='g')
    axes[0].plot(range(epoch1 - len(stats_bag_of_words_features) + 1, epoch1 + 1), valid_acc2,
        linestyle='--', marker='o', color='r')

    axes[0].legend(['Audio Features', 'Emotion Features', 'Bag of Words'])

    axes[1].plot(range(epoch - len(stats_audio_features) + 1, epoch + 1), valid_loss0,
        linestyle='--', marker='o', color='b')
    axes[1].plot(range(epoch - len(stats_emotion_features) + 1, epoch + 1), valid_loss1,
        linestyle='--', marker='o', color='g')
    axes[1].plot(range(epoch1 - len(stats_bag_of_words_features) + 1, epoch1 + 1), valid_loss2,
        linestyle='--', marker='o', color='r')

    axes[1].legend(['Audio Features', 'Emotion Features', 'Bag of Words'])
    plt.pause(0.00001)


    fig.savefig('_training_plot_combinded.png', dpi=200)
    plt.ioff()
    plt.show()



    
    


if __name__ == "__main__":
    main()
