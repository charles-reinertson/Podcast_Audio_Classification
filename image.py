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
    epoch2 = 41
    epoch3 = 16


    stats_audio_features = np.loadtxt('results/stats_audio_features.csv', delimiter=',')
    stats_emotion_features = np.loadtxt('results/stats_emotion_features.csv', delimiter=',')
    stats_bag_of_words_features = np.loadtxt('results/stats_bag_of_words.csv', delimiter=',')
    stats_test_bert_pooled_features = np.loadtxt('results/stats_bert_pooled.csv', delimiter=',')
    stats_test_bert_features = np.loadtxt('results/stats_bert.csv', delimiter=',')
    stats_test_distilbert_features = np.loadtxt('results/stats_distilbert.csv', delimiter=',')
    stats_test_bag_and_bert_features = np.loadtxt('results/stats_bag+bert+weights.csv', delimiter=',')


    valid_acc0 = [s[0] for s in stats_audio_features]
    valid_loss0 = [s[1] for s in stats_audio_features]

    valid_acc1 = [s[0] for s in stats_emotion_features]
    valid_loss1 = [s[1] for s in stats_emotion_features]

    valid_acc2 = [s[0] for s in stats_bag_of_words_features]
    valid_loss2 = [s[1] for s in stats_bag_of_words_features]

    valid_acc3 = [s[0] for s in stats_test_bert_pooled_features]
    valid_loss3 = [s[1] for s in stats_test_bert_pooled_features]

    valid_acc4 = [s[0] for s in stats_test_bert_features]
    valid_loss4 = [s[1] for s in stats_test_bert_features]

    valid_acc5 = [s[0] for s in stats_test_distilbert_features]
    valid_loss5 = [s[1] for s in stats_test_distilbert_features]

    valid_acc6 = [s[0] for s in stats_test_bag_and_bert_features]
    valid_loss6 = [s[1] for s in stats_test_bag_and_bert_features]

    axes[0].plot(range(epoch - len(stats_audio_features) + 1, epoch + 1), valid_acc0,
        linestyle='--', marker='o', color='b')

    axes[0].plot(range(epoch - len(stats_emotion_features) + 1, epoch + 1), valid_acc1,
        linestyle='--', marker='o', color='g')

    axes[0].plot(range(epoch1 - len(stats_bag_of_words_features) + 1, epoch1 + 1), valid_acc2,
        linestyle='--', marker='o', color='r')

    axes[0].plot(range(epoch2 - len(stats_test_bert_pooled_features) + 1, epoch2 + 1), valid_acc3,
        linestyle='--', marker='o', color='c')

    axes[0].plot(range(epoch3 - len(stats_test_bert_features) + 1, epoch3 + 1), valid_acc4,
        linestyle='--', marker='o', color='m')

    axes[0].plot(range(epoch3 - len(stats_test_distilbert_features) + 1, epoch3 + 1), valid_acc5,
        linestyle='--', marker='o', color='y')
    
    axes[0].plot(range(epoch3 - len(stats_test_bag_and_bert_features) + 1, epoch3 + 1), valid_acc6,
        linestyle='--', marker='o', color='k')

    axes[0].legend(['Audio Features', 'Emotion Features', 'Bag of Words', 'Bert Pooled', 'Bert', 'Distilbert', 'Bag of Words + Bert'])

    axes[1].plot(range(epoch - len(stats_audio_features) + 1, epoch + 1), valid_loss0,
        linestyle='--', marker='o', color='b')

    axes[1].plot(range(epoch - len(stats_emotion_features) + 1, epoch + 1), valid_loss1,
        linestyle='--', marker='o', color='g')

    axes[1].plot(range(epoch1 - len(stats_bag_of_words_features) + 1, epoch1 + 1), valid_loss2,
        linestyle='--', marker='o', color='r')

    axes[1].plot(range(epoch2 - len(stats_test_bert_pooled_features) + 1, epoch2 + 1), valid_loss3,
        linestyle='--', marker='o', color='c')

    axes[1].plot(range(epoch3 - len(stats_test_bert_features) + 1, epoch3 + 1), valid_loss4,
        linestyle='--', marker='o', color='m')

    axes[1].plot(range(epoch3 - len(stats_test_distilbert_features) + 1, epoch3 + 1), valid_loss5,
        linestyle='--', marker='o', color='y')
    
    axes[1].plot(range(epoch3 - len(stats_test_bag_and_bert_features) + 1, epoch3 + 1), valid_loss6,
        linestyle='--', marker='o', color='k')

    axes[1].legend(['Audio Features', 'Emotion Features', 'Bag of Words', 'Bert Pooled', 'Bert', 'Distilbert', 'Bag of Words + Bert'])
    plt.pause(0.00001)


    fig.savefig('_training_plot_combinded.png', dpi=200)
    plt.ioff()
    plt.show()



    
    


if __name__ == "__main__":
    main()
