import pickle


def get_batch_nums(directory):
    """
    This function reads the total number of batches processed for each partition.
    For example, if batch_nums['train'][0] = 10, then Colab has processed
    batches [0, 10) for the train1 partition of the dataset.

    :param directory: the directory which holds all of the data from Colab
    :return: a dictionary with the number batches processed by Colab
    """
    batch_nums = {'train': []}
    partition = 'train'
    for i in range(1, 4):
        with open(f'{directory}/{partition}{i}_startup.pkl', mode='rb') as file:
            _, batch_num = pickle.load(file)
        batch_nums[partition].append(batch_num)
    for partition in ['validate', 'test']:
        with open(f'{directory}/{partition}_startup.pkl', mode='rb') as file:
            _, batch_nums[partition] = pickle.load(file)
    print(batch_nums)


if __name__ == '__main__':
    get_batch_nums('./data')
