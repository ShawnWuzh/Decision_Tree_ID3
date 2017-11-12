import pandas as pd

from math import log


def calculateEntropy(data):         # the data is a pandas data frame
    '''
    This function is to calculate the information shannon entropy of the
    data set.
    '''
    num_of_entries = data.shape[0]
    # this is the number of entries in the data frame
    label_frequency = {}
    # this is used for counting the frequency of each label
    shannon_entropy = 0
    for i in range(num_of_entries):
        current_entry = data.iloc[i]
        current_label = current_entry[-1]
        if current_label not in label_frequency.keys():
            label_frequency[current_label] = 0
        label_frequency[current_label] += 1
    for label in label_frequency:
        prob = float(label_frequency[label]) / num_of_entries
        shannon_entropy -= prob * log(prob, 2)
    return shannon_entropy


def BestFeature(data):
    '''
    This function is used to select the feature with maximum information gain
    to split the data
    '''
    features = data.columns
    feature_indexes = [i for i in range(len(features) - 1)]
    maximum_information_gain = 0
    # the initial value of the maximum_infortion gain is 0
    feature_information_entropy = 0
    maximum_feature = -1
    for feature_index in feature_indexes:
        feature_information_entropy = 0
        feature_label = features[feature_index]
        feature_values = data[feature_label].unique()
        for value in feature_values:
            sub_data = data[data[feature_label] == value]
            feature_information_entropy += (len(sub_data) / data.shape[0]) * \
                calculateEntropy(sub_data)
        information_gain = calculateEntropy(data) - feature_information_entropy
        if information_gain >= maximum_information_gain:
            maximum_information_gain = information_gain
            maximum_feature = feature_index
    return features[maximum_feature]


def create_ID3_tree(data):
    '''
    this function is for constructing the ID3 tree
    '''
    data_labels = data.iloc[:, -1].unique()
    features = data.columns
    if len(data_labels) == 1:
        # if the labels in the data set are all the same, then there is no need to
        # split the data
        return data_labels[0]
    if len(features) == 1 or data.iloc[:, :-1].drop_duplicates().shape[0] == 1:
        '''
        if there are no more features,then just return the label appears most
        often or all the samples have the same value of all the features, also
        return the label appears most
        '''
        return maximum_label(data)
    best_feature = BestFeature(data)
    id3_tree = {best_feature: {}}
    for value in data[best_feature].unique():
        subset = data[data[best_feature] == value].copy()
        id3_tree[best_feature][value] = create_ID3_tree(
            subset.drop(best_feature, axis=1))
    return id3_tree
