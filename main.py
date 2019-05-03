# -*- coding: UTF-8 -*-

from numpy import *
from DependencyParser.DependencyGraph import *


def train_dependency_graph(dict_path, data_path, label_path):
    # read dict
    word_dict = {}
    with open(dict_path, 'r', encoding='utf-8') as file_object:
        for line in file_object:
            temp = line.rstrip('\n').split(" ")
            word_dict[temp[0]] = list(map(float, temp[1].split(",")))

    # read data set
    data_set = []
    with open(data_path, 'r', encoding='utf-8') as file_object:
        for line in file_object:
            data_set.append(line.rstrip('\n').split(" "))

    # read label set
    label_set = []
    with open(label_path, 'r', encoding='utf-8') as file_object:
        for line in file_object:
            label_matrix = []
            temp = line.rstrip('\n').split("|")
            for i in range(len(temp)):
                label_matrix.append(list(map(int, temp[i].split(","))))
            label_set.append(label_matrix)

    # word to embedding
    embedding_set = []
    for data in data_set:
        data_matrix = []
        for word in data:
            data_matrix.append(word_dict[word])
        embedding_set.append(data_matrix)

    # create model
    dependency_graph = DependencyGraph(data_set=embedding_set, label_set=label_set, batch_size=563, hidden_num=256,
                                       classes_num=2,
                                       learning_rate=0.01, train_num=500,
                                       model_path='./LanguageArea/ModelMemory/Dependency/KBDependency/')
    dependency_graph.init_model_parameters()
    dependency_graph.build_graph()
    dependency_graph.train()


if __name__ == '__main__':
    pass
