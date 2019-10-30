from SpeechRecognition.AcousticModel.dfsmn_v2.Debug import dfsmn_model_train

if __name__ == '__main__':
    label_path = r'SpeechRecognition\AcousticModel\dfsmn_v1\LabelData\\'
    data_path = r'D:\workspace\datas\\'
    dfsmn_model_train(data_path, label_path)