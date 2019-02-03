from heosImageNet import *
paths = ["C:/Users/Heos/Documents/cifar-10-batches-py/data_batch_4",
         "C:/Users/Heos/Documents/cifar-10-batches-py/data_batch_5",
         "C:/Users/Heos/Documents/cifar-10-batches-py/data_batch_1",
         "C:/Users/Heos/Documents/cifar-10-batches-py/data_batch_2",
         "C:/Users/Heos/Documents/cifar-10-batches-py/data_batch_3"]

test_path = "C:/Users/Heos/Documents/cifar-10-batches-py/test_batch"

ip = Image_Preprocessor()
ip.preprocess(paths, test_data_path=test_path)

train = training.Training(Image_Preprocessor=ip)
train.train()
