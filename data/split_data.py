import os
import random

if __name__ == "__main__":
    """ 
        random split data into training set and testing set, 
        the data was split according to the patient_id to avoid
        volumes from same patient in both training set and testing set 
    """

    random.seed(2109)

    filenames = os.listdir('./abus_data/image/')
    filenames.sort()
    #print(len(filenames))

    patient_id = [filename[:4] for filename in filenames]
    patient_id = set(patient_id)
    patient_id = list(patient_id)
    patient_id.sort()
    random.shuffle(patient_id)
    #print(patient_id)
    #print(len(patient_id))

    test_id = patient_id[:20]
    train_id = patient_id[20:]
    print('train_patients: ', len(train_id))
    print('test_patients: ', len(test_id))
    print('train_patients: ', train_id)
    print('test_patients: ', test_id)

    with open('abus_train.list', 'w') as f:
        for filename in filenames:
            if filename[:4] in train_id:
                f.write(filename+'\n')

    with open('abus_test.list', 'w') as f:
        for filename in filenames:
            if filename[:4] in test_id:
                f.write(filename+'\n')
