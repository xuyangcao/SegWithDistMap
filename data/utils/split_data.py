import os
import random

if __name__ == "__main__":
    """ 
        random split data into training set and testing set, 
        the data was split according to the patient_id to avoid
        volumes from same patient in both training set and testing set 

        the total number of patients is 107
        the total number of volume is 170
    """

    # init arguments
    random.seed(2109)
    file_path = './abus_roi/image/'
    train_list = 'abus_train.list.5'
    test_list = 'abus_test.list.5'
    start_idx = 84 
    end_idx = 107 

    # list all the patients
    filenames = os.listdir(file_path)
    filenames.sort()
    #print(len(filenames))
    patient_id = [filename[:4] for filename in filenames]
    patient_id = set(patient_id)
    patient_id = list(patient_id)
    patient_id.sort()
    random.shuffle(patient_id)
    print(patient_id)
    #print(len(patient_id))

    test_id = patient_id[start_idx:end_idx+1]
    train_id = patient_id[0:start_idx] + patient_id[end_idx+1:] 
    print('train_patients: ', len(train_id))
    print('test_patients: ', len(test_id))
    print('train_patients: ', train_id)
    print('test_patients: ', test_id)

    with open(train_list, 'w') as f:
        for filename in filenames:
            if filename[:4] in train_id:
                f.write(filename+'\n')

    with open(test_list, 'w') as f:
        for filename in filenames:
            if filename[:4] in test_id:
                f.write(filename+'\n')
