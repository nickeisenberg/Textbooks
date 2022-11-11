import os, shutil

path_old = '/Users/nickeisenberg/GitRepos/Textbooks/DeepLearningWithPython_Chollet/Chapter8/DataSets/dogs-vs-cats/train/'
path_new = '/Users/nickeisenberg/GitRepos/Textbooks/DeepLearningWithPython_Chollet/Chapter8/DataSets/dogs_vs_cats_small/'

def make_subset(subset_name, start_ind, end_ind):

    for category in ['dog', 'cat']:
        new_dir = path_new + subset_name + '/' + category + '/'
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)

        files_to_copy = []
        for i in range(start_ind, end_ind):
            files_to_copy.append(f'{category}.{i}.jpg')

        files = os.listdir(path_old)
        for jpg in files_to_copy:
            shutil.copy(path_old + jpg, new_dir + jpg)

if __name__ == '__main__':

    make_subset('train', 0, 1000)
    make_subset('validation', 1000, 1500)
    make_subset('test', 1500, 2500)




