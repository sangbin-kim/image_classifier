import pandas as pd
import os
import shutil
import functions

# 1. Directory Setting
save_directory = r'D:/#.Secure Work Folder/1. Data/1. CMI/1. FLD/test/'
image_directory = r"D:\#.Secure Work Folder\1. Data\1. CMI\1. FLD\prac"

# 2. Create DataFrame for results
result_df = pd.DataFrame(columns= ['ID', 'FILE', '1', '2', '3', 'JUDGE' ])

# 3. load trained model
model_save_path = r"C:/Users/LG/Desktop/ksb/3. CODE/DeepLearning/"
filename = 'n2_2_wo_ok_best.pth'
model = functions.load_checkpoint(model_save_path, filename)

# 4. Classfiying images and reorganize image based on the JUDGE from model

for root, dirs, files in os.walk(image_directory):

    for file in files:

        if os.path.isdir(save_directory + root[-13:]) == False:
            os.mkdir(save_directory + root[-13:])
        if file.endswith("(0).JPG"):
            print(root[-13:])
            print(file)
            probs, classes = functions.predict(os.path.join(root, file), model)
            result_df = result_df.append({'ID': root[-13:],
                                          'FILE': file,
                                          '{}'.format(classes[0]): probs[0],
                                          '{}'.format(classes[1]): probs[1],
                                          '{}'.format(classes[2]): probs[2],
                                          'JUDGE': classes[0]}, ignore_index=True)
            judge = 'JUDGE_{}_'.format(classes[0]) + str(round(int(probs[0] * 100), 0)) + "_" + file
            print(judge)
            shutil.copy(os.path.join(root, file), save_directory + '/' + root[-13:] + '/' + judge)


# 5. Save results and judges
result_df.to_csv( save_directory+ 'result.csv', index= False)
result_df.groupby(['ID', 'JUDGE']).size().to_frame('count').reset_index().to_csv(save_directory + "judge.csv", index = False)