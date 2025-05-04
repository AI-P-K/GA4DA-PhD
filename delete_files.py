import os
import sys
result_root_folder = 'results'

for test in os.listdir(result_root_folder):
    if "T_" in test:
        all_solutions_in_test_folder = sorted(os.listdir(os.path.join(result_root_folder, test)))
        # try:
        #     all_solutions_in_test_folder.remove('Plots')
        #     print('Plots removed from all_solutions_in_test_folder')
        # except:
        #     print("Nothing to remove")
        full_path_solutions_folder = [os.path.join(result_root_folder, test, i) for i in all_solutions_in_test_folder]
        full_path_solutions_folder = [item for item in full_path_solutions_folder if "description.txt" not in item]
        # print(full_path_solutions_folder)
        for run in full_path_solutions_folder:
        	# print(os.listdir(run))
        	files = os.listdir(run)
        	files_full_path = [os.path.join(run, i) for i in files]
        	# print(files_full_path)
        	for file in files_full_path:
        		# print(file)
        		if ".png" in file:
        			# print(file)
        			os.remove(file)

