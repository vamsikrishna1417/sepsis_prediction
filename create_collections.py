import os
import pandas as pd
import pickle

headers = "HR|O2Sat|Temp|SBP|MAP|DBP|Resp|EtCO2|BaseExcess|HCO3|FiO2|pH|PaCO2|SaO2|AST|BUN|Alkalinephos|Calcium|Chloride|Creatinine|Bilirubin_direct|Glucose|Lactate|Magnesium|Phosphate|Potassium|Bilirubin_total|TroponinI|Hct|Hgb|PTT|WBC|Fibrinogen|Platelets|Age|Gender|Unit1|Unit2|HospAdmTime|ICULOS|SepsisLabel"
columns = headers.split('|')
patient_list = []
path = "./training_data/"
files = []
selected_ind = [0,1,2,3,4,5,6,7,10,19,22,26,31,33,34,40]
columns = [columns[i] for i in selected_ind]
print(columns)
dfs_list = []
X_train = []
Y_train = []
for r , d, f in os.walk(path):
	for file in f:
		files.append(os.path.join(r, file))

for file in files:
	file_handle = open(file,'r')

	lineList = file_handle.readlines()
	del lineList[0]
	patient_dict = {}
	hourly_dict = {}
	index = 0
	patient_id = file.split('.')[1].split('/')[2]
	list_filtered_data= []
	for line in  lineList:
		line = line.strip('\n')
		data = line.split('|')
		filtered_data = [float(data[i]) for i in selected_ind]
		list_filtered_data.append(filtered_data)


	#print(list_filtered_data)
	df = pd.DataFrame(list_filtered_data, columns=columns)
	#print(df)
	df = df.interpolate()
	df = df.bfill()
	df = df.fillna(0)

	X_train.append(df.iloc[-1][columns[:-1]])
	Y_train.append(df.iloc[-1][columns[-1]])
	#print(X_train)
	#print(Y_train)
	dfs_list.append(df)


print(len(X_train))
print(len(Y_train))
print(len(dfs_list))
filename = "training"
out_file = open(filename,'wb')
pickle.dump(X_train, out_file)
out_file.close()

filename = "labels"
out_file = open(filename,'wb')
pickle.dump(Y_train, out_file)
out_file.close()

filename = "Patients"
out_file = open(filename,'wb')
pickle.dump(dfs_list, out_file)
out_file.close()

filename = "Patients"
infile = open(filename, 'rb')
p_list = pickle.load(infile)
infile.close()


print(p_list[0])