from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.calibration import calibration_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import math
import collections
import random
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

filename = "Patients"
file = open(filename,'rb')
dfs = pickle.load(file)
file.close()

def createfeatures(k):
	headers = "HR|O2Sat|Temp|SBP|MAP|DBP|Resp|EtCO2|BaseExcess|HCO3|FiO2|pH|PaCO2|SaO2|AST|BUN|Alkalinephos|Calcium|Chloride|Creatinine|Bilirubin_direct|Glucose|Lactate|Magnesium|Phosphate|Potassium|Bilirubin_total|TroponinI|Hct|Hgb|PTT|WBC|Fibrinogen|Platelets|Age|Gender|Unit1|Unit2|HospAdmTime|ICULOS|SepsisLabel"
	columns = headers.split('|')
	#Taking only important features
	selected_ind = [0,1,2,3,4,5,6,7,10,19,22,26,31,33]
	columns = [columns[i] for i in selected_ind]

	age_subgroup = ['a','b','c']
	scaled_dfs = []
	sepsislabels = []
	age_list = []			
	scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
	for i in range(len(dfs)):
		sepsislabels.append(dfs[i].iloc[:,-1])
		age = dfs[i].iloc[0,-2]
		if age > 60:
			age_list.append(age_subgroup[2])
		elif age < 10:
			age_list.append(age_subgroup[0])
		else:
			age_list.append(age_subgroup[1])

		scaled_df = scaler.fit_transform(dfs[i].iloc[:,:-2])
		scaled_df = pd.DataFrame(scaled_df, columns=columns)

		scaled_dfs.append(scaled_df)

	labels = []
	index = 0
	total_features = []
	num = 0
	for df in scaled_dfs:
		if df.shape[0] >= 1:
			n_records = df.shape[0]
			filter_last_k = df.iloc[:-k]
			last_k_labels = sepsislabels[index].iloc[-k:]
			
			if (sum(sepsislabels[index])) >= 1.0:
				num += 1

			if sum(last_k_labels) >= 1:
				labels.append(1)
			else:
				labels.append(0)

			total_minus_k = filter_last_k
			mean=[]
			variance = []
			skew = []
			kurtosis = []
			total_minus_k_Features = []
			for col in list(total_minus_k.columns.values):
				total_minus_k_dif = total_minus_k[col].iloc[-1] - total_minus_k[col].iloc[0]
				total_minus_k_dit = total_minus_k.shape[0]
				total_minus_k_Features.append(total_minus_k_dif/total_minus_k_dit)


			l = total_minus_k_Features
			total_features.append(l)
			
			index += 1

	final_df = pd.DataFrame(total_features)

	print("Final DF shape :", final_df.shape)
	final_df["age"] = age_list

	#One Hot encoding the age feature
	OneHot = pd.get_dummies(final_df["age"])
	final_df = final_df.drop("age",axis = 1)
	final_df = final_df.join(OneHot)

	final_df["sepsislabel"] = labels

	file = open("NewFeatures",'wb')
	pickle.dump(final_df, file)
	file.close()

#Running the prediction before 4 and 6 hours 
for num_hrs in list((4, 6)):
	createfeatures(num_hrs)
	file = open("NewFeatures",'rb')
	data = pickle.load(file)
	file.close()

	#print(data.dtypes)
	true_data = data.loc[data["sepsislabel"] == 1]
	false_data = data.loc[data["sepsislabel"] == 0]

	true_data = true_data.reset_index(drop = True)
	false_data = false_data.reset_index(drop = True)
	print(true_data.shape)
	print(true_data.head())
	print(false_data.shape)
	print(false_data.head())

	#Randomly selecting the training the testing sets randomly
	ind = random.sample(range(true_data.shape[0]), 500)
	true_test_data = true_data.iloc[ind,:]
	true_data = true_data.drop(ind)

	ind = random.sample(range(false_data.shape[0]), 500)
	false_test_data = false_data.iloc[ind,:]
	false_data = false_data.drop(ind)

	test_data = pd.concat([true_test_data, false_test_data])
	X_test = test_data.values[:,:-1]
	Y_test = test_data.values[:,-1]

	#pca = PCA(n_components=100)
	#X_test = pca.fit_transform(X_test)

	predicted_labels_logit_list = []
	predicted_labels_xtrees_list = []
	predicted_labels_gboost_list = []
	predicted_labels_svm_list = []
	predicted_labels_knn_list = []

	for i in range(50):
		#Undersampling negative sepsis class
		ind = random.sample(range(false_data.shape[0]), 3500)
		new_false = false_data.iloc[ind,:]

		model_data = pd.concat([true_data,new_false])

		labels = model_data.values[:,-1]
		features = model_data.values[:,:-1]

		ros = SMOTE(random_state = 0)
		#Random Oversampling of postive sepsis class
		features, labels = ros.fit_sample(features, labels)


		features, labels = shuffle(features, labels, random_state = 0)

		X_train = features
		Y_train = labels

		clf_logi = LogisticRegression(random_state=0, solver='saga',multi_class='ovr', class_weight = 'balanced', C=0.5, max_iter = 500).fit(X_train, Y_train)
		predicted_labels_logit = clf_logi.predict(X_test)

		clf_xtra = ExtraTreesClassifier(n_estimators=100, criterion='entropy')
		clf_xtra= clf_xtra.fit(X_train, Y_train)
		predicted_labels_xtrees = clf_xtra.predict(X_test)

		clf_boost = GradientBoostingClassifier(n_estimators=100, learning_rate = 1.0, max_depth= 1, random_state=0)
		clf_boost= clf_boost.fit(X_train, Y_train)
		predicted_labels_gboost = clf_boost.predict(X_test)

		clf_svm = SVC(C=1.0, cache_size=200, class_weight='balanced', coef0=10.5,
	    decision_function_shape='ovr', degree=3, gamma='scale', kernel='poly',
	    probability=False, random_state=None, shrinking=True,
	    tol=0.001, verbose=False).fit(X_train, Y_train)
		predicted_labels_svm = clf_svm.predict(X_test)

		clf_knn = KNeighborsClassifier(n_neighbors=3, weights='distance', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None)
		clf_knn = clf_knn.fit(X_train, Y_train)
		predicted_labels_knn = clf_knn.predict(X_test)
		#w = accuracy_score(predicted_labels_knn, Y_test)

		predicted_labels_logit = predicted_labels_logit.tolist()
		predicted_labels_xtrees = predicted_labels_xtrees.tolist()
		predicted_labels_gboost = predicted_labels_gboost.tolist()
		predicted_predicabels_svm = predicted_labels_svm.tolist()
		predicted_labels_knn = predicted_labels_knn.tolist()

		predicted_labels_xtrees_list.append(predicted_labels_xtrees)
		predicted_labels_logit_list.append(predicted_labels_logit)
		predicted_labels_gboost_list.append(predicted_labels_gboost)
		predicted_labels_svm_list.append(predicted_labels_svm)
		predicted_labels_knn_list.append(predicted_labels_knn)


	predicted_logit_df = pd.DataFrame(predicted_labels_logit_list)
	predicted_xtrees_df = pd.DataFrame(predicted_labels_xtrees_list)
	predicted_labels_gboost_df = pd.DataFrame(predicted_labels_gboost_list)
	predicted_labels_svm_df = pd.DataFrame(predicted_labels_svm_list)
	predicted_labels_knn_df = pd.DataFrame(predicted_labels_knn_list)

	print(predicted_logit_df.shape)
	print(predicted_logit_df)
	final_predicted_logit = []
	final_predicted_xtrees = []
	final_predicted_gboost = []
	final_predicted_svm = []
	final_predicted_knn = []

	for col in predicted_logit_df.columns.values:
		final_predicted_logit.append(predicted_logit_df[col].mode().values[0])
		final_predicted_xtrees.append(predicted_xtrees_df[col].mode().values[0])
		final_predicted_gboost.append(predicted_labels_gboost_df[col].mode().values[0])
		final_predicted_svm.append(predicted_labels_svm_df[col].mode().values[0])
		final_predicted_knn.append(predicted_labels_knn_df[col].mode().values[0])

	print('\n\n')
	print('PREDICTION OF SEPSIS WITHIN THE NEXT:',num_hrs)
	fig, ax = plt.subplots()
	ax.set_title('calibration curve for '+str(num_hrs)+'hours prediction')
	print('\n')
	print("logistic regression:")
	confusion = confusion_matrix(Y_test, final_predicted_logit)
	TP = confusion[1, 1]
	TN = confusion[0, 0]
	FP = confusion[0, 1]
	FN = confusion[1, 0]
	print("accuracy of logistic regression ",(TP + TN) / float(TP + TN + FP + FN))
	print(confusion_matrix(Y_test, final_predicted_logit))
	print(classification_report(Y_test, final_predicted_logit))
	sensitivity = TP / float(FN + TP)
	print("sensitivity of logistic regression ", sensitivity)
	specificity = TN / float(TN + FP)
	print("specificity of logistic regression ",specificity)
	prob_pos = clf_logi.decision_function(X_test)
	prob_pos = (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
	fraction_of_positives, mean_predicted_value = calibration_curve(Y_test, prob_pos, n_bins=10, normalize=True)
	ax.plot([0, 1], [0, 1], linestyle='--', color='black', label='BASE')
	ax.plot(fraction_of_positives, mean_predicted_value, 'r--', label='LogisticRegression')
	print('\n\n')

	print("ExtraTreesClassifier :")
	confusion = confusion_matrix(Y_test, final_predicted_xtrees)
	TP = confusion[1, 1]
	TN = confusion[0, 0]
	FP = confusion[0, 1]
	FN = confusion[1, 0]
	print("accuracy of extratrees ",(TP + TN) / float(TP + TN + FP + FN))
	print(confusion_matrix(Y_test, final_predicted_xtrees))
	print(classification_report(Y_test, final_predicted_xtrees))
	sensitivity = TP / float(FN + TP)
	print("sensitivity of extratrees ", sensitivity)
	specificity = TN / float(TN + FP)
	print("specificity of extratrees ",specificity)
	print('\n\n')
	# prob_pos = clf_xtra.decision_function(X_test)
	# prob_pos = (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
	# fraction_of_positives, mean_predicted_value = calibration_curve(Y_test, prob_pos, n_bins=10, normalize=True)
	# plt.plot(fraction_of_positives, mean_predicted_value, 'bs')

	print("Support_vector_machine Classifier :")
	confusion = confusion_matrix(Y_test, final_predicted_svm)
	TP = confusion[1, 1]
	TN = confusion[0, 0]
	FP = confusion[0, 1]
	FN = confusion[1, 0]
	print("accuracy of SVM ",(TP + TN) / float(TP + TN + FP + FN))
	print(confusion_matrix(Y_test, final_predicted_svm))
	print(classification_report(Y_test, final_predicted_svm))
	sensitivity = TP / float(FN + TP)
	print("sensitivity of SVM ", sensitivity)
	specificity = TN / float(TN + FP)
	print("specificity of SVM ",specificity)
	prob_pos = clf_svm.decision_function(X_test)
	prob_pos = (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
	fraction_of_positives, mean_predicted_value = calibration_curve(Y_test, prob_pos, n_bins=10, normalize=True)
	ax.plot(fraction_of_positives, mean_predicted_value, 'g.-', label='SVM')
	print('\n\n')

	print("GradientBoostingClassifier :")
	confusion = confusion_matrix(Y_test, final_predicted_gboost)
	TP = confusion[1, 1]
	TN = confusion[0, 0]
	FP = confusion[0, 1]
	FN = confusion[1, 0]
	print("accuracy of gradientboost ",(TP + TN) / float(TP + TN + FP + FN))
	print(confusion_matrix(Y_test, final_predicted_gboost))
	print(classification_report(Y_test, final_predicted_gboost))
	sensitivity = TP / float(FN + TP)
	print("sensitivity of gradientboost ", sensitivity)
	specificity = TN / float(TN + FP)
	print("specificity of gradientboost ",specificity)
	prob_pos = clf_boost.decision_function(X_test)
	prob_pos = (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
	fraction_of_positives, mean_predicted_value = calibration_curve(Y_test, prob_pos, n_bins=10, normalize=True)
	ax.plot(fraction_of_positives, mean_predicted_value, 'c-', label='GradientBoosting')
	print('\n\n')


	print("KNeighborsClassifier :")
	confusion = confusion_matrix(Y_test, final_predicted_knn)
	TP = confusion[1, 1]
	TN = confusion[0, 0]
	FP = confusion[0, 1]
	FN = confusion[1, 0]
	print("accuracy of KNN ",(TP + TN) / float(TP + TN + FP + FN))
	print(confusion_matrix(Y_test, final_predicted_knn))
	print(classification_report(Y_test, final_predicted_knn))
	sensitivity = TP / float(FN + TP)
	print("sensitivity of KNN ", sensitivity)
	specificity = TN / float(TN + FP)
	print("specificity of KNN ",specificity)
	print('\n\n')
	# plot_lines = []
	# plot_lines.append([l2, l3, l4])
	# legend_names = plt.legend([plot_lines[i] for i in [0,1,2]], ["LogisticRegression", "SVM", "GradientBoosting"], loc=1)
	# plt.gca().add_artist(legend_names)
	leg = ax.legend();
	plt.show()
	# prob_pos = clf_knn.decision_function(X_test)
	# prob_pos = (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
	# fraction_of_positives, mean_predicted_value = calibration_curve(Y_test, prob_pos, n_bins=10, normalize=True)
	# plt.plot(fraction_of_positives, mean_predicted_value, 'm*')
	#plt.show()
