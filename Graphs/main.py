import matplotlib.pyplot as plt

epochs3 = [1, 2, 3]
epochs8 = [1, 2, 3, 4, 5, 6, 7, 8]
################################## MNIST DATASET ####################################################################
#Laptop EfficientNetB0
accuracy_MI_EF_LA = [92.8, 96.74, 97.38]
plt.plot(epochs3, accuracy_MI_EF_LA, color='blue', label="EfficientNetB0 - Laptop")
#Rapsberry Pi 4 EfficientNetB0
accuracy_MI_EF_RP = [92.1, 96.43, 96.945]
plt.plot(epochs3, accuracy_MI_EF_RP, color='orange', label="EfficientNetB0 - Raspberry PI 4")

#Laptop Sequential
accuracy_MI_SE_LA = [96.08, 98.66, 99.31]
plt.plot(epochs3, accuracy_MI_SE_LA, color='green', label="Sequential - Laptop")
#Rapsberry Pi 4 Sequential
accuracy_MI_SE_RP = [96.06, 98.61, 99.25]
plt.plot(epochs3, accuracy_MI_SE_RP, color='red', label="Sequential - Raspberry PI 4")

plt.xlabel('Epoch Nr')
plt.ylabel('Accuracy %')
plt.title('Mean value of the predicted accuracy for each epoch using Mnist dataset (LOW ML)')
plt.legend()
#plt.show()

################################## FASHION MNIST DATASET #############################################################
#Laptop EfficientNetB0
accuracy_FMI_EF_LA = [87.24, 90.35, 91.14]
plt.plot(epochs3, accuracy_FMI_EF_LA, color='blue', label="EfficientNetB0 - Laptop")
#Rapsberry Pi 4 EfficientNetB0
accuracy_FMI_EF_RP = [86.94, 89.86, 90.9]
plt.plot(epochs3, accuracy_FMI_EF_RP, color='orange', label="EfficientNetB0 - Raspberry PI 4")

#Laptop Sequential
accuracy_FMI_SE_LA = [87.13, 91.51, 93.71]
plt.plot(epochs3, accuracy_FMI_SE_LA, color='green', label="Sequential - Laptop")
#Rapsberry Pi 4 Sequential
accuracy_FMI_SE_RP = [86.5, 90.92, 93.1]
plt.plot(epochs3, accuracy_FMI_SE_RP, color='red', label="Sequential - Raspberry PI 4")

plt.xlabel('Epoch Nr')
plt.ylabel('Accuracy %')
plt.title('Mean value of the predicted accuracy for each epoch using Fashion Mnist dataset (MID ML)')
plt.legend()
#plt.show()

################################## FOOD101 DATASET ####################################################################
#Laptop EfficientNetB0
accuracy_FO_EF_LA = [55.79, 66.57, 70.35, 72.51, 74.45, 75.86, 77.08, 78.18]
plt.plot(epochs8, accuracy_FO_EF_LA, color='blue', label="EfficientNetB0 - Laptop")
#Rapsberry Pi 4 EfficientNetB0
accuracy_FO_EF_RP = [55.55, 66.32, 70.1, 71.87, 74.04, 75.6, 76.834, 77.97]
plt.plot(epochs8, accuracy_FO_EF_RP, color='orange', label="EfficientNetB0 - Raspberry PI 4")

#Laptop Sequential
accuracy_FO_SE_LA = [5.37, 13.63, 22.05, 32.8, 46.32, 61.97, 74.11, 81.96]
plt.plot(epochs8, accuracy_FO_SE_LA, color='green', label="Sequential - Laptop")
#Rapsberry Pi 4 Sequential
accuracy_FO_SE_RP = [4.96, 13.28, 22.62, 33.1, 46.76, 61.41, 73.71, 81.49]
plt.plot(epochs8, accuracy_FO_SE_RP, color='red', label="Sequential - Raspberry PI 4")

plt.xlabel('Epoch Nr')
plt.ylabel('Accuracy %')
plt.title('Mean value of the predicted accuracy for each epoch using Food101 dataset (HEAVY ML)')
plt.legend()
#plt.show()

################################## MNIST DATASET TIME ####################################################################
#Laptop EfficientNetB0
time_MI_EF_LA = [21, 43, 66]
plt.plot(time_MI_EF_LA, accuracy_MI_EF_LA, color='blue', label="EfficientNetB0 - Laptop")
#Rapsberry Pi 4 EfficientNetB0
time_MI_EF_RP = [278, 556, 836]
plt.plot(time_MI_EF_RP, accuracy_MI_EF_RP, color='orange', label="EfficientNetB0 - Raspberry PI 4")

#Laptop Sequential
time_MI_SE_LA = [11, 21.5, 33]
plt.plot(time_MI_SE_LA, accuracy_MI_SE_LA, color='green', label="Sequential - Laptop")
#Rapsberry Pi 4 Sequential
time_MI_SE_RP = [141, 274, 475]
plt.plot(time_MI_SE_RP, accuracy_MI_SE_RP, color='red', label="Sequential - Raspberry PI 4")

plt.xlabel('Elapsed time, min')
plt.ylabel('Accuracy, %')
plt.title('Mean value of the predicted accuracy relative to elapsed time using Mnist dataset (LOW ML)')
plt.legend()
#plt.show()

################################## FASHION MNIST DATASET TIME ############################################################
#Laptop EfficientNetB0
time_FMI_EF_LA = [22.8, 43.2, 67]
plt.plot(time_FMI_EF_LA, accuracy_FMI_EF_LA, color='blue', label="EfficientNetB0 - Laptop")
#Rapsberry Pi 4 EfficientNetB0
time_FMI_EF_RP = [278, 556, 836]
plt.plot(time_FMI_EF_RP, accuracy_FMI_EF_RP, color='orange', label="EfficientNetB0 - Raspberry PI 4")

#Laptop Sequential
time_FMI_SE_LA = [10.9, 21.8, 33]
plt.plot(time_FMI_SE_LA, accuracy_FMI_SE_LA, color='green', label="Sequential - Laptop")
#Rapsberry Pi 4 Sequential
time_FMI_SE_RP = [141, 274.5, 475]
plt.plot(time_FMI_SE_RP, accuracy_FMI_SE_RP, color='red', label="Sequential - Raspberry PI 4")

plt.xlabel('Elapsed time, min')
plt.ylabel('Accuracy, %')
plt.title('Mean value of the predicted accuracy relative to elapsed time using Fashion Mnist dataset (MID ML)')
plt.legend()
#plt.show()

################################## FOOD101 DATASET TIME ##################################################################
#Laptop EfficientNetB0
time_FO_EF_LA = [28, 55.5, 83, 110.5, 138, 165.5, 193, 221]
plt.plot(time_FO_EF_LA, accuracy_FO_EF_LA, color='blue', label="EfficientNetB0 - Laptop")
#Rapsberry Pi 4 EfficientNetB0
time_FO_EF_RP = [285, 556, 847, 1134, 1420, 1701, 1989.4, 2274]
plt.plot(time_FO_EF_RP, accuracy_FO_EF_RP, color='orange', label="EfficientNetB0 - Raspberry PI 4")

#Laptop Sequential
time_FO_SE_LA = [14.8, 29.5, 44.4, 59.4, 74.9, 90.5, 105.8, 131]
plt.plot(time_FO_SE_LA, accuracy_FO_SE_LA, color='green', label="Sequential - Laptop")
#Rapsberry Pi 4 Sequential
time_FO_SE_RP = [164, 313.8, 466, 616, 769, 921.7, 1075, 1228.8]
plt.plot(time_FO_SE_RP, accuracy_FO_SE_RP, color='red', label="Sequential - Raspberry PI 4")

plt.xlabel('Time, min')
plt.ylabel('Accuracy %')
plt.title('Mean value of the predicted accuracy relative to elapsed time using Food101 dataset (HEAVY ML)')
plt.legend()
#plt.show()

################################## CALCULATE STANDARD DEVIATION ##########################################################

def printStdev(list_):
    list = []
    for element in list_:
        list.append(element)
        if (len(list) > 1):
            print(st.stdev(list))

import statistics as st
time_MI_EF_LA = [21, 43, 66]
time_MI_EF_RP = [278, 556, 836]
time_MI_SE_LA = [11, 21.5, 33]
time_MI_SE_RP = [141, 274, 475]

time_FMI_EF_LA = [22.8, 43.2, 67]
time_FMI_EF_RP = [278, 556, 836]
time_FMI_SE_LA = [10.9, 21.8, 33]
time_FMI_SE_RP = [141, 274.5, 475]

time_FO_EF_LA = [28, 55.5, 83, 110.5, 138, 165.5, 193, 221]
time_FO_EF_RP = [285, 556, 847, 1134, 1420, 1701, 1989.4, 2274]
time_FO_SE_LA = [14.8, 29.5, 44.4, 59.4, 74.9, 90.5, 105.8, 131]
time_FO_SE_RP = [164, 313.8, 466, 616, 769, 921.7, 1075, 1228.8]

printStdev(time_MI_EF_LA)