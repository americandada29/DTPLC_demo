import pickle
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import numpy as np


#### Retrieve the data for training and testing ####
with open("data.pkl","rb") as f:
    data = pickle.load(f)



##### Load the energy and position data ######
energies = data['energies']
x = data['x']


##### Reshape the data so that sklearn is happy (needs to be x = [[1],[2],[3]...] not x=[1, 2, 3 ...]) #####
all_x = np.reshape(x, (-1,1))
all_energies = np.reshape(energies,(-1,1))


##### Split the data into training and testing sets with the test set recieving 20% of the data #########
x_train, x_test, energy_train, energy_test = train_test_split(all_x, all_energies, test_size=0.2, random_state=42)


##### Create a fitter ########
# clf is the name assigned to the fitter, can be anything
# tol is tolerance, basically how little error you can have before stopping
# epsilon is the distance your point can be from the actual value to be considered "ok"
# kernel is just the function that is being fit. We use the default which is rbf
##############################
clf = SVR(kernel="rbf", tol=1e-8, epsilon=1e-4)


##### Now fit the position vs energy values #####
clf.fit(x_train, energy_train)


##### Use the fitted function to predicted the energies at the test set x positions #####
predicted_energies = clf.predict(x_test)



##### Plot the true function vs the predicted energies #####
plt.plot(x, energies, label="Actuals")
plt.plot(x_test, predicted_energies, "o", label="Predictions")
plt.xlabel("Position (Angstroms)")
plt.ylabel("Energies (eV)")
plt.legend()
plt.show()
##### We can see the fit is pretty good! ######
