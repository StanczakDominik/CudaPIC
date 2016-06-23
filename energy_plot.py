import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
data = pd.read_csv("energies.dat", delimiter=" ", index_col=0,
    names = ["rho", "Field energy", "ElectronKE", "IonKE", "ElectronPX", "IonPX", "ElectronPY", "IonPY","ElectronPZ","IonPZ","Total energy"])

index = data.index.values
plt.plot(index, data['Field energy'], label="Field energy")
plt.plot(index, data['ElectronKE'], label="Electron energy")
plt.plot(index, data['IonKE'], label="Ion energy")
plt.plot(index, data['Total energy'], label="Total energy")
plt.legend(loc='best')
plt.grid()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(data['ElectronPX'], data['ElectronPY'], data['ElectronPZ'], "bo-")
ax.plot(data['IonPX'], data['IonPY'], data['IonPZ'], "go-")
ax.set_xlabel("Px")
ax.set_ylabel("Py")
ax.set_zlabel("Pz")
plt.show()
