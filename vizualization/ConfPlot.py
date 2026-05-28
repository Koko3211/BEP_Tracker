import matplotlib.pyplot as plt

"""
Vizualization script min conf thresh graphs for the report
"""

x = [0.05,0.1,0.15,0.2,0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
bdihota = [58.393,59.569,59.569,60.155,58.445,58.188,56.775,56.09,56.298,54.12]
bdiidf1 = [67.868,69.327,69.327,71.529,68.537,69.269,67.317,65.735,64.499,61.229]
bdiidsw = [236,202,202,171,185,150,101,73,49,46]

sorthota = [40.96,44.817,46.423,49.673,49.673,53.192,52.311,52.172,52.172,52.172]
sortidf1 = [42.024,47.993,48.94,54.085,54.085,61.476,59.576,59.168,59.168,59.671]
sortidsw = [941,725,586,466,466,290,202,136,136,80]
fig, ax = plt.subplots(1,3, figsize=(12, 5))

ax[0].plot(x, bdihota, marker = "o", label = "ours")
ax[0].plot(x, sorthota, marker = "o", label =  "SORT")
ax[0].set_xlabel("Confidence threshold")
ax[0].set_ylabel("HOTA")
ax[0].set_title("HOTA vs Confidence Threshold")
ax[0].grid(True)
ax[0].legend()

ax[1].plot(x, bdiidf1, marker = "o", label = "ours")
ax[1].plot(x, sortidf1, marker = "o", label = "SORT")
ax[1].set_xlabel("Confidence threshold")
ax[1].set_ylabel("IDF1")
ax[1].set_title("IDF1 vs Confidence Threshold")
ax[1].grid(True)
ax[1].legend()

ax[2].plot(x, bdiidsw, marker = "o", label = "ours")
ax[2].plot(x, sortidsw, marker = "o", label = "SORT")
ax[2].set_xlabel("Confidence threshold")
ax[2].set_ylabel("IDSW")
ax[2].set_title("IDSW vs Confidence Threshold")
ax[2].grid(True)
ax[2].legend()

plt.show()