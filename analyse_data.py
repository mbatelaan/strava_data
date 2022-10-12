from stravalib.client import Client
import pickle
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# import seaborn as sns

# https://medium.com/analytics-vidhya/accessing-user-data-via-the-strava-api-using-stravalib-d5bee7fdde17

with open("data/wallaroo_hr.pickle", "rb") as f:
    wallaroo_hr = pickle.load(f)
with open("data/moonta_hr.pickle", "rb") as f:
    moonta_hr = pickle.load(f)
with open("data/wallaroo_activity.pickle", "rb") as f:
    most_recent = pickle.load(f)

# # print(most_recent)
# print([key for key in most_recent])
# print(most_recent["name"])
# print(most_recent["id"])
# print(most_recent["average_heartrate"])
# print(most_recent["max_heartrate"])
# print(most_recent["has_heartrate"])
# print(most_recent["distance"])
# print(most_recent["laps"])
# print(most_recent["average_speed"])
# print(most_recent["max_speed"])
# # print([act for act in activities])

# types = [
#     "time",
#     "latlng",
#     "altitude",
#     "heartrate",
#     "temp",
# ]


# print(wallaroo_hr)
print(moonta_hr)
wallaroo_hr_hist = np.histogram(wallaroo_hr, bins=20)
moonta_hr_hist = np.histogram(moonta_hr, bins=wallaroo_hr_hist[1])
print(wallaroo_hr_hist)
print(moonta_hr_hist)

plt.figure()
plt.hist(
    wallaroo_hr, bins=wallaroo_hr_hist[1], density=True, alpha=0.5, label="Wallaroo RR"
)
plt.hist(
    moonta_hr, bins=wallaroo_hr_hist[1], density=True, alpha=0.5, label="Moonta crit"
)
plt.legend()
plt.show()
