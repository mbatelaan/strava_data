from stravalib.client import Client
import pickle
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# import seaborn as sns

# https://medium.com/analytics-vidhya/accessing-user-data-via-the-strava-api-using-stravalib-d5bee7fdde17


def main():
    # with open("data/run_heartrates.pickle", "rb") as f:
    with open("data/run_data.pickle", "rb") as f:
        run_data = pickle.load(f)

    # print(np.shape(run_data))
    print([key for key in run_data[0]])
    print("\n\n")
    print([key["external_id"] for key in run_data])
    print([key["upload_id"] for key in run_data])
    # print([key["trainer"] for key in run_data])
    # print(run_data[0]["device_name"])
    # for i in run_data:
    #     print(i["average_heartrate"])
    #     print(i["start_date"])

    dates = np.array([i["start_date"] for i in run_data])
    hr_avgs = np.array([i["average_heartrate"] for i in run_data])
    hr_max = np.array([i["max_heartrate"] for i in run_data])
    speed_avg = np.array([i["average_speed"] for i in run_data])
    pace_avg = 1 / speed_avg * 1000 / 60

    mpl_dates = mpl.dates.datestr2num(dates)

    print(len(pace_avg))
    print(len(hr_max))
    pace_avg = pace_avg[np.where(hr_max != None)]
    hr_avgs = hr_avgs[np.where(hr_max != None)]
    hr_max = hr_max[np.where(hr_max != None)]
    mpl_dates = mpl_dates[np.where(hr_max != None)]
    print(len(pace_avg))
    print(len(mpl_dates))
    print(len(hr_max))
    # print(hr_max)
    # print(pace_avg)
    print(np.shape(hr_max))
    print(np.shape(pace_avg))

    # marker_sizes = np.array([float(i) for i in hr_avgs]) - np.min(hr_avgs) + 1
    marker_sizes = np.floor(np.array([float(i) for i in hr_avgs]) / 10)
    marker_sizes = (marker_sizes - np.min(marker_sizes) + 1) * 10
    print(marker_sizes)

    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111)
    # ax.plot_date(
    #     mpl_dates, pace_avg, s=hr_max, marker="o", color="k", xdate=True, ydate=False
    # )
    # ax.scatter(mpl_dates, pace_avg, s=hr_avgs, marker="o", color="k")
    ax.scatter(mpl_dates, pace_avg, s=marker_sizes)
    # ax.plot_date(mpl_dates, hr_max, "k-,", xdate=True, ydate=False)
    # ax.plot_date(mpl_dates, hr_avgs, "b-,", xdate=True, ydate=False)
    # ax.xaxis.set_major_locator(mpl.dates.MonthLocator())
    ax.xaxis.set_major_formatter(mpl.dates.DateFormatter("%Y-%m-%d"))
    plt.show()

    # plt.figure()
    # plt.plot(dates, hr_max)
    # plt.show()

    return


if __name__ == "__main__":
    main()
