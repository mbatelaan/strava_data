from stravalib.client import Client
import pickle
import numpy as np
import pandas as pd
import time, datetime
from matplotlib import rcParams
import matplotlib as mpl
import matplotlib.pyplot as plt

# import seaborn as sns

# https://medium.com/analytics-vidhya/accessing-user-data-via-the-strava-api-using-stravalib-d5bee7fdde17


def plot_summary_stats(activities_data):
    """Analyse all activities and plot the heart rate summary stats against time."""
    # Only keep activities with heartrate
    has_hr = np.where(np.array([i["has_heartrate"] for i in activities_data]))
    activities_data = np.array(activities_data)[has_hr]

    # Get a bunch of data from the activities
    dates = np.array([i["start_date"] for i in activities_data])
    hr_avgs = np.array([i["average_heartrate"] for i in activities_data])
    hr_max = np.array([i["max_heartrate"] for i in activities_data])
    speed_avg = np.array([i["average_speed"] for i in activities_data])
    # pace_avg = 1 / speed_avg * 1000 / 60
    mpl_dates = mpl.dates.datestr2num(dates)

    # Split runs and rides
    activity_type = np.array([ac["type"] for ac in activities_data])
    runs = np.where(activity_type == "Run")
    rides = np.where(activity_type == "Ride")

    nowstring = datetime.datetime.now().strftime("%Y_%m_%d")

    # Plot the average heart rate against date for activities
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111)
    points = ax.scatter(mpl_dates[runs], hr_avgs[runs], c="k", label="runs")
    points = ax.scatter(mpl_dates[rides], hr_avgs[rides], c="b", label="rides")
    ax.xaxis.set_major_formatter(mpl.dates.DateFormatter("%Y-%m-%d"))
    ax.set_ylabel("average heart rate [bpm]")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.savefig(f"plots/last_month_avg_hr_{nowstring}.pdf")
    plt.close()

    # Plot the average heart rate against date for activities
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111)
    points = ax.scatter(mpl_dates[runs], hr_max[runs], c="k", label="runs")
    points = ax.scatter(mpl_dates[rides], hr_max[rides], c="b", label="rides")
    ax.xaxis.set_major_formatter(mpl.dates.DateFormatter("%Y-%m-%d"))
    ax.set_ylabel("max heart rate [bpm]")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.savefig(f"plots/last_month_max_hr_{nowstring}.pdf")
    plt.close()


def plot_hr_hist(heartrate_data):
    """Plot a histogram of all of the heart rate data of the past month combined"""

    nowstring = datetime.datetime.now().strftime("%Y_%m_%d")

    all_hr = np.array([item for hr in heartrate_data for item in hr])

    # bins = np.arange(80, 205, 5)
    bins = np.arange(75, 205, 2)
    # bins = np.arange(80, 200, 4)
    bins2 = np.arange(75, 205, 10)
    bins_zones = np.array([100, 120, 142, 160, 178, 196])

    fig = plt.figure(figsize=(7, 5))
    plt.hist(all_hr, bins=bins, density=True, alpha=0.5, label="all activities")
    plt.legend()
    plt.xlabel("heart rate [bpm]")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"plots/last_month_hr_hist_{nowstring}.pdf")
    plt.close()
    # plt.show()

    fig = plt.figure(figsize=(7, 5))
    histogram = plt.hist(all_hr, bins=bins2, density=False, alpha=0.5, label="runs")
    max_bin_value = np.max(histogram[0])
    ticks = np.arange(0, max_bin_value + 60 * 60, 60 * 60)
    plt.xticks(histogram[1])
    plt.yticks(ticks, [f"{i/(60*60):.0f}" for i in ticks])
    plt.legend()
    plt.xlabel("heart rate [bpm]")
    plt.ylabel("hours")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"plots/last_month_hr_hist_wide_{nowstring}.pdf")
    plt.close()

    fig = plt.figure(figsize=(7, 5))
    histogram = plt.hist(
        all_hr, bins=bins_zones, density=False, alpha=0.5, label="runs"
    )
    max_bin_value = np.max(histogram[0])
    ticks = np.arange(0, max_bin_value + 60 * 60, 60 * 60)
    plt.xticks(histogram[1])
    plt.yticks(ticks, [f"{i/(60*60):.0f}" for i in ticks])
    plt.legend()
    plt.xlabel("heart rate [bpm]")
    plt.ylabel("hours")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"plots/last_month_hr_hist_zones_{nowstring}.pdf")
    plt.close()

    return


def plot_hr_hist_split(heartrate_data, activities_data):
    """Plot a histogram of all of the heart rate data of the past month split into Runs and Rides"""

    has_hr = np.where(np.array([i["has_heartrate"] for i in activities_data]))
    activities_data = np.array(activities_data)[has_hr]

    # Split runs and rides
    activity_type = np.array([ac["type"] for ac in activities_data])
    runs = np.where(activity_type == "Run")[0]
    rides = np.where(activity_type == "Ride")[0]
    hr_runs = [heartrate_data[r] for r in runs]
    hr_rides = [heartrate_data[r] for r in rides]

    all_hr_runs = np.array([item for hr in hr_runs for item in hr])
    all_hr_rides = np.array([item for hr in hr_rides for item in hr])

    bins = np.arange(75, 205, 2)
    bins2 = np.arange(75, 205, 10)
    bins_zones = np.array([100, 120, 142, 160, 178, 196])

    nowstring = datetime.datetime.now().strftime("%Y_%m_%d")

    fig = plt.figure(figsize=(7, 5))
    plt.hist(all_hr_runs, bins=bins, density=False, alpha=0.5, label="runs")
    plt.hist(all_hr_rides, bins=bins, density=False, alpha=0.5, label="rides")
    plt.legend()
    plt.xlabel("heart rate [bpm]")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"plots/last_month_hr_hist_split_{nowstring}.pdf")
    plt.close()

    fig = plt.figure(figsize=(7, 5))
    histogram1 = plt.hist(
        all_hr_runs, bins=bins2, density=False, alpha=0.5, label="runs"
    )
    histogram2 = plt.hist(
        all_hr_rides, bins=bins2, density=False, alpha=0.5, label="rides"
    )
    max_bin_value = np.max(np.append(histogram1[0], histogram2[0]))
    ticks = np.arange(0, max_bin_value + 60 * 60, 60 * 60)
    plt.xticks(histogram1[1])
    plt.yticks(ticks, [f"{i/(60*60):.0f}" for i in ticks])
    plt.legend()
    plt.xlabel("heart rate [bpm]")
    plt.ylabel("hours")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"plots/last_month_hr_hist_split_wide_{nowstring}.pdf")
    plt.close()

    fig = plt.figure(figsize=(7, 5))
    histogram1 = plt.hist(
        all_hr_runs, bins=bins_zones, density=False, alpha=0.5, label="runs"
    )
    histogram2 = plt.hist(
        all_hr_rides, bins=bins_zones, density=False, alpha=0.5, label="rides"
    )
    max_bin_value = np.max(np.append(histogram1[0], histogram2[0]))
    ticks = np.arange(0, max_bin_value + 60 * 60 * 5, 60 * 60 * 5)
    plt.xticks(histogram1[1])
    plt.yticks(ticks, [f"{i/(60*60):.0f}" for i in ticks])
    plt.legend()
    plt.xlabel("heart rate [bpm]")
    plt.ylabel("hours")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"plots/last_month_hr_hist_split_zones_{nowstring}.pdf")
    plt.close()

    return


def main():
    """Get the activities of the last month from strava and save the heart rate data to a file"""

    plt.rc("text", usetex=True)
    rcParams.update({"figure.autolayout": True})

    nowstring = datetime.datetime(2022, 10, 12).strftime("%Y_%m_%d")
    print(f"{nowstring=}")

    with open(f"data/heartrate_data_last_month_{nowstring}.pickle", "rb") as f:
        heartrate_data = pickle.load(f)
    with open(f"data/activities_data_last_month_{nowstring}.pickle", "rb") as f:
        activities_data = pickle.load(f)

    plot_summary_stats(activities_data)

    plot_hr_hist(heartrate_data)

    plot_hr_hist_split(heartrate_data, activities_data)

    return


if __name__ == "__main__":
    main()
