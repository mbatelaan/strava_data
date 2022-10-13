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


def plot_summary_stats(activities_data, nowstring):
    """Analyse all activities and plot the heart rate summary stats against time."""
    # Only keep activities with heartrate
    has_hr = np.where(np.array([i["has_heartrate"] for i in activities_data]))
    activities_data = np.array(activities_data)[has_hr]

    # Get a bunch of data from the activities
    dates = np.array([i["start_date"] for i in activities_data])
    hr_avgs = np.array([i["average_heartrate"] for i in activities_data])
    hr_max = np.array([i["max_heartrate"] for i in activities_data])
    speed_avg = np.array([i["average_speed"] for i in activities_data])
    moving_time = np.array([i["moving_time"] for i in activities_data])
    # pace_avg = 1 / speed_avg * 1000 / 60
    mpl_dates = mpl.dates.datestr2num(dates)

    # Split runs and rides
    activity_type = np.array([ac["type"] for ac in activities_data])
    print(activity_type)
    print(len(activity_type))
    runs = np.where(activity_type == "Run")
    rides = np.where(activity_type == "Ride")

    marker_sizes = np.array([float(i) for i in moving_time]) / np.max(moving_time) * 150

    # Plot the average heart rate against date for activities
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111)
    points1 = ax.scatter(
        mpl_dates[runs], hr_avgs[runs], c="k", label="runs", s=marker_sizes[runs]
    )
    points2 = ax.scatter(
        mpl_dates[rides], hr_avgs[rides], c="b", label="rides", s=marker_sizes[rides]
    )
    # ax.xaxis.set_major_formatter(mpl.dates.DateFormatter("%Y-%m-%d"))
    ax.xaxis.set_major_locator(mpl.dates.MonthLocator([1, 3, 5, 7, 9, 11]))
    ax.xaxis.set_major_formatter(mpl.dates.DateFormatter("%Y-%m"))
    ax.set_ylabel("average heart rate [bpm]")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.3)
    # Add a legend for the marker sizes
    sizes = np.arange(1, 9, 2) * (150 * 60 * 60) / np.max(moving_time)
    labels = [f"{s:.0f}h" for s in sizes * np.max(moving_time) / (150 * 60 * 60)]
    labels = labels + ["runs", "rides"]
    legend_points = [ax.scatter([], [], s=size_, c="gray") for size_ in sizes]
    legend_points = legend_points + [points1, points2]
    plt.legend(legend_points, labels, scatterpoints=1, loc=3)
    plt.savefig(f"plots/last_year_avg_hr_{nowstring}.pdf")
    plt.close()

    # Plot the average heart rate against date for activities
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111)
    points1 = ax.scatter(
        mpl_dates[runs], hr_max[runs], c="k", label="runs", s=marker_sizes[runs]
    )
    points2 = ax.scatter(
        mpl_dates[rides], hr_max[rides], c="b", label="rides", s=marker_sizes[rides]
    )
    # ax.xaxis.set_major_formatter(mpl.dates.DateFormatter("%Y-%m-%d"))
    ax.xaxis.set_major_locator(mpl.dates.MonthLocator([1, 3, 5, 7, 9, 11]))
    ax.xaxis.set_major_formatter(mpl.dates.DateFormatter("%Y-%m"))
    ax.set_ylabel("max heart rate [bpm]")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.3)
    # Add a legend for the marker sizes
    sizes = np.arange(1, 9, 2) * (150 * 60 * 60) / np.max(moving_time)
    labels = [f"{s:.0f}h" for s in sizes * np.max(moving_time) / (150 * 60 * 60)]
    labels = labels + ["runs", "rides"]
    legend_points = [ax.scatter([], [], s=size_, c="gray") for size_ in sizes]
    legend_points = legend_points + [points1, points2]
    plt.legend(legend_points, labels, scatterpoints=1, loc=3)
    plt.savefig(f"plots/last_year_max_hr_{nowstring}.pdf")
    plt.close()


def plot_hr_hist(heartrate_data, nowstring):
    """Plot a histogram of all of the heart rate data of the past year combined"""

    all_hr = np.array([item for hr in heartrate_data for item in hr])

    # bins = np.arange(80, 205, 5)
    bins = np.arange(75, 205, 2)
    # bins = np.arange(80, 200, 4)
    bins2 = np.arange(75, 205, 10)
    bins_zones = np.array([100, 120, 142, 160, 178, 196])

    fig = plt.figure(figsize=(7, 5))
    histogram = plt.hist(
        all_hr, bins=bins, density=False, alpha=0.5, label="all activities"
    )
    max_bin_value = np.max(histogram[0])
    ticks = np.arange(0, max_bin_value + 60 * 60, 60 * 60)
    plt.yticks(ticks, [f"{i/(60*60):.0f}" for i in ticks])
    plt.legend()
    plt.xlabel("heart rate [bpm]")
    plt.ylabel("hours")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"plots/last_year_hr_hist_{nowstring}.pdf")
    plt.close()
    # plt.show()

    fig = plt.figure(figsize=(7, 5))
    histogram = plt.hist(
        all_hr, bins=bins2, density=False, alpha=0.5, label="all activities"
    )
    max_bin_value = np.max(histogram[0])
    ticks = np.arange(0, max_bin_value + 60 * 60 * 5, 60 * 60 * 5)
    plt.xticks(histogram[1])
    plt.yticks(ticks, [f"{i/(60*60):.0f}" for i in ticks])
    plt.legend()
    plt.xlabel("heart rate [bpm]")
    plt.ylabel("hours")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"plots/last_year_hr_hist_wide_{nowstring}.pdf")
    plt.close()

    fig = plt.figure(figsize=(7, 5))
    histogram = plt.hist(
        all_hr, bins=bins_zones, density=False, alpha=0.5, label="all activities"
    )
    max_bin_value = np.max(histogram[0])
    ticks = np.arange(0, max_bin_value + 60 * 60 * 5, 60 * 60 * 5)
    plt.xticks(histogram[1])
    plt.yticks(ticks, [f"{i/(60*60):.0f}" for i in ticks])
    plt.legend()
    plt.xlabel("heart rate [bpm]")
    plt.ylabel("hours")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"plots/last_year_hr_hist_zones_{nowstring}.pdf")
    plt.close()

    return


def plot_hr_hist_split(heartrate_data, activities_data, nowstring):
    """Plot a histogram of all of the heart rate data of the past year split into Runs and Rides"""

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

    fig = plt.figure(figsize=(7, 5))
    histogram1 = plt.hist(
        all_hr_runs, bins=bins, density=False, alpha=0.5, label="runs"
    )
    histogram2 = plt.hist(
        all_hr_rides, bins=bins, density=False, alpha=0.5, label="rides"
    )
    max_bin_value = np.max(np.append(histogram1[0], histogram2[0]))
    ticks = np.arange(0, max_bin_value + 60 * 60, 60 * 60)
    # plt.xticks(histogram1[1])
    plt.yticks(ticks, [f"{i/(60*60):.1f}" for i in ticks])
    plt.legend()
    plt.xlabel("heart rate [bpm]")
    plt.ylabel("hours")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"plots/last_year_hr_hist_split_{nowstring}.pdf")
    plt.close()

    fig = plt.figure(figsize=(7, 5))
    histogram1 = plt.hist(
        all_hr_runs, bins=bins2, density=False, alpha=0.5, label="runs"
    )
    histogram2 = plt.hist(
        all_hr_rides, bins=bins2, density=False, alpha=0.5, label="rides"
    )
    max_bin_value = np.max(np.append(histogram1[0], histogram2[0]))
    ticks = np.arange(0, max_bin_value + 60 * 60 * 5, 60 * 60 * 5)
    plt.xticks(histogram1[1])
    plt.yticks(ticks, [f"{i/(60*60):.0f}" for i in ticks])
    plt.legend()
    plt.xlabel("heart rate [bpm]")
    plt.ylabel("hours")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"plots/last_year_hr_hist_split_wide_{nowstring}.pdf")
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
    plt.savefig(f"plots/last_year_hr_hist_split_zones_{nowstring}.pdf")
    plt.close()

    return


def main():
    """Read the last year's activities from a pickle file and make plots of the data"""

    plt.rc("text", usetex=True)
    rcParams.update({"figure.autolayout": True})

    nowstring = datetime.datetime(2022, 10, 13).strftime("%Y_%m_%d")
    with open(f"data/activities_data_last_year_{nowstring}.pickle", "rb") as f:
        activities_data = pickle.load(f)
    # Plot the average and max heartrate against dates with activity length as marker size.
    plot_summary_stats(activities_data, nowstring)

    # Older data file for these plots
    nowstring = datetime.datetime(2022, 10, 12).strftime("%Y_%m_%d")
    with open(f"data/heartrate_data_last_year_{nowstring}.pickle", "rb") as f:
        heartrate_data = pickle.load(f)
    with open(f"data/activities_data_last_year_{nowstring}.pickle", "rb") as f:
        activities_data = pickle.load(f)

    # Plot a histogram of all the heartrate data
    plot_hr_hist(heartrate_data, nowstring)
    # Plot a histogram of all the heartrate data split into runs and rides
    plot_hr_hist_split(heartrate_data, activities_data, nowstring)

    return


if __name__ == "__main__":
    main()
