from stravalib.client import Client
import pickle
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# import seaborn as sns

# https://medium.com/analytics-vidhya/accessing-user-data-via-the-strava-api-using-stravalib-d5bee7fdde17


def main():

    client = Client()

    MY_STRAVA_CLIENT_ID, MY_STRAVA_CLIENT_SECRET = (
        open("client.secret").read().strip().split(",")
    )
    print(
        f"Client ID and secret read from file {MY_STRAVA_CLIENT_ID}, {MY_STRAVA_CLIENT_SECRET}"
    )

    with open("data/access_token.pickle", "rb") as f:
        STORED_ACCESS_TOKEN = pickle.load(f)

    print(f"Latest access token read from file: {STORED_ACCESS_TOKEN}")

    client = Client(access_token=STORED_ACCESS_TOKEN["access_token"])
    athlete = client.get_athlete()  # Get current athlete details
    print(
        "Athlete's name is {} {}, based in {}, {}".format(
            athlete.firstname, athlete.lastname, athlete.city, athlete.country
        )
    )

    activities = client.get_activities(limit=2000)
    # print(activities)
    activ_list = list(activities)
    types = [
        "time",
        "latlng",
        "altitude",
        "heartrate",
        "temp",
    ]

    run_data = []
    run_avg = []
    run_max = []
    for i, activity in enumerate(activ_list):
        activity_ = activity.to_dict()
        # print(activity_["type"])
        if activity_["type"] == "Run":
            # run_stream = client.get_activity_streams(
            #     activity_["id"], types=types, resolution="high"
            # )
            # run_data.append(np.array(run_stream["heartrate"].data))
            run_avg.append(activity_["average_heartrate"])
            run_max.append(activity_["max_heartrate"])
            run_data.append(activity_)

    with open("data/run_heartrates.pickle", "wb") as f:
        pickle.dump(np.array([run_avg, run_max]), f)
    with open("data/run_data.pickle", "wb") as f:
        pickle.dump(run_data, f)
    return


if __name__ == "__main__":
    main()
