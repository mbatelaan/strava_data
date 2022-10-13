from stravalib.client import Client
import pickle
import numpy as np
import pandas as pd
import time, datetime
import matplotlib as mpl
import matplotlib.pyplot as plt

# import seaborn as sns

# https://medium.com/analytics-vidhya/accessing-user-data-via-the-strava-api-using-stravalib-d5bee7fdde17


def get_access_token():
    client = Client()

    MY_STRAVA_CLIENT_ID, MY_STRAVA_CLIENT_SECRET = (
        open("client.secret").read().strip().split(",")
    )
    print(f"Client ID and secret read from file {MY_STRAVA_CLIENT_ID}")

    with open("data/access_token.pickle", "rb") as f:
        STORED_ACCESS_TOKEN = pickle.load(f)

    print(f"Latest access token read from file: {STORED_ACCESS_TOKEN}")

    if time.time() > STORED_ACCESS_TOKEN["expires_at"]:
        print("Token has expired, will refresh")

        refresh_response = client.refresh_access_token(
            client_id=MY_STRAVA_CLIENT_ID,
            client_secret=MY_STRAVA_CLIENT_SECRET,
            refresh_token=STORED_ACCESS_TOKEN["refresh_token"],
        )

        STORED_ACCESS_TOKEN = refresh_response
        with open("data/access_token.pickle", "wb") as f:
            pickle.dump(refresh_response, f)

        print("Refreshed token saved to file")
        client.access_token = refresh_response["access_token"]
        client.refresh_token = refresh_response["refresh_token"]
        client.token_expires_at = refresh_response["expires_at"]

    else:
        print(
            "Token still valid, expires at {}".format(
                time.strftime(
                    "%a, %d %b %Y %H:%M:%S %Z",
                    time.localtime(STORED_ACCESS_TOKEN["expires_at"]),
                )
            )
        )
        client.access_token = STORED_ACCESS_TOKEN["access_token"]
        client.refresh_token = STORED_ACCESS_TOKEN["refresh_token"]
        client.token_expires_at = STORED_ACCESS_TOKEN["expires_at"]
    return STORED_ACCESS_TOKEN


def save_last_month_data(activ_list, client):
    types = [
        "time",
        "latlng",
        "altitude",
        "heartrate",
        "temp",
    ]

    activities_data = []
    heartrate_data = []

    nowtime = mpl.dates.date2num(datetime.datetime(*time.localtime()[:6]))
    lastmonth = nowtime - 30
    lastmonth_ = mpl.dates.num2date(lastmonth)
    print(nowtime)
    print(lastmonth_)

    for i, activity in enumerate(activ_list):
        activity_ = activity.to_dict()
        print(activity_["start_date"])
        # Get the date of the activity as a float, then check if it is within the last month:
        activity_date = mpl.dates.datestr2num(activity_["start_date"])
        print(f"{activity_date=}")
        print(f"{lastmonth=}")
        if activity_date >= lastmonth:
            activities_data.append(activity_)
            if activity_["has_heartrate"]:
                activity_stream = client.get_activity_streams(
                    activity_["id"], types=types, resolution="high"
                )
                heartrate_data.append(np.array(activity_stream["heartrate"].data))
        else:
            break
    nowstring = datetime.datetime.now().strftime("%Y_%m_%d")

    with open(f"data/heartrate_data_last_month_{nowstring}.pickle", "wb") as f:
        pickle.dump(heartrate_data, f)
    with open(f"data/activities_data_last_month_{nowstring}.pickle", "wb") as f:
        pickle.dump(activities_data, f)
    return


def save_last_year_data(activ_list, client):
    """Give a list of activities, save all the ones from the last year to a pickle file"""

    activities_data = []

    nowtime = mpl.dates.date2num(datetime.datetime(*time.localtime()[:6]))
    lastyear = nowtime - 365

    for i, activity in enumerate(activ_list[:98]):
        activity_ = activity.to_dict()
        print(activity_["start_date"])
        # Get the date of the activity as a float, then check if it is within the last month:
        activity_date = mpl.dates.datestr2num(activity_["start_date"])
        print(f"{activity_date=}")
        print(f"{lastyear=}")
        if activity_date >= lastyear:
            activities_data.append(activity_)
        else:
            break
    nowstring = datetime.datetime.now().strftime("%Y_%m_%d")

    with open(f"data/activities_data_last_year_{nowstring}.pickle", "wb") as f:
        pickle.dump(activities_data, f)
    return


def extend_save_last_year_data(activ_list, client):
    """Give a list of activities, save all the ones from the last year and request their heart rate data stream to save that as well
    Taking into account the rate-limiting by strava, this function will do 79 activities at a time, then wait 15 minutes and then continue again.
    """
    types = [
        "time",
        "latlng",
        "altitude",
        "heartrate",
        "temp",
    ]

    activities_data = []
    heartrate_data = []

    nowtime = mpl.dates.date2num(datetime.datetime(*time.localtime()[:6]))
    lastyear = nowtime - 365

    count = 0
    for i, activity in enumerate(activ_list):
        if count == 97:
            print("sleeping")
            time.sleep(60 * 16)
            count = 0
        activity_ = activity.to_dict()
        print(activity_["start_date"])
        # Get the date of the activity as a float, then check if it is within the last month:
        activity_date = mpl.dates.datestr2num(activity_["start_date"])
        print(f"{activity_date=}")
        print(f"{lastyear=}")
        if activity_date >= lastyear:
            activities_data.append(activity_)
            if activity_["has_heartrate"]:
                activity_stream = client.get_activity_streams(
                    activity_["id"], types=types, resolution="high"
                )
                heartrate_data.append(np.array(activity_stream["heartrate"].data))
        else:
            break
        count = count + 1

    nowstring = datetime.datetime.now().strftime("%Y_%m_%d")

    with open(f"data/heartrate_data_last_year_{nowstring}.pickle", "wb") as f:
        pickle.dump(heartrate_data, f)
    with open(f"data/activities_data_last_year_{nowstring}.pickle", "wb") as f:
        pickle.dump(activities_data, f)
    return


def main():
    """Get the activities of the last month from strava and save the heart rate data to a file"""

    STORED_ACCESS_TOKEN = get_access_token()

    # Use the access token to get data from strava
    client = Client(access_token=STORED_ACCESS_TOKEN["access_token"])
    athlete = client.get_athlete()  # Get current athlete details
    print(
        "Athlete's name is {} {}, based in {}, {}".format(
            athlete.firstname, athlete.lastname, athlete.city, athlete.country
        )
    )

    activities = client.get_activities(limit=400)
    activ_list = list(activities)

    # save_last_month_data(activ_list, client)

    save_last_year_data(activ_list, client)

    # extend_save_last_year_data(activ_list, client)
    return


if __name__ == "__main__":
    main()
