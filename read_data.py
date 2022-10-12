from stravalib.client import Client
import pickle
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# import seaborn as sns

# https://medium.com/analytics-vidhya/accessing-user-data-via-the-strava-api-using-stravalib-d5bee7fdde17

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

activities = client.get_activities(limit=100)
print(activities)
activ_list = list(activities)
print("\n\n\n")
most_recent = activ_list[4].to_dict()
# print(most_recent)
print([key for key in most_recent])
print(most_recent["name"])
print(most_recent["workout_type"])
print(most_recent["id"])
print(most_recent["guid"])
print(most_recent["upload_id"])
print(most_recent["external_id"])
print(most_recent["type"])
print(most_recent["average_heartrate"])
print(most_recent["max_heartrate"])
print(most_recent["has_heartrate"])
print(most_recent["distance"])
print(most_recent["laps"])
print(most_recent["average_speed"])
print(most_recent["max_speed"])
# print([act for act in activities])

types = [
    "time",
    "latlng",
    "altitude",
    "heartrate",
    "temp",
]

wallaroo = client.get_activity_streams(7929221431, types=types, resolution="high")
moonta = client.get_activity_streams(
    activ_list[0].to_dict()["id"], types=types, resolution="high"
)

wallaroo_hr = wallaroo["heartrate"].data
moonta_hr = moonta["heartrate"].data
# hr_vals = np.histogram(wallaroo_hr, bins=20)
# print(hr_vals)

# with open("wallaroo_hr.pickle", "wb") as f:
#     pickle.dump(wallaroo_hr, f)
# with open("moonta_hr.pickle", "wb") as f:
#     pickle.dump(moonta_hr, f)
# with open("wallaroo_activity.pickle", "wb") as f:
#     pickle.dump(most_recent, f)


# exit()
# list(activities)[0:10]
# my_cols = [
#     "name",
#     "start_date_local",
#     "type",
#     "distance",
#     "moving_time",
#     "elapsed_time",
#     "total_elevation_gain",
#     "elev_high",
#     "elev_low",
#     "average_speed",
#     "max_speed",
#     "average_heartrate",
#     "max_heartrate",
#     "start_latitude",
#     "start_longitude",
# ]

# data = []
# for activity in activities:
#     my_dict = activity.to_dict()
#     data.append([activity.id] + [my_dict.get(x) for x in my_cols])

# # Add id to the beginning of the columns, used when selecting a specific activity
# my_cols.insert(0, "id")

# df = pd.DataFrame(data, columns=my_cols)  # Make all walks into hikes for consistency
# df["type"] = df["type"].replace("Walk", "Hike")  # Create a distance in km column
# df["distance_km"] = df["distance"] / 1e3  # Convert dates to datetime type
# df["start_date_local"] = pd.to_datetime(
#     df["start_date_local"]
# )  # Create a day of the week and month of the year columns
# df["day_of_week"] = df["start_date_local"].dt.day_name()
# df["month_of_year"] = df["start_date_local"].dt.month  # Convert times to timedeltas
# df["moving_time"] = pd.to_timedelta(df["moving_time"])
# df["elapsed_time"] = pd.to_timedelta(
#     df["elapsed_time"]
# )  # Convert timings to hours for plotting
# df["elapsed_time_hr"] = df["elapsed_time"].astype(int) / 3600e9
# df["moving_time_hr"] = df["moving_time"].astype(int) / 3600e9

# day_of_week_order = [
#     "Monday",
#     "Tuesday",
#     "Wednesday",
#     "Thursday",
#     "Friday",
#     "Saturday",
#     "Sunday",
# ]

# print(df["day_of_week"])
# print(np.shape(df["day_of_week"]))

# plt.figure()
# plt.scatter(df["day_of_week"], df["distance_km"])
# plt.show()
# exit()


# g = sns.catplot(
#     x="day_of_week",
#     y="distance_km",
#     kind="strip",
#     data=df,
#     order=day_of_week_order,
#     col="type",
#     height=4,
#     aspect=0.9,
#     palette="pastel",
# )
# (
#     g.set_axis_labels("Week day", "Distance (km)")
#     .set_titles("Activity type: {col_name}")
#     .set_xticklabels(rotation=30)
# )

# exit()

# # url = client.authorization_url(
# #     client_id=MY_STRAVA_CLIENT_ID,
# #     redirect_uri="http://127.0.0.1:5000/authorization",
# #     scope=["read_all", "profile:read_all", "activity:read_all"],
# # )
# # print(url)


# # CODE = "197d416159f3c7eebd3d5c3ee3d39cd449121471"
# # access_token = client.exchange_code_for_token(
# #     client_id=MY_STRAVA_CLIENT_ID, client_secret=MY_STRAVA_CLIENT_SECRET, code=CODE
# # )

# # with open("access_token.pickle", "wb") as f:
# #     pickle.dump(access_token, f)


# # ['id', 'guid', 'external_id', 'upload_id', 'athlete', 'name', 'distance', 'moving_time', 'elapsed_time', 'total_elevation_gain', 'elev_high', 'elev_low', 'type', 'start_date', 'start_date_local', 'timezone', 'utc_offset', 'start_latlng', 'end_latlng', 'location_city', 'location_state', 'location_country', 'start_latitude', 'start_longitude', 'achievement_count', 'pr_count', 'kudos_count', 'comment_count', 'athlete_count', 'photo_count', 'total_photo_count', 'map', 'trainer', 'commute', 'manual', 'private', 'flagged', 'gear_id', 'gear', 'average_speed', 'max_speed', 'device_watts', 'has_kudoed', 'best_efforts', 'segment_efforts', 'splits_metric', 'splits_standard', 'average_watts', 'weighted_average_watts', 'max_watts', 'suffer_score', 'has_heartrate', 'average_heartrate', 'max_heartrate', 'average_cadence', 'kilojoules', 'average_temp', 'device_name', 'embed_token', 'calories', 'description', 'workout_type', 'photos', 'instagram_primary_photo', 'partner_logo_url', 'partner_brand_tag', 'from_accepted_tag', 'segment_leaderboard_opt_out', 'highlighted_kudosers', 'laps', 'resource_state']
