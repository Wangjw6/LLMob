import numpy as np
from geopy.distance import geodesic
import pandas as pd


def softmax(x, temperature=2.):
    """Compute softmax values for each set of scores in x."""
    x = x / temperature
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Shift for numerical stability
    return e_x / np.sum(e_x, axis=1, keepdims=True)



def commute_loc(route):
    pattern = r"([\w\s\/]+) \(([\d.]+), ([\d.]+)\)"
    # Find all matches and convert them into the desired structure
    locations = [(match[0].strip(), (str(match[1]), str(match[2]))) for match in re.findall(pattern, route)]
    distances = []
    for i in range(len(locations) - 1):
        start = (float(locations[i][1][0]), float(locations[i][1][1]))
        end = (float(locations[i + 1][1][0]), float(locations[i + 1][1][1]))
        distance = geodesic(start, end).kilometers
        # print(distance)
        distances.append((locations[i][0] + " to " + locations[i + 1][0], distance))
    return locations[0], locations[-1], np.sum([dist for _, dist in distances])


def commute_time(route):
    pattern = r'\bat\s(\d{2}:\d{2})'
    # Find all matches for the pattern
    times = re.findall(pattern, route)
    # Convert string times to datetime objects for analysis
    times = [datetime.strptime(time, "%H:%M") for time in times]

    # Analyze the time span
    start_time = times[0]
    end_time = times[-1]
    start_time = datetime.strptime(str(start_time), "%Y-%m-%d %H:%M:%S").time()
    end_time = datetime.strptime(str(end_time), "%Y-%m-%d %H:%M:%S").time()
    return start_time, end_time


def spatial_temporal_preference(person):
    # activity_pattern = r"([\w\s\/]+) \([\d.]+, [\d.]+\) at (\d{2}:\d{2}:\d{2})"
    activity_pattern = r"(\w[\w\s/]*?) \(([\d.]+), ([\d.]+)\) at ([\d:]+)"
    all_activities = []
    for route in person.raw_train_routine_list:
        activities = re.findall(activity_pattern, route)
        cleaned_activities = [tuple(item.replace("'", "") for item in match) for match in activities]
        # activities_no_spaces = [(location.strip(), time) for (location, time) in activities]
        activities_no_spaces = [(person.loc_map[location + f" ({lat}, {lng})"], time) for (location, lat, lng, time) in
                                cleaned_activities]
        if len(activities_no_spaces) > 0:
            all_activities.extend(activities_no_spaces)
    # Convert times to datetime objects and create a DataFrame
    df_activities = pd.DataFrame(all_activities, columns=['Location', 'Time'])
    try:
        df_activities['Time'] = pd.to_datetime(df_activities['Time'], format='%H:%M:%S')
    except:
        df_activities['Time'] = pd.to_datetime(df_activities['Time'], format='%H:%M')

    # Create time intervals (10 minutes each)
    interval_ = 30
    time_intervals = pd.date_range('00:00:00', '23:50:00', freq=f'{interval_}T').time

    # Initialize a DataFrame to represent the matrix
    matrix_df = pd.DataFrame(0, index=df_activities['Location'].unique(), columns=time_intervals)

    # Fill the matrix
    for _, row in df_activities.iterrows():
        # Find the closest interval_-minute interval to the activity time
        interval = str(row['Time'].replace(second=0, microsecond=0))
        interval = datetime.strptime(interval, "%Y-%m-%d %H:%M:%S").time()
        if interval.minute % interval_ != 0:
            interval = interval.replace(minute=(interval.minute // interval_) * interval_)
        # Mark the activity in the matrix
        matrix_df.at[row['Location'], interval] += 1
    top_5_visit = matrix_df.sum(axis=1).sort_values(ascending=False).head(5)
    base = ', '.join(top_5_visit.index.tolist()).lower()

    prompt_set = []
    for loc in top_5_visit.index:
        prompt2 = f"{loc} at {matrix_df.loc[loc].idxmax()}"
        prompt_set.append(prompt2)
    prompt_string = "You usually visit " + ", ".join(prompt_set)
    return prompt_string


def extract_knowledge(person):
    prompt = ""
    knowledge = {"commute_dist": [], "commute_duration": [], "commute_begin_time": [], "commute_end_time": [],
                 "commute_begin_loc": [], "commute_end_loc": []}
    for route in person.raw_train_routine_list:
        date_ = route.split(": ")[0].split(" ")[-1]
        date_obj = datetime.strptime(date_, '%Y-%m-%d')
        if date_obj.weekday() >= 5:
            continue
        begin_loc, end_loc, dists_ = commute_loc(route)
        begin_time, end_time = commute_time(route)
        knowledge["commute_dist"].append(dists_)
        knowledge["commute_duration"].append(end_time.hour * 60 + end_time.minute -
                                             begin_time.hour * 60 - begin_time.minute)
        knowledge["commute_begin_time"].append(begin_time)
        knowledge["commute_end_time"].append(end_time)

        knowledge["commute_begin_loc"].append(person.loc_map[begin_loc[0] + ' ' + str(begin_loc[1]).replace("'", "")])
        knowledge["commute_end_loc"].append(person.loc_map[end_loc[0] + ' ' + str(end_loc[1]).replace("'", "")])
    dist_tot = -1
    if len(knowledge["commute_dist"]) > 0:
        dist_tot = int(np.mean(knowledge["commute_dist"]) / 10) * 10
        begin_time_max = max(knowledge["commute_begin_time"], key=knowledge["commute_begin_time"].count)
        end_time_max = max(knowledge["commute_end_time"], key=knowledge["commute_end_time"].count)
        begin_loc_max = max(knowledge["commute_begin_loc"], key=knowledge["commute_begin_loc"].count)

        end_loc_max = max(knowledge["commute_end_loc"], key=knowledge["commute_end_loc"].count)
    if dist_tot < 0:
        prompt += f" During weekday, you don't like to travel, "
    else:
        prompt += f" During weekday, you usually travel over {dist_tot} kilometers a day, "
        prompt += f"you usually begin your daily trip at {begin_time_max} and end your daily trip at {end_time_max},"
        prompt += (f" you usually visit {begin_loc_max} at the beginning of the day and go to {end_loc_max} before "
                   f"returning home. ")

    knowledge = {"commute_dist": [], "commute_duration": [], "commute_begin_time": [], "commute_end_time": [],
                 "commute_begin_loc": [], "commute_end_loc": []}
    for route in person.history_routine.split("\n"):
        date_ = route.split(": ")[0].split(" ")[-1]
        date_obj = datetime.strptime(date_, '%Y-%m-%d')
        if date_obj.weekday() < 5:
            continue
        begin_loc, end_loc, dists_ = commute_loc(route)
        begin_time, end_time = commute_time(route)
        knowledge["commute_dist"].append(dists_)
        knowledge["commute_duration"].append(end_time.hour * 60 + end_time.minute -
                                             begin_time.hour * 60 - begin_time.minute)
        knowledge["commute_begin_time"].append(begin_time)
        knowledge["commute_end_time"].append(end_time)

        knowledge["commute_begin_loc"].append(person.loc_map[begin_loc[0] + ' ' + str(begin_loc[1]).replace("'", "")])
        knowledge["commute_end_loc"].append(person.loc_map[end_loc[0] + ' ' + str(end_loc[1]).replace("'", "")])
    dist_tot = -1
    if len(knowledge["commute_dist"]) > 0:
        dist_tot = int(np.mean(knowledge["commute_dist"]) / 10) * 10
        begin_time_max = max(knowledge["commute_begin_time"], key=knowledge["commute_begin_time"].count)
        end_time_max = max(knowledge["commute_end_time"], key=knowledge["commute_end_time"].count)
        begin_loc_max = max(knowledge["commute_begin_loc"], key=knowledge["commute_begin_loc"].count)

        end_loc_max = max(knowledge["commute_end_loc"], key=knowledge["commute_end_loc"].count)
    if dist_tot < 0:
        prompt += f" During weekend, you don't like to travel, "
    else:
        prompt += f" During weekend, you usually travel over {dist_tot} kilometers a day, "
        prompt += f"you usually begin your daily trip at {begin_time_max} and end your daily trip at {end_time_max},"
        prompt += (f" you usually visit {begin_loc_max} at the beginning of the day and go to {end_loc_max} before "
                   f"returning home. ")

    prompt += spatial_temporal_preference(person)

    return prompt


from datetime import timedelta



def change_interval_to_time(t, interval=10):
    time_duration = timedelta(minutes=int(t) * interval)
    # Extract hours and minutes
    hours, remainder = divmod(time_duration.seconds, 3600)
    minutes = remainder // 60
    # Format the time
    formatted_time = f'{hours:02d}:{minutes:02d}'
    return formatted_time


import re
def clean_traj(traj):
    acts = traj.split(": ")[-1]
    acts = acts.replace(", ", " at ")
    acts = acts.replace("Indulge in ", "")
    acts = acts.replace("Try in ", "")
    acts = acts.replace("Grab a quick bite at ", "")
    acts = acts.replace("Try ", "")
    acts = acts.replace("Car Dealership", "Auto Dealership")
    acts = acts.replace("Enjoy ", "")
    acts = acts.replace("Mall", "Shopping Mall")
    acts = acts.replace("Outlet Shopping Mall", "Shopping Mall")
    acts = acts.replace("Shopping Shopping Mall", "Shopping Mall")
    acts = acts.replace("Relax at ", "")
    acts = acts.replace("Experience ", "")
    acts = acts.replace("Ramem Restaurant", "Ramen Restaurant")
    acts = acts.replace("Bed and Breakfast#885", "small lodging establishment#885")
    acts = acts.replace("Discover ", "")
    acts = acts.replace("Drop by ", "")
    acts = acts.replace("Stop by ", "")
    acts = acts.replace("End the day at ", "")
    acts = acts.replace("Visit ", "")
    acts = acts.replace("Go to ", "")
    acts = acts.replace("Sip coffee at ", "")
    acts = acts.replace("Noodle Restaurant", "Noodle House")
    acts = acts.replace("Explore ", "")
    acts = acts.replace("Visit ", "")
    acts = acts.replace("Shopping at ", "")
    acts = acts.replace("Lunch at ", "")
    acts = acts.replace("Lunch Break ", "Sandwich Shop#1 ")
    acts = acts.replace("Office District", "Office")

    acts = acts.replace("Head to ", "")
    acts = acts.replace("Savor ", "")
    acts = acts.replace("Discover ", "")
    if len(acts)==0:
        print(55)
    return acts

from datetime import datetime
def calculate_intervals_to_midnight(times, interval=10):
    midnight = datetime.strptime('00:00:00', '%H:%M:%S')
    intervals = []
    for time in times:
        if time.strip('.') == "24:00":
            time =  "23:59"
        try:
            current_time = datetime.strptime(time.strip('.'), '%H:%M:%S')
        except:
            current_time = datetime.strptime(time.strip('.'), '%H:%M')
        # Calculate the time difference in seconds and convert to minutes
        time_diff_minutes = (current_time - midnight).seconds / 60
        # Calculate the number of 10-minute intervals
        number_of_intervals = time_diff_minutes // interval
        intervals.append(int(number_of_intervals))
    return intervals

def valid_generation(person, traj):

    cat = person.cat
    traj_acts = clean_traj(traj)
    loc_times = traj_acts.split(" at ")
    locs = []
    times = []
    acts = []
    k = 0

    while k < len(loc_times):
        loc_times[k] = loc_times[k].replace(".", "")
        if k % 2 == 0:
            try:
                clean_loc = loc_times[k]
            except:
                clean_loc = clean_loc.split("#")[0] + str(1)
                print(clean_loc)
            if "Home" in clean_loc or "home" in clean_loc:
                k += 2
                continue
            try:
                acts.append(cat[loc_times[k].split("#")[0].strip()])
            except Exception as e:
                print(e)
                assert False
            locs.append(loc_times[k])

            if loc_times[k].split("#")[0].strip() not in cat:
                acts.append("")
                print(loc_times[k].split("#")[0])
                print(traj)
                assert False
            #                 acts.append(cat[loc_times[k].split("#")[0]])
            if acts[-1] == ('Outdoors & Recreation',):
                print(loc_times[k])
                assert False
        else:
            times.append(loc_times[k].split(" ")[0])
        k += 1

    assert len(locs) == len(times), locs
    assert len(acts) == len(times), locs
    assert len(locs) > 0

    return True



