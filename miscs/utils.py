import json
import os
import re
import time

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import multivariate_normal
from collections import Counter
from simulator.engine.evaluation.metrics import *
import math
import datetime
from datetime import datetime, timedelta


def haversine(lat1, lon1, lat2, lon2):
    # Radius of the Earth in kilometers
    R = 6371.0

    # Convert latitude and longitude from degrees to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    # Differences in coordinates
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    # Haversine formula
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # Distance in kilometers
    distance = R * c

    return distance


def is_weekday_or_weekend(date_string):
    # Convert the input date string to a datetime object
    date = datetime.strptime(date_string, "%Y-%m-%d")

    # Check the day of the week (0 = Monday, 6 = Sunday)
    day_of_week = date.weekday()

    if day_of_week < 5:  # 0 to 4 represent weekdays (Monday to Friday)
        # print(f'{date_string} is a weekday (Mon-Fri)')
        return 0
    else:  # 5 and 6 represent the weekend (Saturday and Sunday)
        # print(f'{date_string} is a weekend (Sat-Sun)')
        return 1


def find_detail_weekday(date_string):
    if is_weekday_or_weekend(date_string) == 0:
        return "weekday"
    else:
        return "weekend"


def get_days_of_week(date_string):
    # Convert the input date string to a datetime object
    date = datetime.datetime.strptime(date_string, "%Y-%m-%d")

    # Define a list of days of the week
    days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    # Get the day of the week as an integer (0 to 6, where 0 is Monday and 6 is Sunday)
    day_of_week_index = date.weekday()

    # Return the day of the week as a string
    return days_of_week[day_of_week_index]


def evaluate(real_activities, infer_activities, metric, class_loc_map=None):
    if metric == "semantic":
        return semantic_similarity_bert(real_activities, infer_activities).cpu().numpy().reshape(-1)[0]
    elif metric == "mat":
        assert class_loc_map is not None
        return act_mat_compute(real_activities.split(": ")[-1], infer_activities.split(": ")[-1], class_loc_map)


def extract_word(text):
    # Regular expression pattern to match text within double curly braces
    pattern = r"\*([^*]+)\*"

    # Using re.search to find the first occurrence
    match = re.search(pattern, text)

    # If a match is found, return the captured group; otherwise return None
    return match.group(1) if match else None


def extract_lat_log(s):
    pattern = r'\((-?\d+\.\d+), (-?\d+\.\d+)\)'

    # Use re.search to find the first match of the pattern in the input string
    match = re.search(pattern, s)

    # Check if a match was found
    if match:
        latitude = float(match.group(1))
        longitude = float(match.group(2))
        return latitude, longitude
    else:
        print(f'No latitude or longitude found in {s}')
        return None, None


def extract_lat_log_all(s):
    pattern = r'\((-?\d+\.\d+), (-?\d+\.\d+)\)'
    lat_logs = []
    # Use re.search to find the first match of the pattern in the input string
    matches = re.findall(pattern, s)
    for match in matches:
        latitude = float(match[0])
        longitude = float(match[1])
        lat_logs.append((latitude, longitude))
    return lat_logs


def vis_traj_heatmap(routine_lat_log, net_grids):
    # Create a DataFrame from the coordinates
    df = pd.DataFrame(routine_lat_log, columns=['Latitude', 'Longitude'])

    # Create a heatmap
    sns.kdeplot(data=df, x='Latitude', y='Longitude', cmap='YlOrRd', fill=True)
    plt.title('Spatial Heatmap')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    net_grids = np.array(net_grids)
    plt.xlim(np.min(net_grids[:, 0]), np.max(net_grids[:, 0]))
    plt.ylim(np.min(net_grids[:, 1]), np.max(net_grids[:, 1]))
    # Show the plot
    plt.show()


def spatial_mvn(routine_lat_log):
    data = np.array(routine_lat_log)

    # Calculate the mean and covariance matrix of the data
    mean = np.mean(data, axis=0)
    cov_matrix = np.cov(data, rowvar=False)

    # Create a multivariate normal distribution based on the calculated mean and covariance
    mvn = multivariate_normal(mean=mean, cov=cov_matrix)

    return mvn


def sample_with_spatial_mvn_from_city(mvn, net_grids, num_samples):
    probabilities = mvn.pdf(net_grids)

    # Normalize probabilities to make them sum to 1
    normalized_probabilities = probabilities / np.sum(probabilities)

    samples_indices = np.random.choice(len(net_grids), size=num_samples, p=normalized_probabilities, replace=False)
    samples = [net_grids[i] for i in samples_indices]

    return samples


def get_location_string(loc_dict, routines):
    loc_string = ''
    k = 1
    # Define a regular expression pattern to extract location name and coordinates
    pattern = r'(\w+(?: \w+)*?) \(([-+]?\d*\.\d+), ([-+]?\d*\.\d+)\)'

    # Use regex to find all matches of the pattern in the input string
    matches = re.findall(pattern, routines)

    # Extracted location names and their coordinates
    locations = [(name, str(lat), str(lng)) for name, lat, lng in matches]

    # Print the extracted location names and coordinates
    for i, (name, lat, lng) in enumerate(locations, start=1):
        key = f"{name} ({lat}, {lng})".replace("'", "")
        if 'free Shop' in key and 'Duty' not in key:
            key = "Duty-" + key
        if k == len(matches):
            loc_string += f"{loc_dict[key]}" + '.' + '\n'
        else:
            try:
                loc_string += f"{loc_dict[key]}" + ', '
            except:
                print()
        k += 1
    # loc_string += f"{k}. Home @ ({loc[0]}, {loc[1]})" + '\n'
    return loc_string.strip()


def get_representative_routine_string(rpr):
    routine_string = ''
    for r in rpr:
        routine_string += r + '\n'
    return routine_string.strip()


def shorten_representative_routine_string_bk(routine_string, location_map):
    for k, v in location_map.items():
        if k in routine_string:
            routine_string_ = routine_string
            routine_string = routine_string.replace(k, v)
            if "Turkish Restaurant#949" in routine_string:
                print()
    return routine_string.strip()


def shorten_representative_routine_string(routine_string, location_map):
    for k, v in location_map.items():
        if ": " + k in routine_string:
            routine_string = routine_string.replace(": " + k, ": " + v)

        if ", " + k in routine_string:
            routine_string = routine_string.replace(", " + k, ", " + v)
            continue
        if ":" + k in routine_string:
            routine_string = routine_string.replace(":" + k, ":" + v)

        if "," + k in routine_string:
            routine_string = routine_string.replace("," + k, "," + v)
            continue
    if "(" in routine_string and ")" in routine_string:
        print(routine_string)
        assert False
    return routine_string.strip()


def prompts_to_df(prompts, scores, result):
    """Converts a list of prompts into a dataframe."""
    df = pd.DataFrame()
    df['Prompt'] = prompts
    df['scores'] = scores
    df['result'] = result
    # df['log(p)'] = df['log(p)'].apply(lambda x: round(x, 3))  # Round the scores to 3 decimal places
    # df = df.head(15)  # Only show the top 15 prompts
    return df


def clean_files_in_folder(path, type='.txt'):
    files = os.listdir(path)
    # Iterate through the files and delete .txt files
    for file in files:
        if file.endswith(type):
            file_path = os.path.join(path, file)
            try:
                os.remove(file_path)
                print(f"Deleted {file_path}")
            except OSError as e:
                print(f"Error while deleting {file_path}: {e.strerror}")


def list_to_file(res_list, file_name, list_in_list=True):
    with open(file_name, 'w', encoding='iso-8859-1') as file:
        # Write each element from the list to the file, one per line
        if list_in_list:
            for item in res_list:
                file.write(', '.join(item) + '\n')
        else:
            for item in res_list:
                file.write(item + '\n')


def mask_time(activities):
    replacement_string = ' <INPUT>'
    pattern_to_replace = r' ([A-Za-z ]+#\d+) at (\d{2}:\d{2}:\d{2})'
    activities_masked = re.sub(pattern_to_replace, r'\1 at ' + replacement_string, activities)

    return activities_masked


def mask_space(activities):
    replacement_string = '<INPUT> '
    pattern_to_replace = r'([A-Za-z ]+#\d+) '
    activities_masked = re.sub(pattern_to_replace, replacement_string, activities)

    return activities_masked


def mask_all(activities):
    replacement_string = '<INPUT>'
    pattern_to_replace = r'[A-Za-z ]+#\d+ at \d{2}:\d{2}:\d{2}'
    activities_masked = re.sub(pattern_to_replace, replacement_string, activities)
    return activities_masked


def match_counts(s1, s2):
    c = 0
    s1 = s1.split(": ")[-1].split(", ")
    s2 = s2.split(": ")[-1].split(", ")
    for ss1, ss2 in zip(s1, s2):
        if ss1 == ss2:
            c += 1
    return c


def load_res_json(contents):
    try:
        res = json.loads(contents)
    except Exception as e:
        print(contents)
        print(e)
        contents_utf8 = contents.decode('iso-8859-1')
        res = json.loads(contents_utf8)
    return res


def top_k_occurrences(arr, k):
    # Count the occurrences of each element in the list
    counts = Counter(arr)

    # Sort elements by occurrences in descending order
    sorted_keys = sorted(counts, key=counts.get, reverse=True)

    # Select the top k elements
    top_k_elements = sorted_keys[:k]

    return top_k_elements, counts


def get_referenced_doc(response):
    # Get the referenced document when using llmaindex (llamachain) as the retrieval engine
    if hasattr(response, 'metadata'):
        document_info = str(response.metadata)
        find = re.findall(r"'page_label': '[^']*', 'file_name': '[^']*'", document_info)

        print('\n' + '=' * 60 + '\n')
        print('Context Information')
        print(str(find))
        print('\n' + '=' * 60 + '\n')


def llama_token_counter(token_counter):
    print(
        "Embedding Tokens: ",
        token_counter.total_embedding_token_count,
        "\n",
        "LLM Prompt Tokens: ",
        token_counter.prompt_llm_token_count,
        "\n",
        "LLM Completion Tokens: ",
        token_counter.completion_llm_token_count,
        "\n",
        "Total LLM Token Count: ",
        token_counter.total_llm_token_count,
        "\n",
    )


def remove_duplicates_keep_order(seq):
    seen = set()
    result = []
    for item in seq:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def collect_loc_time(contents):
    time_pattern = r'\d{2}:\d{2}:\d{2}'

    # Regular expression for locations
    location_pattern = r'\w+#\d+'

    # Find all matches for times and locations
    times = re.findall(time_pattern, contents)
    times = remove_duplicates_keep_order(times)
    locations = re.findall(location_pattern, contents)
    locations = remove_duplicates_keep_order(locations)

    # Pairing times with locations
    time_location_pairs = list(zip(times, locations))
    plan = [f"{loc} at {t}" for t, loc in time_location_pairs]
    # plan = remove_duplicates_keep_order(plan)
    # plan = sorted(plan, key=lambda x: x.split(' ')[0])

    return plan


def first2second(words):
    words = words.replace("I ", "You ")
    words = words.replace(", I ", ", you ")
    words = words.replace(". I ", ". You ")
    words = words.replace(" I ", " you ")
    words = words.replace("my ", "your ")
    words = words.replace(" am ", " are ")
    words = words.replace("My ", "Your ")
    words = words.replace(" me ", " you ")
    words = words.replace(" myself ", " yourself ")
    return words


def check_consecutive_dates(plans, date_):
    # Check for consecutive days in reverse order and break once a non-consecutive day is found
    dates = [datetime.strptime(activity.split(":")[0].split("at")[1].strip(), "%Y-%m-%d") for activity in plans]
    dates.append(datetime.strptime(date_, "%Y-%m-%d"))
    max_streak = 1
    current_streak = 1
    for i in range(len(dates) - 1, 0, -1):
        if dates[i] - dates[i - 1] == timedelta(days=1):
            current_streak += 1
        else:
            break  # Break once a non-consecutive day is found

    max_streak = current_streak
    return max_streak


def count_replacements(original, replacement, test):
    # Original and replacement strings

    # The difference in length between the original and the replacement
    length_difference = len(replacement) - len(original)

    # Replace "Bed and Breakfast" with "small lodging establishment"
    modified_test = test.replace(original, replacement)

    # Calculate the number of replacements
    num_replacements = (len(modified_test) - len(test)) // length_difference
    return num_replacements
