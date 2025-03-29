import pickle
import numpy as np
import scipy.stats
import math
from datetime import datetime
from math import sin, cos, asin, sqrt, radians
import os

import argparse


def invert_dict(d):
    return {value: key for key, value in d.items()}


def load_pickle(file_path):
    """Utility function to load a pickle file."""
    with open(file_path, 'rb') as f:
        return pickle.load(f)


# Key: location name + latitude + longitude
# Value: ID in the city network
pos_map = load_pickle('./data/pos_map.pkl')

# Key: "location name + latitude + longitude" (to ensure uniqueness)
# Value: "location name + a unique ID for this location (same name locations get different IDs)"
loc_map = load_pickle('./data/loc_map.pkl')

# Key: location name, Value: category from foursquare
cat = load_pickle('./data/location_activity_map.pkl')
map_loc = invert_dict(loc_map)


def geodistance(lng1, lat1, lng2, lat2):
    lng1, lat1, lng2, lat2 = map(radians, [float(lng1), float(lat1), float(lng2), float(lat2)])
    dlon = lng2 - lng1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    distance = 2 * asin(sqrt(a)) * 6371 * 1000
    distance = round(distance / 1000, 3)
    return distance


def calculate_intervals_to_midnight(times, interval=10):
    midnight = datetime.strptime('00:00:00', '%H:%M:%S')
    intervals = []
    for time in times:
        if time.strip('.') == "24:00":
            time = "23:59"
        try:
            current_time = datetime.strptime(time.strip('.'), '%H:%M:%S')
        except:
            current_time = datetime.strptime(time.strip('.'), '%H:%M')
        time_diff_minutes = (current_time - midnight).seconds / 60
        number_of_intervals = time_diff_minutes // interval
        intervals.append(number_of_intervals)
    return intervals


def clean_traj(traj):
    acts = traj.split(": ")[-1]
    acts = acts.replace(", ", " at ").replace("Indulge in ", "").replace("Try in ", "").replace("Grab a quick bite at ",
                                                                                                "").replace("Try ", "")
    acts = acts.replace("Car Dealership", "Auto Dealership").replace("Enjoy ", "").replace("Mall", "Shopping Mall")
    acts = acts.replace("Outlet Shopping Mall", "Shopping Mall").replace("Shopping Shopping Mall", "Shopping Mall")
    acts = acts.replace("Relax at ", "").replace("Experience ", "").replace("Ramem Restaurant", "Ramen Restaurant")
    acts = acts.replace("Drop by ", "").replace("Stop by ", "").replace("End the day at ", "").replace("Visit ", "")
    acts = acts.replace("Go to ", "").replace("Sip coffee at ", "").replace("Noodle Restaurant", "Noodle House")
    acts = acts.replace("Explore ", "").replace("Visit ", "").replace("Shopping at ", "").replace("Lunch at ", "")
    acts = acts.replace("Savor ", "").replace("Discover ", "")
    return acts


def duration(p):
    d = [[i[0] - u[index][0] for index, i in enumerate(u[1:])] for u in p]
    d = [round(i * 10) for u in d for i in u]
    return d


def obtain_analysis_traj(data):
    traj_ids = []
    traj_lat_lngs = []
    traj_act_ts = []
    for d, traj in data.items():
        traj = data[d]
        if ": : " in traj:
            traj = traj.replace(": : ", ": ")
        if " :" in traj:
            traj = traj.replace(", ", "")
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
                    print(traj)
                    k += 2
                    continue
                locs.append(loc_times[k])

            else:
                times.append(loc_times[k].split(" ")[0])
            k += 1
        times_interval = calculate_intervals_to_midnight(times)
        traj_id, traj_lat_lng, traj_act_t = [], [], []
        for i in range(len(locs)):
            if "Home" in locs[i] or "home" in locs[i]:
                continue
            try:
                loc_with_lat_lng = map_loc[locs[i].strip()]
            except:
                continue
            loc_with_lat_lng_ = loc_with_lat_lng.replace(" (", ", ")
            loc_with_lat_lng_ = loc_with_lat_lng_.replace(")", "")
            lat_lng = [float(loc_with_lat_lng_.split(", ")[1]), float(loc_with_lat_lng_.split(", ")[2])]
            loc_id = pos_map[loc_with_lat_lng]
            t = times_interval[i]
            traj_id.append([loc_id, t])
            traj_act_t.append([acts[i], t])
            traj_lat_lng.append([lat_lng[0], lat_lng[1], t])
        traj_ids.append(traj_id)
        traj_act_ts.append(traj_act_t)
        traj_lat_lngs.append(traj_lat_lng)
    return traj_ids, traj_lat_lngs, traj_act_ts


p2id = {'Travel & Transport': 0, 'Food': 1, 'Shop & Service': 2,
        'Nightlife Spot': 3, 'Arts & Entertainment': 4, 'Professional & Other Places': 5,
        'Outdoors & Recreation': 6,
        'College & University': 7, 'Residence': 8, 'Event': 9}


def transfer(data):
    transfer_data = []
    locs_id = data[0]
    lat_lngs = data[1]
    acts = data[2]
    for i in range(len(locs_id)):
        this_day = []
        for j in range(len(lat_lngs[i])):
            this_day.append([locs_id[i][j][1], p2id[acts[i][j][0]], [lat_lngs[i][j][0], lat_lngs[i][j][1]]])
        sorted_this_day = sorted(this_day, key=lambda x: x[0])
        transfer_data.append(sorted_this_day)
    return transfer_data


class Evaluation(object):
    def __init__(self, args):
        self.args = args

    def arr_to_distribution(self, arr, Min, Max, bins):
        distribution, base = np.histogram(arr[arr <= Max], bins=bins, range=(Min, Max))
        m = np.array([len(arr[arr > Max])], dtype='int64')
        distribution = np.hstack((distribution, m))
        return distribution, base[:-1]

    def get_js_divergence(self, p1, p2):
        p1 = p1 / (p1.sum() + 1e-9)
        p2 = p2 / (p2.sum() + 1e-9)
        m = (p1 + p2) / 2
        js = 0.5 * scipy.stats.entropy(p1, m) + 0.5 * scipy.stats.entropy(p2, m)
        return js

    def distance_one_step(self, p1, p2):
        f = [geodistance(i[2][0], i[2][1], u[index][2][0], u[index][2][1]) for u in p1 for index, i in enumerate(u[1:])]
        r = [geodistance(i[2][0], i[2][1], u[index][2][0], u[index][2][1]) for u in p2 for index, i in enumerate(u[1:])]
        MIN = 0
        MAX = 10
        bins = math.ceil(MAX - MIN)
        r_list, _ = self.arr_to_distribution(np.array(r), MIN, MAX, bins)
        f_list, _ = self.arr_to_distribution(np.array(f), MIN, MAX, bins)
        r_list = r_list / (r_list.sum() + 1e-9)
        f_list = f_list / (f_list.sum() + 1e-9)
        JSD = self.get_js_divergence(r_list, f_list)
        return JSD

    def st_act_jsd(self, p1, p2):
        st_act_dict = {}
        for u in p1:
            for i in u:
                if str(i[0]) + '_' + str(i[1]) not in st_act_dict:
                    st_act_dict[str(i[0]) + '_' + str(i[1])] = len(st_act_dict)
        for u in p2:
            for i in u:
                if str(i[0]) + '_' + str(i[1]) not in st_act_dict:
                    st_act_dict[str(i[0]) + '_' + str(i[1])] = len(st_act_dict)
        f, r = [], []
        for u in p1:
            for i in u:
                f.append(st_act_dict[str(i[0]) + '_' + str(i[1])])
        for u in p2:
            for i in u:
                r.append(st_act_dict[str(i[0]) + '_' + str(i[1])])
        MIN = np.min(r + f)
        MAX = np.max(r + f)
        bins = 1000
        r = (np.array(r) - MIN) / (MAX - MIN)
        f = (np.array(f) - MIN) / (MAX - MIN)
        r_list, _ = self.arr_to_distribution(r, 0, 1, bins)
        f_list, _ = self.arr_to_distribution(f, 0, 1, bins)
        JSD = self.get_js_divergence(r_list, f_list)
        return JSD

    def st_loc_jsd(self, p1, p2):
        st_act_dict = {}
        for u in p1:
            for i in u:
                if str(i[0]) + '_' + str(i[2][0]) + '_' + str(i[2][1]) not in st_act_dict:
                    st_act_dict[str(i[0]) + '_' + str(i[2][0]) + '_' + str(i[2][1])] = len(st_act_dict)
        for u in p2:
            for i in u:
                if str(i[0]) + '_' + str(i[2][0]) + '_' + str(i[2][1]) not in st_act_dict:
                    st_act_dict[str(i[0]) + '_' + str(i[2][0]) + '_' + str(i[2][1])] = len(st_act_dict)
        f, r = [], []
        for u in p1:
            for i in u:
                f.append(st_act_dict[str(i[0]) + '_' + str(i[2][0]) + '_' + str(i[2][1])])
        for u in p2:
            for i in u:
                r.append(st_act_dict[str(i[0]) + '_' + str(i[2][0]) + '_' + str(i[2][1])])
        MIN = np.min(r + f)
        MAX = np.max(r + f)
        bins = 400
        r = (np.array(r) - MIN) / (MAX - MIN)
        f = (np.array(f) - MIN) / (MAX - MIN)
        r_list, _ = self.arr_to_distribution(np.array(r), 0, 1, bins)
        f_list, _ = self.arr_to_distribution(np.array(f), 0, 1, bins)
        JSD = self.get_js_divergence(r_list, f_list)
        return JSD

    def duration_jsd(self, p1, p2):
        f = duration(p1)
        r = duration(p2)
        MIN = 0
        MAX = 12
        bins = math.ceil(MAX - MIN)
        r_list, _ = self.arr_to_distribution(np.array(r), MIN, MAX, bins)
        f_list, _ = self.arr_to_distribution(np.array(f), MIN, MAX, bins)
        JSD = self.get_js_divergence(r_list, f_list)
        return JSD

    def get_JSD(self, real, fake):
        duration_jsd = self.duration_jsd(real, fake)
        distance_step = self.distance_one_step(real, fake)
        st_act_jsd = self.st_act_jsd(real, fake)
        st_loc_jsd = self.st_loc_jsd(real, fake)
        return duration_jsd, distance_step, st_act_jsd, st_loc_jsd


def eval(dataset='normal', mode=0):
    # Load required data
    mode_name = {0: "llm_l", 1: "llm_e"}
    mode = mode_name[mode]
    truth = {}
    person_to_test = []
    scenario_tag = {
        '2019': 'normal',
        '2021': 'abnormal',
        '20192021': 'normal_abnormal'
    }
    scenario = scenario_tag[dataset]
    # Define paths
    ground_truth_paths = {
        'normal': f'./result/normal/ground_truth/{mode}/',
        'abnormal': f'./result/abnormal/ground_truth/{mode}/',
        'normal_abnormal': f'./result/normal_abnormal/ground_truth/{mode}/'
    }
    generated_paths = {
        'normal': f'./result/normal/generated/{mode}/',
        'abnormal': f'./result/abnormal/generated/{mode}/',
        'normal_abnormal': f'./result/normal_abnormal/generated/{mode}/'
    }

    # Choose the last defined path
    ground_truth_path = ground_truth_paths[scenario]
    gen_path = generated_paths[scenario]
    folders = [d for d in os.listdir(ground_truth_path) if os.path.isdir(os.path.join(ground_truth_path, d))]
    # Process ground truth data
    for f in folders:
        person = load_pickle(os.path.join(ground_truth_path, f + '/results.pkl'))
        person_id = f
        test_traj_ids, test_lat_lngs, test_act_ts = obtain_analysis_traj(person)

        truth[person_id] = {
            "test": [test_traj_ids, test_lat_lngs, test_act_ts, person]
        }

        person_to_test.append(person_id)

    # Load generated data
    gen = {}
    for p in person_to_test:
        gen_key = f"{p}_{mode}"
        gen[gen_key] = []
        try:
            result_path = os.path.join(gen_path, p, 'results.pkl')
            res = load_pickle(result_path)
            res_traj_ids, res_traj_lat_lngs, res_traj_acts = obtain_analysis_traj(res)
            gen[gen_key].append([res_traj_ids, res_traj_lat_lngs, res_traj_acts])
        except FileNotFoundError:
            pass

    # Prepare data for evaluation
    gen_data, real_data = {}, {}
    for p in person_to_test:
        gen_key = f"{p}_{mode}"
        for i in range(len(gen[gen_key])):
            if mode not in gen_data:
                gen_data[mode] = transfer(gen[gen_key][i])
                real_data[mode] = transfer(truth[p]["test"])
            else:
                gen_data[mode].extend(transfer(gen[gen_key][i]))
                real_data[mode].extend(transfer(truth[p]["test"]))

    # Initialize evaluation results
    evaluation = Evaluation(None)
    duration_jsd_dict, st_act_jsd_dict = {}, {}
    distance_step_dict, st_loc_jsd_dict = {}, {}

    # Compute evaluation metrics
    duration_jsd, distance_step, st_act_jsd, st_loc_jsd = evaluation.get_JSD(real_data[mode], gen_data[mode])
    duration_jsd_dict[mode] = duration_jsd
    st_act_jsd_dict[mode] = st_act_jsd
    distance_step_dict[mode] = distance_step
    st_loc_jsd_dict[mode] = st_loc_jsd

    print(f"{scenario}")
    # Print evaluation results
    print(
        f"{mode}: "
        f"SD: {np.mean(distance_step_dict[mode]):.4f}, "
        f"SI: {np.mean(duration_jsd_dict[mode]):.4f}, "
        f"DARD: {np.mean(st_act_jsd_dict[mode]):.4f}, "
        f"STVD: {np.mean(st_loc_jsd_dict[mode]):.4f}"
    )
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Define arguments before parsing
    parser.add_argument('--dataset', type=str, default='2019',
                        help='Specify the dataset: ')
    parser.add_argument('--mode', type=int, default=0,
                        help='Specify the mode type: 0 for llm_l, 1 for llm_e')

    args = parser.parse_args()  # Parse after defining arguments

    # Call the eval function with parsed arguments
    eval(dataset=args.dataset, mode=args.mode)
