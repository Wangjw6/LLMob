import numpy as np
from sentence_transformers import SentenceTransformer, util
from datetime import datetime
# import geobleu
import re
def DTWDistance(s, t):
    n, m = len(s), len(t)
    DTW = np.full((n + 1, m + 1), np.inf)
    DTW[0, 0] = 0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = d(s[i - 1], t[j - 1])  # Assuming a distance function 'd' is defined
            DTW[i, j] = cost + min(DTW[i - 1, j],  # insertion
                                   DTW[i, j - 1],  # deletion
                                   DTW[i - 1, j - 1])  # match
            #print(i - 1, j - 1, cost)

    return DTW[n, m]


# Example distance function (Euclidean distance)
def d(point1, point2):
    return np.sqrt(np.sum((np.array(point1) - np.array(point2)) ** 2))


def get_route_numeric(route, map_latlog):
    numeric_record = []
    for a in route:
        location_str = a.split(" at ")[0].lstrip()
        time_str = a.split(" at ")[1]
        # time = datetime.strptime(time_str, "%H:%M:%S").time()
        match = re.search(r'\(([^,]+),\s*([^)]+)\)', map_latlog[location_str])
        if match:
            # Extracting latitude and longitude
            lat, lng = match.groups()
            lat = float(lat)
            lng = float(lng)
        else:
            raise ValueError(f"Location string '{location_str}' not in expected format.")
        numeric_record.append((lat, lng))
    return numeric_record


# def geo_bleu_score(generated, reference):
#     geobleu_val = geobleu.calc_geobleu(generated, reference, processes=3)
#     return geobleu_val

def jaccard_similarity(list1, list2):
    """
    Jaccard Similarity J(A,B) = |A∩B| / |A∪B| = |A∩B| / (|A|+|B|-|A∩B|)
    :param list1: activity set 1
    :param list2: activity set 2
    :return: Jacard similarity (gauging the similarity and diversity between two sets of activities)
    """
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection

    return float(intersection) / union


def semantic_similarity_bert(sentence1, sentence2):
    try:
        model = SentenceTransformer('E:/llm/data/sentence-transformers_paraphrase-distilroberta-base-v1')
    except:
        model = SentenceTransformer('/home/jiaweiwang/llm/data/sentence-transformers_paraphrase-distilroberta-base-v1')
    # encode sentences to get their embeddings
    embeddings1 = model.encode(sentence1, convert_to_tensor=True)
    embeddings2 = model.encode(sentence2, convert_to_tensor=True)
    # compute similarity scores of two embeddings
    cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)
    return cosine_scores

def categorize_place(name):
    # Convert the name to lowercase for case-insensitive matching
    lower_name = name.lower()

    # Define categories and corresponding keywords
    categories = {
        'dining': ['pizza', 'fried','breakfast','bbq','wings','taco','deli','noodle', 'restaurant', 'diner', 'eatery', 'burger', 'cafe', 'cheese', 'hot dog', 'tea', 'food', 'salad', 'café', 'soup', 'sandwich', 'BBQ', 'burrito','bakery'],
        'outdoors': ['park', 'garden', 'green', 'river', 'sea', 'beach', 'outdoors', 'neighbor', 'campground'],
        'sightseeing': ['church', 'travel','synagogue','temple', 'museum', 'gallery','monument', 'landmark', 'historic', 'castle', 'shrine', 'zoo', 'aquarium', 'art','scenic', 'mosque', 'arcade', 'planetarium'],
        'shopping': ['mall', 'shop', 'market', 'store','rental','snack'],
        'entertaining': ['entertainment', 'concert','brewery','bar', 'game', 'movie', 'nightlife', 'theater', 'cinema', 'casino','comedy',  'distillery', 'winery', 'spa', 'rest', 'music'],
        'sports': ['sport', 'bowling','gym','playground','basketball', 'ski', 'pool', 'stadium'],
        'work-related': ['office', 'government', 'emabssy','factory','company','building','bank'],
        'study-related': ['school', 'library', 'college', 'student', 'university'],
        'personal': ['event', 'service', 'pet', 'housing', 'facility', 'medical', 'residential', 'shelter', 'wash', 'home', 'cemetery','salon'],
        'secret': ['military ']
        # Add more categories and keywords as needed
    }

    # Check for matches in each category
    for category, keywords in categories.items():
        if any(keyword in lower_name for keyword in keywords):
            return category

    # Return a default category if no match is found
    return 'Other'

interval = 60 #min
def map_traj2mat(traj, class_loc_map, act_map):
    mat = np.zeros((int(1440 / interval), len(act_map)))
    traj = traj.replace(":00, ", " at ")
    traj_ = traj.split(" at ")
    i = 0
    while i < len(traj_)-1:
        loc = traj_[i].split("#")[0]
        time_str = traj_[i+1]
        act_class = class_loc_map[loc]


        time_obj = None
        try:
            if '.' not in time_str:
                time_obj = datetime.strptime(time_str, '%H:%M')
            else:
                time_obj = datetime.strptime(time_str, '%H:%M:%S.')
        except:
            print('map_traj2mat', traj)
            pass
        if time_obj is None:
            continue
        minutes = time_obj.hour * 60 + time_obj.minute
        mat[int(minutes / interval), act_map[act_class]] += 1
        i+=2
    return mat

def activitieslist(traj):
    act_list = []
    for act in traj.split(", "):
        if 'Activities' in act:
            loc = act.split(" at ")[1].split("#")[0]
        else:
            loc = act.split(" at ")[0].split("#")[0]
        act_class = categorize_place(loc)
        act_list.append(act_class)
    return act_list
def act_mat_compute(traj1, traj2, class_loc_map):
    # pattern = r'\(([\d.]+),\s*([\d.]+)\)'
    #
    # # Find all matches
    # lat_long_pairs1 = re.findall(pattern, traj1)
    # lat_long_pairs1 = [(float(lat), float(lon)) for lat, lon in lat_long_pairs1]
    # # print(lat_long_pairs1)
    # # print(traj1)
    # lat_long_pairs2 = re.findall(pattern, traj2)
    # lat_long_pairs2 = [(float(lat), float(lon)) for lat, lon in lat_long_pairs2]
    #
    # comp = DTWDistance(lat_long_pairs1, lat_long_pairs2)
    # return -comp
    # comp = geobleu.calc_geobleu(traj1, traj2, processes=3)
    # traj1a = activitieslist(traj1)
    # traj2a = activitieslist(traj2)
    # comp = jaccard_similarity(traj1a, traj2a)
    # traj1 = "Activities at 2013-02-14: Convenience Store#201 at 14:20:00."
    # traj2 = "Activities at 2013-02-14: Miscellaneous Shop#85 at 14:00:00."
    act_map = {v: id_ for id_, v in enumerate(list(set(class_loc_map.values())))}
    mat1 = map_traj2mat(traj1, class_loc_map, act_map)
    mat2 = map_traj2mat(traj2, class_loc_map, act_map)
    # print(np.where((mat1 == mat2) & (mat1 >= 1)))
    # comp = np.sum((mat1 - mat2) ** 2)
    comp = np.where((mat1 == mat2) & (mat1 >= 1))[0].shape[0] / max(np.where((mat1 >= 1))[0].shape[0], np.where((mat2 >= 1))[0].shape[0])
    # print("act_mat_compute ")
    # print(traj1)
    # print(traj2)
    # print(comp)

    return comp


def aggregate_traj(trajs):
    demand = {}
    for traj in trajs:
        traj = traj.split(": ")[1]
        for act in traj.split(", "):
            loc = act.split(" at ")[0]
            time_str = act.split(" at ")[-1]
            if '.' not in time_str:
                try:
                    time_obj = datetime.strptime(time_str, '%H:%M:%S')
                except:
                    print('aggregate_traj1', traj)
                    time_obj = datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
            else:
                try:
                    time_obj = datetime.strptime(time_str, '%H:%M:%S.')
                except:
                    print('aggregate_traj2', traj)
                    time_obj = datetime.strptime(time_str, '%H:%M:%S.%f')

            minutes = time_obj.hour * 60 + time_obj.minute
            if loc not in demand:
                demand[loc] = np.zeros(144)
                demand[loc][int(minutes % 144)] += 1
            else:
                demand[loc][int(minutes % 144)] += 1

    return demand
