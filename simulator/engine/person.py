"""
Person class to store necessary information about a person in the simulation.
"""
from simulator.engine.functions.traj_infer import *
from simulator.engine.functions.PISC import *
from simulator.engine.memory.retrieval_helper import *
from simulator.engine.openai_api import OpenAIGPTAPI as LLM
from miscs.utils import *
from simulator.engine.evaluation.metrics import *

coordinate_tokyo = (35.689487, 139.691711)


class Person:
    def __init__(self, name, model, city_area, loc_map, history_routine, activity_cat, top_k_routine=6,
                 pos_map=None, map_pos=None, prob_pos=None, cat=None, id_=-1):
        """
                Initializes a new Person instance.

                Args:
                    name (str): Name of the person.
                    model: The language model used to generate the impression.
                    city_area: The area of the city.
                    loc_map: A map of location identifiers to their descriptions.
                    history_routine: The historical routine of the person， i.e., personal activity trajectory database.
                    activity_cat: Categories of activities involved.
                    top_k_routine (int, optional): The number of top routines to consider. Defaults to 6.
                    pos_map (optional): A mapping of positions. Defaults to None.
                    map_pos (optional): Inverse of pos_map. Defaults to None.
                    prob_pos (optional): Probabilities associated with positions. Defaults to None.
                    cat (optional): Category of the person. Defaults to None.
                    id_ (int, optional): Identifier for the person. Defaults to -1.
                """
        self.train_routine_list = None  # list of training routines
        self.test_routine_list = None  # list of testing routines
        self.name = name
        self.history_routine = history_routine
        self.real_routine = history_routine
        self.city_area = city_area
        self.llm = model
        self.top_k_routine = top_k_routine
        self.iter = 0
        self.loc_map = loc_map
        self.map_loc = {v: k for k, v in self.loc_map.items()}
        self.loc_id_map = {v: id_ for id_, v in enumerate(self.loc_map.values())}
        self.pos_map = pos_map
        self.map_pos = map_pos
        self.cat = cat
        self.id = id_

        self.global_prob_pos = prob_pos
        self.activity_cat = activity_cat
        self.init()
        self.attribute = None
        print("Person {} is created".format(self.name))

    def init(self):
        self.iter = 0
        self.routine_lat_log = extract_lat_log_all(self.history_routine)  # (lat, log) pair of poi in the routine
        self.activity_area = get_location_string(self.loc_map, self.history_routine).split(",")  # location string

        self.full_area = [item.lstrip() for item in self.activity_area]

        self.activity_area, self.area_freq = top_k_occurrences(self.full_area, 30)
        self.loc_cat = {}
        for loc_, freq in self.area_freq.items():
            for k, v in self.cat.items():
                try:
                    if loc_.split("#")[0] in k:
                        self.loc_cat[loc_.split("#")[0]] = v
                except:
                    continue
        self.set_up_mobility_probs()
        self.set_train_test()

        print("Person {} is initialized".format(self.name))

    def set_up_mobility_probs(self):
        person_pos_map = {}
        n = 0
        self.full_area = [item.replace(".", "") for item in self.full_area]
        for loc in list(set(self.full_area)):
            try:
                loc_ = self.map_loc[loc.replace(".", "")]
            except:
                continue
            assert loc_ not in person_pos_map, "duplicate location"
            person_pos_map[loc_] = n
            n += 1

        self.pos_map = person_pos_map
        self.map_pos = {v: k for k, v in self.pos_map.items()}

        routine_ = self.history_routine.split("\n")
        weekday_rank_pos = np.zeros((len(self.pos_map), len(self.pos_map) + 1))
        weekend_rank_pos = np.zeros((len(self.pos_map), len(self.pos_map) + 1))
        weekday_begin_time = np.zeros(6 * 24)  # record the counts of time slot for the first trip every day
        weekend_begin_time = np.zeros(6 * 24)
        weekday_end_time = np.zeros(6 * 24)  # record the counts of time slot for the last trip every day
        weekend_end_time = np.zeros(6 * 24)

        weekday_st = np.zeros((6 * 24, len(self.pos_map)))
        weekend_st = np.zeros((6 * 24, len(self.pos_map)))
        for i in range(len(routine_)):
            if ":" not in routine_[i]:
                continue
            date_ = routine_[i].split(": ")[0].split(" ")[-1]
            loc_list = ", " + routine_[i].split(": ")[-1]
            pattern = r"(?<=\,\s)(.*?)(?=\)\sat)"
            # Extracting matches
            matches = re.findall(pattern, loc_list)
            loc_list = [item + ")" for item in matches]
            if len(loc_list) - 1 < 1:
                continue
            time_list = re.findall(r"\d{2}:\d{2}:\d{2}", routine_[i])
            time_list = calculate_intervals_to_midnight(time_list)
            time_list = [int(item) for item in time_list]
            if is_weekday_or_weekend(date_) == 0:
                weekday_begin_time[time_list[0]] += 1
                weekday_end_time[time_list[-1]] += 1
            else:
                weekend_begin_time[time_list[0]] += 1
                weekend_end_time[time_list[-1]] += 1
            for j in range(len(loc_list) - 1):
                pre_loc = self.pos_map[loc_list[j].split(" at ")[0]]
                next_loc = self.pos_map[loc_list[j + 1].split(" at ")[0]]
                # track_data.append(pos_map[loc_list[j + 1].split(" at ")[0]])
                if is_weekday_or_weekend(date_) == 0:
                    weekday_st[time_list[j], pre_loc] += 1
                    weekday_rank_pos[pre_loc, next_loc] += 1
                else:
                    weekend_st[time_list[j], pre_loc] += 1
                    weekend_rank_pos[pre_loc, next_loc] += 1
            pre_loc = self.pos_map[loc_list[- 1].split(" at ")[0]]
            if is_weekday_or_weekend(date_) == 0:
                weekday_rank_pos[pre_loc, -1] += 1
                weekday_st[time_list[-1], pre_loc] += 1
            else:
                weekend_rank_pos[pre_loc, -1] += 1
                weekend_st[time_list[-1], pre_loc] += 1
            for tt in range(144):
                assert np.sum(weekend_st[tt, :]) >= weekend_begin_time[tt], \
                    f"{np.sum(weekend_st[tt, :])} {weekend_begin_time[tt]}"
        self.begin_time_counts = [weekday_begin_time.reshape(1, -1),
                                  weekend_begin_time.reshape(1, -1)]  # prob of begin time
        self.st_counts = [weekday_st, weekend_st]  # prob of location at each time slot
        self.next_loc_counts = [weekday_rank_pos, weekend_rank_pos]  # prob of next location
        self.end_time_counts = [weekday_end_time.reshape(1, -1), weekend_end_time.reshape(1, -1)]

    def get_history_routine(self):
        return self.history_routine

    def set_train_test(self, representative_routine=None, prior=False):
        if representative_routine is None and prior is False:
            history_routine_list = self.history_routine.split("\n")
            clean_history_routine_list = [item for item in history_routine_list if ":" in item]
            self.clean_history_routine_list_short = []
            for route in clean_history_routine_list:
                s_route = shorten_representative_routine_string(route, self.loc_map)
                self.clean_history_routine_list_short.append(s_route)

            self.first_split = int(len(self.clean_history_routine_list_short) * 0.8)
            self.train_routine_list = self.clean_history_routine_list_short[: self.first_split]
            self.test_routine_list = self.clean_history_routine_list_short[self.first_split:]
            self.raw_train_routine_list = self.history_routine.split("\n")[: self.first_split]
            self.raw_test_routine_list = self.history_routine.split("\n")[self.first_split:]

            if False:
                self.train_routine_list = []
                self.test_routine_list = []
                for i in range(len(self.clean_history_routine_list_short)):
                    if "2019" in self.clean_history_routine_list_short[i]:
                        self.train_routine_list.append(self.clean_history_routine_list_short[i])
                    else:
                        self.test_routine_list.append(self.clean_history_routine_list_short[i])

                self.raw_train_routine_list = []
                self.raw_test_routine_list = []
                for i in range(len(self.history_routine.split("\n"))):
                    if "2019" in self.history_routine.split("\n")[i]:
                        self.raw_train_routine_list.append(self.history_routine.split("\n")[i])
                    else:
                        self.raw_test_routine_list.append(self.history_routine.split("\n")[i])


        else:
            assert representative_routine is not None, "representative_routine is None"

    def check_valid(self, ):
        assert len(self.train_routine_list) > 20, "not enough data to train"
        assert len(self.test_routine_list) > 5, f"not enough data to test"

    def init_retriever(self, ):
        self.retriever = TemporalRetriever(self.raw_train_routine_list,
                                           self.top_k_routine,
                                           is_train=1, class_id_map=self.loc_cat)

    def set_active_area(self, active_area=None):
        """
        iter = 0: set the batch of active area based on the base area and the city area
        """
        if self.iter == 0:
            self.active_area = [list(set(self.u_routine_net_lat_log))]
            self.iter += 1
        else:
            assert active_area is not None, "active area is None"


def setup(tag="30min", city=""):
    if os.name == 'nt':
        root_directory = "../../"
    else:
        root_directory = "/home/jiaweiwang/llm/"
    with open(root_directory + f"database/{city}_routine_sentence_{tag}.pkl", 'rb') as file:
        routine_sentence = pickle.load(file)
    with open(root_directory + "database/activity_cat.pkl", 'rb') as file:
        activity_cat = pickle.load(file)
    with open(root_directory + 'database/catto.pkl', 'rb') as f:
        cat = pickle.load(f)
    with open(root_directory + f"database/{city}_network_{tag}.txt", 'r') as file:
        city_net = file.read()
    with open(root_directory + "database/names.txt", 'r') as file:
        names = file.read()
    names = names.split("\n")
    random.shuffle(names)
    peoples = list(routine_sentence.keys())

    # adjust some ambiguous location names
    city_net = city_net.replace("&", "and")
    routine_sentence = {key: value.replace('&', 'and') for key, value in routine_sentence.items()}

    city_net = city_net.replace("Other Great Outdoors", "Outdoors")
    routine_sentence = {key: value.replace('Other Great Outdoors', 'Outdoors') for key, value in
                        routine_sentence.items()}

    city_net = city_net.replace("general travel", "Outdoors")
    routine_sentence = {key: value.replace('general travel', 'Outdoors') for key, value in
                        routine_sentence.items()}

    city_net = city_net.replace("Harbor / Marina", "Harbor")
    routine_sentence = {key: value.replace('Harbor / Marina', 'Harbor') for key, value in
                        routine_sentence.items()}

    city_net = city_net.replace("Sauna / Steam Room", "Steam Room")
    routine_sentence = {key: value.replace('Sauna / Steam Room', 'Steam Room') for key, value in
                        routine_sentence.items()}

    city_net = city_net.replace("Furniture / Home Store", "Furniture Store")
    routine_sentence = {key: value.replace('Furniture / Home Store', 'Furniture Store') for key, value in
                        routine_sentence.items()}

    city_net = city_net.replace("Gym / Fitness Center", "Gym")
    routine_sentence = {key: value.replace('Gym / Fitness Center', 'Gym') for key, value in
                        routine_sentence.items()}

    city_net = city_net.replace("Bed and Breakfast", "small lodging establishment")
    routine_sentence = {key: value.replace('Bed and Breakfast', 'small lodging establishment') for key, value in
                        routine_sentence.items()}

    city_net = city_net.replace("Ramen / Noodle House", "Noodle Restaurant")
    routine_sentence = {key: value.replace('Ramen / Noodle House', 'Noodle Restaurant') for key, value in
                        routine_sentence.items()}

    city_net = city_net.replace("Dentist's Office", "Dentist")
    routine_sentence = {key: value.replace("Dentist's Office", 'Dentist') for key, value in
                        routine_sentence.items()}

    city_net = city_net.replace("Monument / Landmark", "Landmark")
    routine_sentence = {key: value.replace("Monument / Landmark", 'Landmark') for key, value in
                        routine_sentence.items()}

    city_net = city_net.replace("Doctor's Office", "Medical Center")
    routine_sentence = {key: value.replace("Doctor's Office", 'Medical Center') for key, value in
                        routine_sentence.items()}
    city_net = city_net.replace("Men's Store", "Men Store")
    routine_sentence = {key: value.replace("Men's Store", 'Men Store') for key, value in routine_sentence.items()}

    city_net = city_net.replace("Women's Store", "Men Store")
    routine_sentence = {key: value.replace("Women's Store", 'Men Store') for key, value in routine_sentence.items()}

    city_net = city_net.replace("Home (private)", "Home")
    routine_sentence = {key: value.replace("Home (private)", 'Home') for key, value in
                        routine_sentence.items()}

    city_net = city_net.replace("Shabu-Shabu Restaurant", "Shabu Shabu Restaurant")
    routine_sentence = {key: value.replace("Shabu-Shabu Restaurant", 'Shabu Shabu Restaurant') for key, value in
                        routine_sentence.items()}

    city_net = city_net.replace("Drugstore / Pharmacy", "Pharmacy")
    routine_sentence = {key: value.replace('Drugstore / Pharmacy', 'Pharmacy') for key, value in
                        routine_sentence.items()}
    city_net = city_net.replace("Deli / Bodega", "Indian Restaurant")
    routine_sentence = {key: value.replace('Deli / Bodega', 'Indian Restaurant') for key, value in
                        routine_sentence.items()}

    city_net = city_net.replace("Salon / Barbershop", "Barbershop")
    routine_sentence = {key: value.replace('Salon / Barbershop', 'Barbershop') for key, value in
                        routine_sentence.items()}

    city_net = city_net.replace("Thrift / Vintage Store", "Vintage Store")
    routine_sentence = {key: value.replace('Thrift / Vintage Store', 'Vintage Store') for key, value in
                        routine_sentence.items()}

    city_net = city_net.replace("Gas Station / Garage", "Garage")
    routine_sentence = {key: value.replace('Gas Station / Garage', 'Garage') for key, value in
                        routine_sentence.items()}

    city_net = city_net.replace("Spa / Massage", "Spa")
    routine_sentence = {key: value.replace('Spa / Massage', 'Spa') for key, value in
                        routine_sentence.items()}

    city_net = city_net.replace("Toy / Game Store", "Game Store")
    routine_sentence = {key: value.replace('Toy / Game Store', 'Game Store') for key, value in
                        routine_sentence.items()}

    city_net = city_net.replace("Non-Profit", "Non Profit")
    routine_sentence = {key: value.replace('Non-Profit', 'Non Profit') for key, value in
                        routine_sentence.items()}

    city_net = city_net.replace("Duty-free Shop", "Duty free Shop")
    routine_sentence = {key: value.replace('Duty-free Shop', 'Duty free Shop') for key, value in
                        routine_sentence.items()}

    city_net = city_net.replace("Pop-Up Shop", "Fashion Shop")
    routine_sentence = {key: value.replace('Pop-Up Shop', 'Fashion Shop') for key, value in
                        routine_sentence.items()}

    city_net = city_net.replace("Pop-Up Shop", "Fashion Shop")
    routine_sentence = {key: value.replace('Pop-Up Shop', 'Fashion Shop') for key, value in
                        routine_sentence.items()}

    city_net = city_net.replace("Paper / Office Supplies Store", "Office Supplies Store")
    routine_sentence = {key: value.replace('Paper / Office Supplies Store', 'Office Supplies Store') for key, value in
                        routine_sentence.items()}

    pattern = r'Intersection \(\d{2}\.\d{3}, \d{3}\.\d{3}\) at \d{1,2}:\d{2}:\d{2}, '
    routine_sentence = {key: re.sub(pattern, '', value) for key, value in
                        routine_sentence.items()}
    pattern = r', Intersection \(\d{2}\.\d{3}, \d{3}\.\d{3}\) at \d{1,2}:\d{2}:\d{2}.'
    routine_sentence = {key: re.sub(pattern, '.', value) for key, value in
                        routine_sentence.items()}
    pattern = r': Intersection \(\d{2}\.\d{3}, \d{3}\.\d{3}\) at \d{1,2}:\d{2}:\d{2}'
    routine_sentence = {key: re.sub(pattern, ': ', value) for key, value in
                        routine_sentence.items()}
    city_net = "\n".join([line for line in city_net.strip().split('\n') if 'intersection' not in line.lower()])

    loc_map = {}
    dup_loc = {}

    for loc in city_net.split("\n"):
        if "Location" not in loc:
            print(loc)
            continue
        slim_loc = loc.split(": ")[1].split(" (")[0]

        if slim_loc not in dup_loc:
            dup_loc[slim_loc] = 1
        else:
            dup_loc[slim_loc] += 1
        loc_map[loc.split(": ")[1]] = loc.split(": ")[1].split(" (")[0] + "#" + str(
            dup_loc[loc.split(": ")[1].split(" (")[0]])

    clean_files_in_folder(root_directory + "logs/", type='.txt')
    clean_files_in_folder(root_directory + "logs/", type='.csv')

    # initialize uniform position list
    pos = 0
    pos_map = {}
    map_pos = {}
    for loc in list(loc_map.keys()):
        pos_map[loc] = pos
        map_pos[pos] = loc
        pos += 1

    prob_pos = None
    track_datas = None
    return (routine_sentence, names, peoples, city_net, loc_map, activity_cat, root_directory, prob_pos,
            pos_map, map_pos, track_datas, cat)


def preprocess_pos(pos_map, routine_sentence):
    rank_pos = np.zeros((len(pos_map), len(pos_map)))
    track_data_dict = []

    if type(routine_sentence) == dict:
        for person_name, routine in routine_sentence.items():
            track_datas = []
            routine_ = routine.split("\n")
            for i in range(len(routine_)):
                if ":" not in routine_[i]:
                    continue
                track_data = []
                loc_list = ", " + routine_[i].split(": ")[-1]
                pattern = r"(?<=\,\s)(.*?)(?=\)\sat)"
                matches = re.findall(pattern, loc_list)
                loc_list = [item + ")" for item in matches]
                track_data.append(pos_map[loc_list[0].split(" at ")[0]])
                for j in range(len(loc_list) - 1):
                    pre_loc = pos_map[loc_list[j].split(" at ")[0]]
                    next_loc = pos_map[loc_list[j + 1].split(" at ")[0]]
                    track_data.append(pos_map[loc_list[j + 1].split(" at ")[0]])
                    rank_pos[pre_loc, next_loc] += 1
                if '@' in str(person_name):
                    track_datas.append(track_data)
            if '@' in str(person_name):
                track_data_dict.append(track_datas)
        prob_pos = softmax(rank_pos)
        return prob_pos, track_data_dict


if __name__ == "__main__":
    random.seed(0)
    routine_sentence, names, peoples, city_net, loc_map, activity_cat, root_directory, prob_pos, pos_map, map_pos, track_datas, cat = setup(
        tag="10min_slim", city="Tokyo")

    history_routine_counts = []
    available1921 = [1004, 1032, 1172, 1184, 13, 1310, 1431, 1481, 1492, 1556, 1568, 1626, 1775, 1784, 1874, 1883,
                     1974, 2078, 225, 2266, 2337, 2356, 2402, 2513, 2542, 2610, 2680, 2683, 2721, 2956, 317, 323, 3255,
                     3282, 3453, 3534, 3599, 3637, 3638, 3781, 3784, 4007, 4105, 439, 4396, 4768, 5252, 5326, 540,
                     5449, 5551, 573, 5765, 606, 6144, 6157, 6249, 638, 6581, 6615, 6670, 6814, 6863, 6973, 6998, 7228,
                     7259, 835, 934]
    available2019 = [2575, 1481, 1784, 2721, 638,  7626, 1626, 7266, 1568, 2078, 2610, 1908, 2683, 1883, 3637, 225,  914,
                     6863, 6670, 323,  3282, 2390, 2337, 4396, 7259, 1310, 5849, 3802, 1522, 1219, 1004, 4105, 540,  827,
                     6157, 1556, 2266, 13,   1874, 317,  2513, 3255, 934,  3599, 1775, 606,  3033, 3784, 5252, 3365, 6581,
                     6171, 5326, 2831, 3453, 3781, 2402, 4843, 439,  1172, 3501, 1032, 2542, 1184, 1531, 6615, 7228, 1492,
                     4987, 6204, 6693, 6973, 4057, 67,   2680, 2956, 3138, 3638, 5765, 835,  1431, 6249, 6998, 573,  884,
                     2356, 6463, 930,  3534, 6814, 5551, 5449, 6144, 6156, 4768, 2620, 4007, 1974]
    available2021 = [1481, 1784, 2721, 638, 7626, 13, 47, 107, 225, 323, 392, 413, 439, 540, 572, 606, 638, 643, 789,
                     1032, 1172, 1345, 1481, 1503, 1556, 1568, 1626, 1745, 1775, 6863, 7015, 7068, 7626, 7936,
                     1784, 1874, 1883, 1920, 2078, 2337, 2482, 2513, 2610, 2650, 2721, 2956, 3282, 3494, 3599, 3638,
                     3656, 4105, 4396, 4768, 4947, 5106, 5252, 5326, 6027, 6144, 6204, 6581, 6697, 7982]
    for k in available2019:
        print("Person: " + str(k), peoples[k])
        history_routine = routine_sentence[peoples[k]]
        history_routine_ = history_routine.split("\n")
        history_routine_long = []

        for i in range(len(history_routine_)):
            # select experiment data
            if "2020" in history_routine_[i] or "2021" in history_routine_[i] or "2022" in history_routine_[i]:
                continue
            tuples = re.findall(r'\(([^)]+)', history_routine_[i])
            tuples = [tuple(map(float, t.split(','))) for t in tuples]
            flag = 0
            for coordinate in tuples:
                # Do not use location data that is too far from Tokyo
                if haversine(coordinate[0], coordinate[1], coordinate_tokyo[0], coordinate_tokyo[1]) > 100.:
                    flag = 1
                    break
            if flag == 1:
                continue
            if len(history_routine_[i].split(": ")[-1].split(", ")) > 10:
                history_routine_long.append(history_routine_[i])
        history_routine_counts.append(len(history_routine_long))

        history_routine = "\n".join(history_routine_long)

        P = Person(name=names[k], model=LLM(), city_area=city_net, loc_map=loc_map,
                   history_routine=history_routine, activity_cat=activity_cat,
                   pos_map=pos_map, map_pos=map_pos, prob_pos=prob_pos, cat=cat, id_=k)

        try:
            P.check_valid()
        except AssertionError as e:
            print(e)
            k += 1
            continue

        neg_routines = []
        for i in range(min(len(P.train_routine_list), 30)):
            numbers = [i for i in range(len(peoples)) if i != k]
            selected_number = random.choice(numbers)
            neg_routines.append(routine_sentence[peoples[selected_number]].split("\n")[0])

        # identify the pattern of the person
        P = pisc(P, neg_routines=neg_routines)
        # initialize the retriever
        P.init_retriever()
        # activity generation
        plan_new_day(P)

    print("done")
