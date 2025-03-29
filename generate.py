from engine.llm_configs.openai_api import OpenAIGPTAPI as LLM
from engine.trajectory_generate import *
from engine.persona_identify import *
from engine.agent import *
import argparse
import random

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str,
                    default='2019')  # 2019 for data only available in 2019, 2021 for data only available in 2021, 20192021 for data available in both 2019 and 2021
parser.add_argument('--mode', type=int,
                    default=0)  # mode = 0 for learning based retrieval, 1 for evolving based retrieval
parser.add_argument('--seed', type=int, default=123)

if __name__ == "__main__":
    args = parser.parse_args()
    random.seed(args.seed)

    available1921 = [1004, 1032, 1172, 1184, 13, 1310, 1431, 1481, 1492, 1556, 1568, 1626, 1775, 1784, 1874, 1883,
                     1974, 2078, 225, 2266, 2337, 2356, 2402, 2513, 2542, 2610, 2680, 2683, 2721, 2956, 317, 323, 3255,
                     3282, 3453, 3534, 3599, 3637, 3638, 3781, 3784, 4007, 4105, 439, 4396, 4768, 5252, 5326, 540,
                     5449, 5551, 573, 5765, 606, 6144, 6157, 6249, 638, 6581, 6615, 6670, 6814, 6863, 6973, 6998, 7228,
                     7259, 835, 934]
    available2019 = [2575, 1481, 1784, 2721, 638, 7626, 1626, 7266, 1568, 2078, 2610, 1908, 2683, 1883, 3637, 225, 914,
                     6863, 6670, 323, 3282, 2390, 2337, 4396, 7259, 1310, 3802, 1522, 1219, 1004, 4105, 540,
                     6157, 1556, 2266, 13, 1874, 317, 2513, 3255, 934, 3599, 1775, 606, 3033, 3784, 5252, 3365, 6581,
                     6171, 5326, 2831, 3453, 3781, 2402, 4843, 439, 1172, 3501, 1032, 2542, 1184, 1531, 6615, 7228,
                     1492 , 6973, 67, 2680, 2956, 3138, 3638, 5765, 835, 1431, 6249, 6998, 573, 884,
                     2356, 6463, 930, 3534, 6814, 5551, 5449, 6144, 6156, 4768, 2620, 4007, 1974]
    available2021 = [1481, 1784, 2721, 638,  7626, 13,   47,    107, 225,  323,  392,  413,  439,  540,  572, 606, 638, 643, 789,
                     1032, 1172, 1345, 1481, 1503, 1556, 1568, 1626, 1745, 1775, 6863, 7015, 7068, 7626, 7936,
                     1784, 1874, 1883, 1920, 2078, 2337, 2482, 2513, 2610, 2650, 2721, 2956, 3282, 3494, 3599, 3638,
                     3656, 4105, 4396, 4768, 4947, 5106, 5252, 5326, 6027, 6144, 6204, 6581, 6697, 7982]
    folder = f"./data/{args.dataset}/"
    data = {"2019": available2019, "2021": available2021, "20192021": available1921}
    scenario_tag = {
        '2019': 'normal',
        '2021': 'abnormal',
        '20192021': 'normal_abnormal'
    }
    for k in data[args.dataset]:
        with open(folder + str(k) + ".pkl", "rb") as f:
            att = pickle.load(f)
            P = Person(name=k, model=LLM(), person_id=k)
            P.train_routine_list, P.test_routine_list, P.attribute, P.cat, P.domain_knowledge, P.neg_routines, P.activity_area, P.area_freq,  P.loc_cat = \
                att[0], att[1], att[2],  att[4], att[5], att[6], att[7], att[8], att[11]

        # identify the pattern of the person based on self-consistency
        P = identify(P)
        # # initialize the retriever
        if args.mode == 0:
            P.init_retriever()
        # mobility generation
        mob_gen(P, mode=args.mode, scenario_tag=scenario_tag[args.dataset])

    print("done")
