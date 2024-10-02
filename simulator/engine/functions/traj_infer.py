from simulator.engine.functions.prompt_paths import *
from simulator.engine.functions.helper import *
from simulator.gpt_structure import *
from miscs.utils import *
from simulator.engine.memory.retrieval_helper import *
import os

root_directory = "./simulator/"


def plan_new_day(person, sample_num=1, mode=0):
    infer_template = root_directory + "prompt_template/final_version/one-shot_infer_mot.txt"
    # mode = 0 for learning based retrieval, 1 for evolving based retrieval
    describe_mot_template = root_directory + motivation_infer_prompt_paths[mode]
    motivation_ways = ["Following are the motivation that you want to achieve:",
                       "Following are the thing you focus in the last few days:"
                       ]

    folder_path = root_directory + f"/{str(person.id)}/"
    if os.path.exists(folder_path) is False:
        os.makedirs(folder_path)

    for k in range(sample_num):
        results = {}
        prompts = {}
        reals = {}
        motivations = {}
        his_routine = person.train_routine_list[-person.top_k_routine:]
        for test_route in person.test_routine_list:
            date_ = test_route.split(": ")[0].split(" ")[-1]

            def is_date_in_range(date_str, begin, end):
                try:
                    date = datetime.strptime(date_str, "%Y-%m-%d")
                except ValueError:
                    return False
                start_date = datetime(begin[0], begin[1], begin[2])
                end_date = datetime(end[0], end[1], end[2])
                return start_date <= date <= end_date

            # get motivation
            consecutive_past_days = check_consecutive_dates(his_routine, date_)
            if mode == 0:
                # learning based retrieved
                retrieve_route = person.retriever.retrieve(date_)
                print("retrieve_route: ", retrieve_route[0])
                demo = retrieve_route[0]
            else:
                # evolving based retrieved
                demo = his_routine[-1]
            # flag = 0
            # Specify the pandemic period prompt
            # his_date_ = retrieve_route[0].split(": ")[0].split(" ")[-1]
            # if is_date_in_range(his_date_, [2021,1,7], [2021,3,21]) or is_date_in_range(his_date_, [2021,4,25], [2021,6,20]) or is_date_in_range(his_date_, [2021,7,12], [2021,9,30]):
            #     flag = 1
            #     hint="**Now it is the pandemic period in your city. The government is declaring a state of emergency. Think about the current situation.** "#"Now it is the pandemic period and government has asked that residents to postpone travel and events and to telecommute as much as possible. "
            # else:
            #     hint = ""
            hint = ""
            # shorten_representative_routine_string(retrieve_route[0], person.loc_map)
            curr_input = [person.attribute, "Go to " + demo.split(": ")[-1], consecutive_past_days, hint]

            prompt = generate_prompt(curr_input, describe_mot_template)

            try:
                area = retrieve_loc(person, demo)
            except:
                assert False

            motivation = execute_prompt(prompt, person.llm, objective=f"Think about motivation")
            motivation = first2second(motivation)
            motivations[date_] = motivation

            his_routine = his_routine[1:] + [test_route]

            weekday = find_detail_weekday(date_)

            hint = ""
            if motivation is not None:
                curr_input = [person.attribute, motivation, date_, ',  '.join(area), weekday, demo,
                              motivation_ways[mode],
                              hint]

            prompt = generate_prompt(curr_input, infer_template)
            prompts[date_] = prompt
            max_trial = 10
            trial = 0
            while trial < max_trial:
                contents = execute_prompt(prompt, person.llm,
                                          objective=f"one_shot_infer_re_{len(results) + 1}/{len(person.test_routine_list)}")
                try:
                    res = json.loads(contents)
                    valid_generation(person, f"Activities at {date_}: " + ', '.join(res["plan"]))
                except Exception as e:
                    print(e)
                    trial += 1
                    continue
                break
            if trial >= max_trial:
                res = {"plan": demo.split(": ")[-1]}
            print(contents)
            print("Motivation: ", motivation)
            print("Real: ", test_route)

            reals[date_] = test_route  # shorten_representative_routine_string(test_route, person.loc_map)
            results[date_] = f"Activities at {date_}: " + ', '.join(res["plan"])
            person.retriever.nodes.append(reals[date_])

    print(folder_path)
