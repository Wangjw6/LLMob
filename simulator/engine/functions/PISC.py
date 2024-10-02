from simulator.gpt_structure import *
from miscs.utils import *
from simulator.engine.functions.helper import *
import os
from simulator.engine.evaluation.metrics import *

root_directory = "./simulator/"


def pisc(person, candidate_num=10):
    neg_routines = person.neg_routines
    i_template = root_directory + "/prompt_template/final_version/init.txt"
    role_template = root_directory + "/prompt_template/final_version/roles.txt"
    infer_role_template = root_directory + "/prompt_template/final_version/init_role.txt"
    e_template1 = root_directory + "/prompt_template/final_version/eval.txt"
    # extract the basic information of the person from the history routine
    domain_knowledge = person.domain_knowledge  # extract_knowledge(person)
    roles = {}
    demo = ""
    with open(role_template, 'r') as file:
        for line in file:
            if ": " in line:
                key, value = line.split(": ", 1)
                demo += line
                demo += "\n"
                roles[key.strip()] = value.strip()
            if len(roles) == candidate_num:
                break
    # add role from gpt
    curr_input = [domain_knowledge, demo]
    prompt = generate_prompt(curr_input, infer_role_template)
    while True:
        try:
            contents = execute_prompt(prompt, person.llm,
                                      objective=f"init role...", history=None)
            for c in contents.split("\n"):
                if c:
                    role, description = c.split(": ")
                    roles[role] = description
                    break
        except:
            continue
        break

    att_hub = []
    for role, description in roles.items():
        role = role.split("#")[0]
        description_first_view = f"I am a {role} in this urban neighborhood," + description.replace("you ", "I ")
        description_first_view = description_first_view.replace("your ", "my ")
        curr_input = [role, description, domain_knowledge, ', '.join(person.activity_area[5:]), description_first_view]
        prompt = generate_prompt(curr_input, i_template)
        contents = execute_prompt(prompt, person.llm,
                                  objective=f"HTS...", history=None)
        try:
            print(contents)
            answers = description_first_view + contents
            answers = first2second(answers)
            person.attribute = answers
            att_hub.append(answers)

        except:
            continue
    scores_dict = score_from_rating(person, att_hub, e_template1, metric="rate", neg_routines=neg_routines)

    max_score = 0
    final_att = ""
    print(person.id)
    for att, scores in scores_dict.items():
        print("selection score: ", sum(scores))
        print("candidate att: ", att)
        if sum(scores) > max_score:
            max_score = sum(scores)
            final_att = att
    print("final att: ", final_att)
    print("final score: ", max_score)
    person.attribute = final_att
    return person


def score_from_rating(person, att_hub, e_template, metric="binary", neg_routines=None):
    scores_dict = {}
    r = 0
    for att in att_hub:
        r += 1
        for i in range(min(30, len(person.train_routine_list))):
            train_route = person.train_routine_list[
                i]  # shorten_representative_routine_string(person.train_routine_list[i], person.loc_map)
            date_str = train_route.split(": ")[0].split(" ")[-1]
            train_route = train_route.split(": ")[-1]
            curr_input = [att, train_route]
            prompt = generate_prompt(curr_input, e_template)
            history = None
            trial = 0
            while True:
                if history is not None:
                    contents = execute_prompt(history, person.llm,
                                              objective=f"eval...{r}/{len(att_hub)}", history=history)
                else:
                    contents = execute_prompt(prompt, person.llm,
                                              objective=f"eval...{r}/{len(att_hub)}")
                try:
                    print(contents)
                    if metric == "binary":
                        ans = re.search(r'\d+', contents).group()
                        target = is_weekday_or_weekend(date_str)
                        if int(ans) == target:
                            score = 1
                        else:
                            score = 0
                            history = [
                                {"role": "user", "content": prompt},
                                {"role": "assistant", "content": contents},
                                {"role": "user", "content": "You have one more chance to give me answer."}
                            ]
                            if trial == 0 and history is not None:
                                trial += 1
                                continue
                        print("Score: ", score)
                    if metric == "rate":
                        score = int(re.search(r'\d+', contents).group())
                        if trial == 0 and history is not None:
                            trial += 1
                            continue
                    else:
                        assert False
                    if att not in scores_dict:
                        scores_dict[att] = [score]
                    else:
                        scores_dict[att].append(score)
                except:
                    continue
                break
        if person.neg_routines is not None:
            for i in range(len(person.neg_routines)):
                train_route = person.neg_routines[
                    i]  # shorten_representative_routine_string(neg_routines[i], person.loc_map)
                date_str = train_route.split(": ")[0].split(" ")[-1]

                train_route = train_route.split(": ")[-1]
                curr_input = [att, train_route]
                prompt = generate_prompt(curr_input, e_template)
                prompt = prompt
                history = None
                trial = 0
                while True:
                    if history is not None:
                        contents = execute_prompt(history, person.llm,
                                                  objective=f"eval...{r}/{len(att_hub)}", history=history)
                    else:
                        contents = execute_prompt(prompt, person.llm,
                                                  objective=f"eval...{r}/{len(att_hub)}")
                    try:
                        print(contents)
                        if metric == "binary":
                            ans = re.search(r'\d+', contents).group()
                            target = is_weekday_or_weekend(date_str)
                            if int(ans) == target:
                                score = 1
                            else:
                                score = 0
                                history = [
                                    {"role": "user", "content": prompt},
                                    {"role": "assistant", "content": contents},
                                    {"role": "user", "content": "You have one more chance to give me answer."}
                                ]
                                if trial == 0 and history is not None:
                                    trial += 1
                                    continue
                            print("Score: ", score)
                        if metric == "rate":
                            score = int(re.search(r'\d+', contents).group())
                            if trial == 0 and history is not None:
                                trial += 1
                                continue
                        else:
                            assert False
                        if att not in scores_dict:
                            scores_dict[att] = [-score]
                        else:
                            scores_dict[att].append(-score)

                    except:
                        continue
                    break
    return scores_dict
