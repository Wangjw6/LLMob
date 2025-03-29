from engine.llm_configs.gpt_structure import *
from engine.utilities.process_tools import *

root_directory = "./engine/"


def identify(person, candidate_num=10):
    neg_routines = person.neg_routines
    i_template = root_directory + "/prompt_template/init.txt"
    role_template = root_directory + "/prompt_template/roles.txt"
    infer_role_template = root_directory + "/prompt_template/init_role.txt"
    e_template1 = root_directory + "/prompt_template/eval.txt"
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
        except Exception as e:
            print("Role extraction error: ", e)
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
                                  objective=f"Describe patterns ...", history=None)
        try:
            print(contents)
            answers = description_first_view + contents
            answers = first2second(answers)
            person.attribute = answers
            att_hub.append(answers)
        except Exception as e:
            print("Attribute extraction error: ", e)
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


def score_from_rating(person, att_hub, e_template, metric, neg_routines=None):
    scores_dict = {}
    r = 0
    for att in att_hub:
        r += 1
        for i in range(min(30, len(person.train_routine_list))):
            train_route = person.train_routine_list[
                i]
            train_route = train_route.split(": ")[-1]
            curr_input = [att, train_route]
            prompt = generate_prompt(curr_input, e_template)
            while True:
                contents = execute_prompt(prompt, person.llm,
                                          objective=f"eval...{r}/{len(att_hub)}")
                try:
                    print(contents)
                    score = int(re.search(r'\d+', contents).group())
                    if att not in scores_dict:
                        scores_dict[att] = [score]
                    else:
                        scores_dict[att].append(score)
                except Exception as e:
                    print("Score extraction error: ", e)
                    continue
                break
        if person.neg_routines is not None:
            for i in range(len(person.neg_routines)):
                train_route = person.neg_routines[i]
                train_route = train_route.split(": ")[-1]
                curr_input = [att, train_route]
                prompt = generate_prompt(curr_input, e_template)
                prompt = prompt
                while True:
                    contents = execute_prompt(prompt, person.llm,
                                              objective=f"eval...{r}/{len(att_hub)}")
                    try:
                        print(contents)
                        score = int(re.search(r'\d+', contents).group())
                        if att not in scores_dict:
                            scores_dict[att] = [-score]
                        else:
                            scores_dict[att].append(-score)
                    except Exception as e:
                        print("Score extraction error: ", e)
                        continue
                    break
    return scores_dict
