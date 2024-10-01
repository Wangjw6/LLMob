"""
Person class to store necessary information about a person in the simulation.
"""

from simulator.engine.memory.retrieval_helper import *


class Person:
    def __init__(self, name, model, id_=-1):

        self.train_routine_list = None  # list of training routines
        self.test_routine_list = None  # list of testing routines
        self.name = name
        self.city_area = None
        self.llm = model
        self.iter = 0
        self.loc_map = None
        self.map_loc = None
        self.loc_id_map = None
        self.pos_map = None
        self.map_pos = None
        self.cat = None
        self.id = id_
        self.global_prob_pos = None
        self.activity_cat = None
        self.domain_knowledge = None
        self.neg_routines = None
        self.attribute = None
        self.top_k_routine = 6
        print("Person {} is created".format(self.name))


    def init_retriever(self, ):
        self.retriever = TemporalRetriever(self.train_routine_list,
                                           6,
                                           is_train=1, class_id_map=self.loc_cat)





