"""
Person class to store necessary information about a person in the simulation.
"""

from engine.utilities.retrieval_helper import *


class Person:
    def __init__(self, name, model, person_id=-10):
        self.retriever = None
        self.train_routine_list = None  # list of training routines
        self.test_routine_list = None  # list of testing routines
        self.name = name
        self.city_area = None
        self.llm = model
        self.cat = None
        self.id = person_id
        self.domain_knowledge = None
        self.neg_routines = None
        self.attribute = None
        self.loc_cat = None
        self.top_k_routine = 6
        print("Person {} is created".format(self.name))

    def init_retriever(self, ):
        self.retriever = TemporalRetriever(self.train_routine_list,
                                           6,
                                           is_train=1, class_id_map=self.loc_cat)
