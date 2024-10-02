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
