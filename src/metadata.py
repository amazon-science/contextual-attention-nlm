# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import logging
import math
import torch
from datetime import datetime

"""
Functions for converting metadata strings into metadata tokens.
"""

logger = logging.getLogger(__name__)

# Datetime metadata processing functions

def split_datetime_md(md_string):
    """Convert md_string into date object"""
    date = datetime.strptime(md_string,"%Y-%m-%d-%H")
    return date


def extract_hour_token(md_string):
    """Returns one token per md_string corresponding into one of 24 hours tokens"""
    date = split_datetime_md(md_string)
    md_token = "<MDHOUR" + str(date.hour) + ">"
    return md_token


def extract_part_of_day_token(md_string):
    """Splits into one of 3 tokens: morning, midday and evening"""
    date = split_datetime_md(md_string)
    hour = date.hour
    md_token = "<MD"
    if hour >= 5 and hour < 10:
        md_token = md_token + "MORNING" + ">"
    elif hour >= 10 and hour < 14:
        md_token = md_token + "MIDDAY" + ">"
    else:
        md_token = md_token + "EVENING" + ">"
    return md_token


def extract_day_of_week_token(md_string):
    """ Splits into one of 7 days """
    date = split_datetime_md(md_string)
    md_token = "<MDDAY" + str(date.weekday()) + ">"
    return md_token


def extract_weekend_weekday_token(md_string):
    """ Splits into one of 2: weekday, and weekend """
    date = split_datetime_md(md_string)
    md_token = "<MD"
    weekno = date.weekday()
    if weekno < 5:
        md_token = md_token + "WEEKDAY" + ">"
    else:
        md_token = md_token + "WEEKEND" + ">"
    return md_token


def extract_week_of_year_token(md_string):
    """ Splits into 1 of 53 weeks numbers in range(1,54) """
    date = split_datetime_md(md_string)
    md_token = "<MDWEEK" + str(date.isocalendar()[1]) + ">"
    return md_token


def extract_month_of_year_token(md_string):
    """ Splits into 1 of 12 months numbers in range(1,13) """
    date = split_datetime_md(md_string)
    md_token = "<MDMONTH" + str(date.month) + ">"
    return md_token


def extract_year_token(md_string):
    """ Returns one of 7 year numbers (2014-2020)"""
    date = split_datetime_md(md_string)
    md_token = "<MDYEAR" + str(date.year) + ">"
    return md_token


def extract_all_tokens(md_string):
    """ Combines month, week, day and hour information """
    date = split_datetime_md(md_string)
    month_token = "<MDMONTH" + str(date.month) + ">"
    week_token = "<MDWEEK" + str(date.isocalendar()[1]) + ">"
    day_token = "<MDDAY" + str(date.weekday()) + ">"
    hour_token = "<MDHOUR" + str(date.hour) + ">"

    md_data = [month_token, week_token, day_token, hour_token]
    md_tokens = " ".join(md_data)
    return md_tokens


def extract_radians(md_string):
    """
    Creates a radian representation of date information that uses: hour, day, week,
    month information encapsulated in one 8-dimensional vector.
    """
    date = split_datetime_md(md_string)

    hour = date.hour # Between 0 and 23
    day = date.weekday() # Between 0 and 6
    week = date.isocalendar()[1] - 1 # Between 0 and 52
    month = date.month - 1 # between 0 and 11

    radians_hour = [math.sin(2*math.pi*hour/24), math.cos(2*math.pi*hour/24)]
    radians_day = [math.sin(2*math.pi*day/7), math.cos(2*math.pi*day/7)]
    radians_week = [math.sin(2*math.pi*week/53), math.cos(2*math.pi*week/53)]
    radians_month = [math.sin(2*math.pi*month/12), math.cos(2*math.pi*month/12)]

    all_radians = radians_hour + radians_day + radians_week + radians_month

    return torch.FloatTensor(all_radians)

# Geo hash metadata processing functions
from string import ascii_lowercase as alphabet
geo_hash_chars = list(alphabet) + list(range(0,10))
geo_hash_set = {f"{x}{y}" for x in geo_hash_chars for y in geo_hash_chars}

def extract_geo_hash(md_string):
    if md_string not in geo_hash_set:
        md_string = "None"
    return f"<MD{md_string}GHASH>"

### Metadata transformation class

TRANSFORM_MAP = {
    "hour_token": (extract_hour_token, set(["<MDHOUR" + str(i) + ">" for i in range(24)])),
    "part_of_day_token": (extract_part_of_day_token, set(["<MDMORNING>", "<MDMIDDAY>", "<MDEVENING>"])),
    "day_of_week_token": (extract_day_of_week_token, set(["<MDDAY" + str(i) + ">" for i in range(7)])),
    "weekend_weekday_token": (extract_weekend_weekday_token, set(["<MDWEEKEND>", "<MDWEEKDAY>"])),
    "week_of_year_token": (extract_week_of_year_token, set(["<MDWEEK" + str(i) + ">" for i in range(1,54)])),
    "month_of_year_token": (extract_month_of_year_token, set(["<MDMONTH" + str(i) + ">" for i in range(1,13)])),
    "year_token": (extract_year_token, set(["<MDYEAR" + str(i) + ">" for i in range(2014,2021)])),
    "all_tokens": (extract_all_tokens, set(["<MDMONTH" + str(i) + ">" for i in range(1,13)] +\
                                           ["<MDWEEK" + str(i) + ">" for i in range(1,54)] +\
                                           ["<MDDAY" + str(i) + ">" for i in range(7)] +\
                                           ["<MDHOUR" + str(i) + ">" for i in range(24)])),
    "radians": (extract_radians, set()),
    "geo_hash": (extract_geo_hash, {f"<MD{md_string}GHASH>" for md_string in geo_hash_set}),
}

class MetaDataTransformer():
    def __init__(self, text_index, md_indices, md_transformations):
        self.text_index = int(text_index)
        self.md_transformations = md_transformations.split(',') if md_transformations else []
        self.md_indices = [int(x) for x in md_indices.split(',') if x]

        assert(len(self.md_indices) == len(self.md_transformations)), \
            "Length of metadata indices and metadata transformations are mismatched"

    def get_md_tokens(self):
        all_tokens = set()
        for md_transform in self.md_transformations:
            _, tokens = TRANSFORM_MAP[md_transform]
            all_tokens.update(tokens)
        return all_tokens

    def parse_raw_input(self, utterance):
        md = {}
        split_utterance = utterance.rstrip().split('\t')
        for md_index, md_transform in zip(self.md_indices, self.md_transformations):
            raw_md = split_utterance[md_index].rstrip()
            md_transform_func, _ = TRANSFORM_MAP[md_transform]
            md[md_transform] = md_transform_func(raw_md)
        text = split_utterance[self.text_index]
        return (md, text)
