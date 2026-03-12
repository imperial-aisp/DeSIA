"""
Utility functions to process data used by multiple methods
Partially modified from the source code under https://github.com/terranceliu/dp-query-release/blob/main/src/utils
"""
import torch
import itertools
import numpy as np
import pandas as pd
from typing import *
from dataclasses import dataclass
from collections import defaultdict

def hamming_distance(record1, record2):
    return sum(x != y for x, y in zip(record1, record2))

def _find_potential_records(_input_args):
    (query, _), domain_config = _input_args
    
    query_dict = {col: values for col, values in query}
    ext_query_dict = defaultdict(list)
    for col in domain_config:
        # If the attribute is inside one condition clause of the query, select values in that condition clause
        if col in query_dict:
            ext_query_dict[col] = list(query_dict[col])
        # Else, select all possible values for that attribute
        else:
            ext_query_dict[col] = list(range(domain_config[col]))
    _record_range = ext_query_dict.values()

    potential_records = []
    for _record in itertools.product(*_record_range):
        _ext_potential_record = {col: val for col, val in zip(domain_config, _record)}
        _potential_record = {}
        for col in domain_config:
            if col != "cenrace":
                _potential_record.update({col: _ext_potential_record[col]})
            else:
                detail_race = id_to_race[_ext_potential_record["cenrace"]]
                _potential_record.update({majority_race: int(detail_race.__dict__[majority_race]) for majority_race in detail_race.majority_races})
        potential_records.append(tuple(_potential_record.values()))
    return query, potential_records

# The function to generate constraints encoded by each query in parallel
def _generate_constraints(_input_args):
    (query, count), records, row_indent, non_zero_potential_record_indexs = _input_args

    constraint_variables = ["V[{}]".format(non_zero_potential_record_indexs[record]) for record in records if record in non_zero_potential_record_indexs]
    if len(constraint_variables) > 0:
        # There are some non-zero potential records bounded by this constraint
        _tmp_script = "\n{}constraint_variables = [{}]".format(row_indent, ",".join(constraint_variables))
        _tmp_script += "\n{}C[{}] = model.addConstr(quicksum(constraint_variables) == {})".format(row_indent, query, count)
    else:
        # All non-zero potential records are not bounded by this constraint
        _tmp_script = ""
    return _tmp_script

def _generate_encoding_matrix(_input_args):
    (query, _potential_records_under_query), non_zero_potential_records = _input_args
    return query, [1. if record in _potential_records_under_query else 0. for record in non_zero_potential_records]

def _find_record_indexs(_input_args):
    query, all_records = _input_args
    record_indexs = set(range(len(all_records)))
    for col_values_pair in query:
        col = col_values_pair[0]
        values = col_values_pair[1]
        for record_index, record in enumerate(all_records):
            if len(set(record).intersection(set(itertools.product([col], values)))) == 0:
                record_indexs.discard(record_index)
            if len(record_indexs) == 0:
                return query, []
    return query, list(record_indexs)

def _find_useful_query(input_args):
    query, _unique_record, sens_attribute = input_args
    _query = {col: values for col, values in query}
    conflicted = False
    if sens_attribute in _query:
        for col in _query:
            if col != sens_attribute:
                if _unique_record[col] not in _query[col]:
                    conflicted = True
        if conflicted is False:
            return query
    return None

def _to_potential_record(input_args):
    domain_config, record = input_args
    _ext_potential_record = {col: val for col, val in zip(domain_config, record)}
    _potential_record = {}
    for col in domain_config:
        if col != "cenrace":
            _potential_record.update({col: _ext_potential_record[col]})
        else:
            detail_race = id_to_race[_ext_potential_record["cenrace"]]
            _potential_record.update({majority_race: int(detail_race.__dict__[majority_race]) for majority_race in detail_race.majority_races})
    potential_record = tuple(_potential_record.values())
    return potential_record

# Convert census race between single integer label and multiple binary labels
# Modify from the code examples/data_preprocessing/ppmf/census_race.py in dp-query-release repository
@dataclass(frozen=True, eq=True)
class CensusRace:
    """
    Encodes one of the 63 values for the CENRACE column. The 63 different race
    values correspond to all possible non-empty subsets of 6 race categories:
    - White
    - Black or African American
    - American Indian and Alaska Native
    - Asian
    - Native Hawaiian and Other Pacific Islander
    - Some Other Race
    """

    white: bool = False
    black_or_african_american: bool = False
    american_indian_and_alaska_native: bool = False
    asian: bool = False
    native_hawaiian_and_other_pacific_islander: bool = False
    some_other_race: bool = False
    majority_races = ["white", "black_or_african_american", "american_indian_and_alaska_native", "asian", "native_hawaiian_and_other_pacific_islander", "some_other_race"]

    def num_races(self) -> int:
        """`num_races()

        Returns the number of race categories that this race belongs to. Always
        an integer between 1 and 6 inclusive.
        """
        return (
            self.white
            + self.black_or_african_american
            + self.american_indian_and_alaska_native
            + self.asian
            + self.native_hawaiian_and_other_pacific_islander
            + self.some_other_race
        )

    def to_id(self) -> int:
        """`to_id()`
        
        Return the numeric value that encodes this race in the CENRACE column.
        """
        return race_to_id[self]

    def __str__(self):
        return _race_strs[self.to_id()]

    @staticmethod
    def from_id(id: int):
        """`from_id(id: int) -> CensusRace
        
        Return the instance of `CensusRace` that is encoded by `id` in the
        CENRACE column.
        """
        return id_to_race[id]

    @staticmethod
    def parse_census_race(race_str: str):
        """`parse_census_race(race_str: str) -> CensusRace`

        A helper function that parses the text descriptions of a census race.
        """
        result = CensusRace()
        if race_str.endswith(" alone"):
            race_str = race_str[:-6]
        race_str = race_str.lower()
        races = [r.strip() for r in race_str.split(";") if r != ""]

        for race in races:
            field_name = race.replace(" ", "_")
            if field_name not in result.__dict__.keys():
                raise KeyError(f"Race {race} is not one of the census races")
            result.__dict__[race.replace(" ", "_")] = True
        return result

    @staticmethod
    def from_predicate(pred):
        """`from_predicate(pred: Callable[[CensusRace]m, bool]) -> List[CensusRace]`
        
        Given a predicate defined over instances of `CensusRace`, returns the
        list of all `CensusRace` instances that satisfy this predicate.
        """
        return [cr for cr in id_to_race[1:] if pred(cr)]


# Obtained from https://www2.census.gov/programs-surveys/decennial/2020/program-management/data-product-planning/2010-demonstration-data-products/ppmf/2020-05-27-ppmf-record-layout.pdf
_race_strs = [
    "White alone",
    "Black or African American alone",
    "American Indian and Alaska Native alone",
    "Asian alone",
    "Native Hawaiian and Other Pacific Islander alone",
    "Some Other Race alone",
    "White; Black or African American",
    "White; American Indian and Alaska Native",
    "White; Asian",
    "White; Native Hawaiian and Other Pacific Islander",
    "White; Some Other Race",
    "Black or African American; American Indian and Alaska Native",
    "Black or African American; Asian",
    "Black or African American; Native Hawaiian and Other Pacific Islander",
    "Black or African American; Some Other Race",
    "American Indian and Alaska Native; Asian",
    "American Indian and Alaska Native; Native Hawaiian and Other Pacific Islander",
    "American Indian and Alaska Native; Some Other Race",
    "Asian; Native Hawaiian and Other Pacific Islander",
    "Asian; Some Other Race",
    "Native Hawaiian and Other Pacific Islander; Some Other Race",
    "White; Black or African American; American Indian and Alaska Native",
    "White; Black or African American; Asian",
    "White; Black or African American; Native Hawaiian and Other Pacific Islander",
    "White; Black or African American; Some Other Race",
    "White; American Indian and Alaska Native; Asian",
    "White; American Indian and Alaska Native; Native Hawaiian and Other Pacific Islander",
    "White; American Indian and Alaska Native; Some Other Race",
    "White; Asian; Native Hawaiian and Other Pacific Islander",
    "White; Asian; Some Other Race",
    "White; Native Hawaiian and Other Pacific Islander; Some Other Race",
    "Black or African American; American Indian and Alaska Native; Asian",
    "Black or African American; American Indian and Alaska Native; Native Hawaiian and Other Pacific Islander",
    "Black or African American; American Indian and Alaska Native; Some Other Race",
    "Black or African American; Asian; Native Hawaiian and Other Pacific Islander",
    "Black or African American; Asian; Some Other Race",
    "Black or African American; Native Hawaiian and Other Pacific Islander; Some Other Race",
    "American Indian and Alaska Native; Asian; Native Hawaiian and Other Pacific Islander",
    "American Indian and Alaska Native; Asian; Some Other Race",
    "American Indian and Alaska Native; Native Hawaiian and Other Pacific Islander; Some Other Race",
    "Asian; Native Hawaiian and Other Pacific Islander; Some Other Race",
    "White; Black or African American; American Indian and Alaska Native; Asian",
    "White; Black or African American; American Indian and Alaska Native; Native Hawaiian and Other Pacific Islander",
    "White; Black or African American; American Indian and Alaska Native; Some Other Race",
    "White; Black or African American; Asian; Native Hawaiian and Other Pacific Islander",
    "White; Black or African American; Asian; Some Other Race",
    "White; Black or African American; Native Hawaiian and Other Pacific Islander; Some Other Race",
    "White; American Indian and Alaska Native; Asian; Native Hawaiian and Other Pacific Islander",
    "White; American Indian and Alaska Native; Asian; Some Other Race",
    "White; American Indian and Alaska Native; Native Hawaiian and Other Pacific Islander; Some Other Race",
    "White; Asian; Native Hawaiian and Other Pacific Islander; Some Other Race",
    "Black or African American; American Indian and Alaska Native; Asian; Native Hawaiian and Other Pacific Islander",
    "Black or African American; American Indian and Alaska Native; Asian; Some Other Race",
    "Black or African American; American Indian and Alaska Native; Native Hawaiian and Other Pacific Islander; Some Other Race",
    "Black or African American; Asian; Native Hawaiian and Other Pacific Islander; Some Other Race",
    "American Indian and Alaska Native; Asian; Native Hawaiian and Other Pacific Islander; Some Other Race",
    "White; Black or African American; American Indian and Alaska Native; Asian; Native Hawaiian and Other Pacific Islander",
    "White; Black or African American; American Indian and Alaska Native; Asian; Some Other Race",
    "White; Black or African American; American Indian and Alaska Native; Native Hawaiian and Other Pacific Islander; Some Other Race",
    "White; Black or African American; Asian; Native Hawaiian and Other Pacific Islander; Some Other Race",
    "White; American Indian and Alaska Native; Asian; Native Hawaiian and Other Pacific Islander; Some Other Race",
    "Black or African American; American Indian and Alaska Native; Asian; Native Hawaiian and Other Pacific Islander; Some Other Race",
    "White; Black or African American; American Indian and Alaska Native; Asian; Native Hawaiian and Other Pacific Islander; Some Other Race",
]

id_to_race = [CensusRace.parse_census_race(race_str) for race_str in _race_strs]
race_to_id = {race: id for (id, race) in enumerate(id_to_race)}