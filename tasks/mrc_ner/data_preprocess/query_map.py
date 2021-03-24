#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# file: query_map.py
# ---------------------------------------------
# query collections for different dataset
# ---------------------------------------------

en_conll03_query = {
    "default": {
        "ORG": "organization entities are limited to named corporate, governmental, or other organizational entities.",
        "PER": "person entities are named persons or family.",
        "LOC": "location entities are the name of politically or geographically defined locations such as cities, provinces, countries, international regions, bodies of water, mountains, etc.",
        "MISC": "examples of miscellaneous entities include events, nationalities, products and works of art."
    },
    "labels": ["ORG", "PER", "LOC", "MISC"]
}

queries_for_dataset = {
    "en_conll03": en_conll03_query
}



