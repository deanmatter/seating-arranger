import pandas as pd
import numpy as np
import json
from collections import defaultdict
import random

# DEFINITIONS
# a circle is a collection of guests, which hopefully will get along. Corresponds to a table
# a group is an external collection of guests, sharing a common interest, location, church or friendship group
# a point is a unit of 'likelihood' that a guest will get along with another guest

# read in the list of guests as a pandas df, and combine name and surname into the name col only, drop surname col
guests = pd.read_csv("guests.csv")
guests["name"] = guests["name"] + " " + guests["surname"]
guests.drop("surname", axis=1, inplace=True)

# the actual table arrangements existing
tables = [14, 14, 14, 14, 14, 14, 9]

# the number of points given if two people are in same group
group_points = {
    "family": 40,
    "plett": 10,
    "enwc": 10,
    "enstb": 10,
    "knysnavin": 10,
    "Matter-side": 10,
    "Neville-side": 10,
    "damgroup": 10,
    "metanoia": 5,
    "oldness": 20,
    "exclusions": -100,
    "great friends": 20,
    "good friends": 10,
    "avg friends": 5,
}


def get_getalong_matrix(guests, group_points):
    """Generate a len(guests) x len(guests) matrix, with each pairwise
    entry i,j being the total of points between guest i and j,
    calculated according to which groups the guests share externally."""
    m = np.zeros((len(guests), len(guests)))
    for i, guest_i in guests.iterrows():
        print(f"Calculating for {guest_i['name']}")
        for j, guest_j in guests.iterrows():
            if i == j:
                continue
            for point_category in group_points:
                m[i][j] += (
                    group_points[point_category]
                    if guest_i[point_category] == guest_j[point_category]
                    else 0
                )
    return m


def construct_initial_circles(guests, getalong_matrix):
    """Given the guests dataframe, constructs an initial dict of circles, with each family in a circle"""
    circles = {}
    for g_index, guest in guests.iterrows():
        if guest["family"] in circles:
            # add this guest to the family's circle, and update the circle's points tally
            circles[guest["family"]][
                "circle_points"
            ] += sum(  # additional points to circle because of guest addition
                (  # use a generator to save memory
                    getalong_matrix[g_index][
                        guest_j.name
                    ]  # guest_j is pd.Series, thus .name is the index in guests DF
                    for guest_j in circles[guest["family"]][
                        "circle_guests"
                    ]  # for each guest already in the circle
                )
            )
            circles[guest["family"]]["circle_guests"].append(guest)
        elif guest["name"] in circles:
            print(f"A guest {guest['name']} already exists in this circle.")
            raise LookupError
        else:
            # Define the circle with the initial guest
            if pd.notna(guest["family"]):
                # if guest is in a family group (i.e. family is not null)
                circles.update(
                    {guest["family"]: {"circle_points": 0, "circle_guests": [guest]}}
                )
                print(f"Circle made for {guest['family']}")
            else:
                # guest is single
                circles.update(
                    {guest["name"]: {"circle_points": 0, "circle_guests": [guest]}}
                )
                print(f"Circle made for {guest['name']}")

    # add all circle guests to be in the 'start' and 'end' of the circle.
    # when circles are merged with others, this will either be at the start or end of it
    for circle in circles:
        circles[circle]["start_guests"] = circles[circle]["circle_guests"]
        circles[circle]["end_guests"] = circles[circle]["circle_guests"]

    return circles


def points_from_merging_circles(circle_1, circle_2, getalong_matrix):
    """
    How many connection points would arise from a given circle being merged with another given circle?
    Merges are done according to the 'start' and 'end' of each circle. The start and end of circle 1
    can be paired with the start or end of the other circle.
    This allows for easier arrangements of who sits next to whom in large circles.
    """
    # try match 'start' of 1 with end of 2
    start1_to_end2 = 0
    for g_1 in circle_1["start_guests"]:
        for g_2 in circle_2["end_guests"]:
            # g1 and g2 are pd.Series from the guests DF, so .name is the index in guests
            start1_to_end2 += getalong_matrix[g_1.name][g_2.name]

    end1_to_end2 = 0
    if circle_1["start_guests"] != circle_1["end_guests"]:
        # try match 'end' of 1 with end of 2
        for g_1 in circle_1["end_guests"]:
            for g_2 in circle_2["end_guests"]:
                end1_to_end2 += getalong_matrix[g_1.name][g_2.name]

    # try match start of 1 to start of 2
    start1_to_start2 = 0
    for g_1 in circle_1["start_guests"]:
        for g_2 in circle_2["start_guests"]:
            start1_to_start2 += getalong_matrix[g_1.name][g_2.name]

    end1_to_start2 = 0
    if circle_2["start_guests"] != circle_2["end_guests"]:
        # try match end of 1 with start of 2
        for g_1 in circle_1["end_guests"]:
            for g_2 in circle_2["start_guests"]:
                end1_to_start2 += getalong_matrix[g_1.name][g_2.name]

    # return the max of combining the two
    best_merge_points = max(
        start1_to_start2, start1_to_end2, end1_to_start2, end1_to_end2
    )
    # if end1_to_start2 == best_merge_points:
    #     print("end1 to start 2")
    # elif end1_to_end2 == best_merge_points:
    #     print("end1_to_end2")
    # elif start1_to_end2 == best_merge_points:
    #     print("start1 to end 2")
    # elif start1_to_start2 == best_merge_points:
    #     print("start1_to_start2")
    # else:
    #     print("Unidentified best merge strategy")

    return best_merge_points


def get_circle_merge_matrix(circles, getalong_matrix):
    """Return a len(circles) x len(circles) ndarray, where for circles[i] and circles[j],
    entry i,j will be the change in points resulting from the circles being merged."""
    # initialize the matrix (an ndarray, actually), to be all NaN values
    merge_matrix = np.empty((len(circles), len(circles)))
    merge_matrix[:] = np.NaN
    circle_indexes = {circle_name: index for index, circle_name in enumerate(circles)}

    # iterate for each circle pair i and j
    for circle_i in circles:
        print(f"Checking circle value for {circle_i}")
        circle_i_idx = circle_indexes[circle_i]
        for circle_j in circles:
            circle_j_idx = circle_indexes[circle_j]
            if circle_i == circle_j:
                continue
            elif pd.notna(merge_matrix[circle_j_idx][circle_i_idx]) and pd.notna(
                merge_matrix[circle_i_idx][circle_j_idx]
            ):
                # if merge matrix values i,j and j,i both not NaN (set already), continue
                continue
            else:
                # set i,j and j,i to be the points change from combining them
                merge_matrix[circle_j_idx][circle_i_idx] = points_from_merging_circles(
                    circles[circle_i], circles[circle_j], getalong_matrix
                )
                merge_matrix[circle_i_idx][circle_j_idx] = merge_matrix[circle_j_idx][
                    circle_i_idx
                ]
    return merge_matrix


def merge_circles(circles, circle_merge_matrix, getalong_matrix):
    """Recursively merge circles in order to maximise total circle points"""
    # the pair of circles with the max points resulting from them being merged
    max_index_pair = np.unravel_index(
        np.nanargmax(circle_merge_matrix), circle_merge_matrix.shape
    )
    circle_names = [list(circles.items())[index][0] for index in max_index_pair]
    print(f"Merging {circle_names[0]} and {circle_names[1]}.")

    # TODO add either to the start or end of the othercircle

    new_circle_name = f"{circle_names[0]}-{circle_names[1]}"
    circles[new_circle_name] = {
        "circle_points": circles[circle_names[0]]["circle_points"]  # circle 1's points
        + circles[circle_names[1]]["circle_points"]  # plus circle 2's points
        + circle_merge_matrix[max_index_pair[0]][
            max_index_pair[1]
        ],  # plus the points gained by merging them
        "circle_guests": circles[circle_names[0]]["circle_guests"]
        + circles[circle_names[1]]["circle_guests"],
    }
    circles[new_circle_name].update(
        {
            "start_guests": circles[circle_names[0]]["circle_guests"],
            "end_guests": circles[circle_names[1]]["circle_guests"],
        }
    )

    # delete each of the rows, and columns represented by the previous
    circle_merge_matrix = np.delete(circle_merge_matrix, max_index_pair[0], 0)
    circle_merge_matrix = np.delete(circle_merge_matrix, max_index_pair[1], 0)
    circle_merge_matrix = np.delete(circle_merge_matrix, max_index_pair[0], 1)
    circle_merge_matrix = np.delete(circle_merge_matrix, max_index_pair[1], 1)

    # add the new circle to the matrix
    print(circle_merge_matrix)
    new_circle_points = [
        points_from_merging_circles(circles[new_circle_name], circle, getalong_matrix)
        for circle in circles
    ]
    circle_merge_matrix.append(new_circle_points)
    print(circle_merge_matrix)

    quit()

    del circles[circle_names[0]]
    del circles[circle_names[1]]

    quit()

    if len(circles <= 7):  # if all circles are full
        return circles
    else:
        merge_circles(circles, circle_merge_matrix)


m = get_getalong_matrix(guests, group_points)
# print("Who is the most integrated into the wedding party? i.e. fits with most people?")
# print(m.mean(axis=1))
# print("Who is the best match for each person, in terms of groups?")
# print([(guests["name"][i], guests["name"][np.argmax(m[i])]) for i in range(len(m))])

print("Constructing initial set of circles...")
circles = construct_initial_circles(guests=guests, getalong_matrix=m)

print("Calculating the value of merging each circle with each other one...")
circle_merge_matrix = get_circle_merge_matrix(circles=circles, getalong_matrix=m)

# Merge circles recursively
final_circles = merge_circles(circles, circle_merge_matrix, getalong_matrix=m)

while True:

    no_significant_improvement = True
    if no_significant_improvement:
        break
