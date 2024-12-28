import re


def parse_points(data):
    """
    Parses the input data and extracts all Original and Updated point pairs.

    Args:
        data (str): The multiline string containing Original and Updated points.

    Returns:
        list of tuples: Each tuple contains two tuples representing the Original and Updated points.
    """
    # Regular expression to capture Original and Updated points with three floats each
    pattern = r"Original Point \d+: \[\s*([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s*\].*?Updated Point \d+: \[\s*([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s*\]"

    # Find all matches using re.DOTALL to allow '.' to match newlines
    matches = re.findall(pattern, data, re.DOTALL)

    # Convert matches to a list of tuples: [((x1, y1, z1), (x2, y2, z2)), ...]
    point_pairs = [((float(m[0]), float(m[1]), float(m[2])),
                    (float(m[3]), float(m[4]), float(m[5]))) for m in matches]

    return point_pairs


def build_mapping(point_pairs):
    """
    Builds a dictionary mapping Original points to their Updated points.

    Args:
        point_pairs (list of tuples): Each tuple contains Original and Updated point tuples.

    Returns:
        dict: A dictionary where keys are Original point tuples and values are Updated point tuples.
    """
    mapping = {}
    for original, updated in point_pairs:
        mapping[original] = updated
    return mapping


def find_chains(mapping):
    """
    Finds all continuous evolution chains based on the mapping.

    Args:
        mapping (dict): A dictionary mapping Original points to Updated points.

    Returns:
        list of lists: Each sublist represents a chain of points in evolution order.
    """
    # Create a reverse mapping to identify which points are updated from others
    updated_to_original = {
        updated: original for original, updated in mapping.items()}

    # Identify starting points: Original points that are not updated points themselves
    starting_points = set(mapping.keys()) - set(updated_to_original.keys())

    chains = []

    for start in starting_points:
        chain = [start]
        current = start
        while current in mapping:
            next_point = mapping[current]
            if next_point in chain:
                # Detected a cycle; stop to prevent infinite loop
                break
            chain.append(next_point)
            current = next_point
        chains.append(chain)

    return chains


def format_chain(chain):
    """
    Formats a chain of points into a string representation.

    Args:
        chain (list of tuples): A list of point tuples.

    Returns:
        str: Formatted string representing the evolution chain.
    """
    return " --> ".join([f"[{p[0]}  {p[1]}  {p[2]}]" for p in chain])


def main():
    data = """
    Original Point 0: [26.   0.7  0.6] Real Acc: 0.5099957113628952
         Updated Point 0: [24.    0.64  0.66] Expected Acc: 0.5555283157407533
    Original Point 1: [22.   0.9  0.8] Real Acc: 0.5881263697857567
         Updated Point 1: [20.    0.84  0.86] Expected Acc: 0.6080616345110444
    Original Point 2: [22.   0.8  0.8] Real Acc: 0.5657434123945486
         Updated Point 2: [22.    0.74  0.86] Expected Acc: 0.5857681724317321
    Original Point 3: [22.    0.85  0.6 ] Real Acc: 0.5387847975151788
         Updated Point 3: [20.    0.79  0.66] Expected Acc: 0.5840809352208107
    Original Point 4: [24.    0.65  0.7 ] Real Acc: 0.5778543945409995
         Updated Point 4: [22.    0.6   0.76] Expected Acc: 0.5982598441770238
    Original Point 5: [22.    0.7   0.65] Real Acc: 0.5661715363665216
         Updated Point 5: [20.    0.64  0.71] Expected Acc: 0.6045717915229212
    Original Point 6: [26.    0.6   0.85] Real Acc: 0.5941787748378712
         Updated Point 6: [24.   0.6  0.9] Expected Acc: 0.5978088581201136
    Original Point 7: [24.    0.6   0.85] Real Acc: 0.5710188190480344
         Updated Point 7: [22.   0.6  0.9] Expected Acc: 0.6205692569187079
    Original Point 8: [26.    0.65  0.8 ] Real Acc: 0.5465367144988497
         Updated Point 8: [24.    0.6   0.86] Expected Acc: 0.5914347401939224
    Original Point 9: [26.   0.6  0.6] Real Acc: 0.5108356920119942
         Updated Point 9: [24.    0.6   0.66] Expected Acc: 0.553699557494012
    
    Original Point 0: [22.   0.6  0.9] Real Acc: 0.567209021500026
        Updated Point 0: [22.    0.66  0.9 ] Expected Acc: 0.5842443284414894
    Original Point 1: [20.    0.84  0.86] Real Acc: 0.6080532499007012
        Updated Point 1: [20.   0.9  0.9] Expected Acc: 0.6155786244264169
    Original Point 2: [20.    0.64  0.71] Real Acc: 0.5862940969402466
        Updated Point 2: [20.    0.6   0.65] Expected Acc: 0.5867686074779428
    Original Point 3: [22.    0.6   0.76] Real Acc: 0.5388684124291536
        Updated Point 3: [22.   0.6  0.7] Expected Acc: 0.5758416801965511
    Original Point 4: [24.   0.6  0.9] Real Acc: 0.5520680342421449
        Updated Point 4: [24.    0.66  0.9 ] Expected Acc: 0.5876637449523945
    Original Point 5: [24.    0.6   0.86] Real Acc: 0.5625191752048797
        Updated Point 5: [24.    0.66  0.9 ] Expected Acc: 0.5876637449523958
    Original Point 6: [22.    0.74  0.86] Real Acc: 0.6008587940155361
        Updated Point 6: [20.   0.8  0.9] Expected Acc: 0.6065400601984644
    Original Point 7: [20.    0.79  0.66] Real Acc: 0.5624829592772689
        Updated Point 7: [20.    0.73  0.72] Expected Acc: 0.5840050365212582
    
    Original Point 0: [24.    0.66  0.9 ] Real Acc: 0.5522104902536576
        Updated Point 0: [24.    0.72  0.84] Expected Acc: 0.6042265573253653
    Original Point 1: [20.    0.6   0.65] Real Acc: 0.5453770948899901
        Updated Point 1: [20.    0.66  0.71] Expected Acc: 0.5739591089295003
    Original Point 2: [22.    0.66  0.9 ] Real Acc: 0.5541483703198384
        Updated Point 2: [22.    0.72  0.84] Expected Acc: 0.6045202711023919
    Original Point 3: [20.    0.73  0.72] Real Acc: 0.5760697938021349
        Updated Point 3: [20.    0.71  0.7 ] Expected Acc: 0.5712851470027418
    Original Point 4: [22.   0.6  0.7] Real Acc: 0.5710496580034082
        Updated Point 4: [22.    0.66  0.76] Expected Acc: 0.5781133275832823
    
    Original Point 0: [22.    0.72  0.84] Real Acc: 0.5896170093425321
        Updated Point 0: [20.    0.78  0.78] Expected Acc: 0.6037804140234314
    Original Point 1: [24.    0.72  0.84] Real Acc: 0.5844893998380724
        Updated Point 1: [22.    0.78  0.78] Expected Acc: 0.5936013500809927
    Original Point 2: [22.    0.66  0.76] Real Acc: 0.5722931002077565
        Updated Point 2: [22.    0.72  0.7 ] Expected Acc: 0.579241996027133
    Original Point 3: [20.    0.66  0.71] Real Acc: 0.5488322007692451
        Updated Point 3: [22.    0.72  0.77] Expected Acc: 0.5838527951963526
    Original Point 4: [20.    0.71  0.7 ] Real Acc: 0.5571172370747511
        Updated Point 4: [20.    0.71  0.76] Expected Acc: 0.5692038522249591
    
    Original Point 0: [20.    0.78  0.78] Real Acc: 0.5930589514216941
        Updated Point 0: [20.    0.72  0.84] Expected Acc: 0.6119686075068582
    Original Point 1: [22.    0.78  0.78] Real Acc: 0.5635191132498963
        Updated Point 1: [20.    0.72  0.84] Expected Acc: 0.6122379782633246
    Original Point 2: [22.    0.72  0.77] Real Acc: 0.5800209927333381
        Updated Point 2: [22.    0.78  0.77] Expected Acc: 0.5872237320297834
    Original Point 3: [22.    0.72  0.7 ] Real Acc: 0.5995094744162499
        Updated Point 3: [24.    0.78  0.76] Expected Acc: 0.623685239971078
    Original Point 4: [20.    0.71  0.76] Real Acc: 0.5715991757262644
        Updated Point 4: [22.    0.77  0.74] Expected Acc: 0.6112807685721735
    
    Original Point 0: [24.    0.78  0.76] Real Acc: 0.5830099191912641
        Updated Point 0: [26.    0.72  0.7 ] Expected Acc: 0.6160797185417927
    Original Point 1: [20.    0.72  0.84] Real Acc: 0.5927878094358663
        Updated Point 1: [20.    0.78  0.9 ] Expected Acc: 0.614011149933264
    Original Point 2: [22.    0.77  0.74] Real Acc: 0.5732068219803639
        Updated Point 2: [22.    0.75  0.68] Expected Acc: 0.5917618987649556
    Original Point 3: [22.    0.78  0.77] Real Acc: 0.5835288071208106
        Updated Point 3: [22.    0.8   0.71] Expected Acc: 0.5831206329490058

    Original Point 0: [26.    0.72  0.7 ] Real Acc: 0.5572719468865925
        Updated Point 0: [24.    0.66  0.76] Expected Acc: 0.6013018844348904
    Original Point 1: [22.    0.75  0.68] Real Acc: 0.586348394392175
        Updated Point 1: [24.    0.81  0.74] Expected Acc: 0.6113952225448849
    Original Point 2: [22.    0.8   0.71] Real Acc: 0.5887243155906432
        Updated Point 2: [24.    0.74  0.65] Expected Acc: 0.6096096699311906

    Original Point 0: [24.    0.81  0.74] Real Acc: 0.5830723883872316
        Updated Point 0: [24.    0.87  0.68] Expected Acc: 0.6059712003390775
    Original Point 1: [24.    0.74  0.65] Real Acc: 0.5247660432311216
        Updated Point 1: [22.    0.68  0.71] Expected Acc: 0.6017710962967061
    Original Point 2: [24.    0.66  0.76] Real Acc: 0.5401279334430764
        Updated Point 2: [22.    0.72  0.7 ] Expected Acc: 0.5957394385167626
    
    Original Point 0: [24.    0.87  0.68] Real Acc: 0.5578031225089123
        Updated Point 0: [24.    0.81  0.74] Expected Acc: 0.5803401503115885
    Original Point 1: [22.    0.68  0.71] Real Acc: 0.5684431524896407
        Updated Point 1: [24.    0.74  0.77] Expected Acc: 0.5908484666863153
    
    Original Point 0: [24.    0.74  0.77] Real Acc: 0.5974517023972802
        Updated Point 0: [26.    0.68  0.71] Expected Acc: 0.629419217220853
    """

    # Step 1: Parse the points
    point_pairs = parse_points(data)

    # Step 2: Build the mapping from Original to Updated points
    mapping = build_mapping(point_pairs)

    # Step 3: Find all evolution chains
    chains = find_chains(mapping)

    # Step 4: Print all chains
    print("Evolution Chains:\n")
    for idx, chain in enumerate(chains):
        print(f"Chain {idx}: {format_chain(chain)}")


if __name__ == "__main__":
    main()
