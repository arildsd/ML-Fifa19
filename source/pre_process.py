import pandas
import math
import numpy as np


INPUT_FILE_PATH = r"../data/out_field_players.csv"
PROCESSED_OUTPUT_FILE_PATH = r"../data/processed.csv"


def truncate(number, digits) -> float:
    stepper = pow(10.0, digits)
    return math.trunc(stepper * number) / stepper


def pre_process():
    # Read the file
    df = pandas.read_csv(INPUT_FILE_PATH)

    # Parse wage
    df = process_wage(df)

    # Parse value
    #vector_values = process_value(df)

    # Convert height to meter
    df = convert_height(df)

    # Convert weight to kgs
    df = convert_weight(df)

    # Parse values and add them for each position
    df = process_positions(df)

    # Remove date for loaned players
    df = process_loaned_players(df)

    # Make binary
    df = process_preferred_foot(df)
    df = process_face(df)

    # Split into two, make integer values
    df = process_work_rate(df)

    # Remove date, fix the loaned players
    df = process_joined(df)

    # One hot encode position
    df = position_one_hot_encode(df)

    # One hot encode nationality
    df = nationality_one_hot_encode(df)

    # Remove players with market value equal to 0
    df = remove_zero_values(df)

    # Replace the club name with the average overall score for the players in the club
    df = process_club(df)

    # Make loaned from binary, 1 if a player is on loan, 0 otherwise
    df = process_loaned_from(df)

    # Remove ID, Name, Photo, Flag, ClubLogo, Special, BodyType, Release Clause
    df = remove_columns(df)

    # Normalize the data by subtracting the mean and dividing by the standard deviation
    df = normalize(df)

    return df


def process_wage(df):
    df['Wage'] = df['Wage'].map(lambda x: parse_values(x))
    return df


def process_value(df):
    df['Value'] = df['Value'].map(lambda x: parse_values(x))
    values_vector = df['Value']
    values_vector.to_csv("../data/value.csv", index=False)



def parse_values(value):
    value_string = str(value)
    value_parsed = value_string.lstrip('£$€â‚¬').rstrip('kKmM')
    if 'K' in value_string:
        return float(value_parsed) * 1000
    elif 'M' in value_string:
        return float(value_parsed) * 1000000
    else:
        return value_parsed


def convert_height(df):
    df['Height'] = df['Height'].map(lambda x: feet_to_meters(x))
    return df


def feet_to_meters(height):
    split_string = str(height).split("'")
    if len(split_string) > 1:
        feet = split_string[0]
        inches = split_string[1]
        height_in_meters = 0.3048 * int(feet) + 0.0254 * int(inches)
        return truncate(height_in_meters, 2)
    else:
        return height


def convert_weight(df):
    df['Weight'] = df['Weight'].map(lambda x: lbs_to_kg(x))
    return df


def lbs_to_kg(weight):
    weight_string = str(weight).rstrip('lbsLBS')
    weight_in_kgs = 0.45359237 * float(weight_string)
    if math.isnan(weight_in_kgs):
        return weight_in_kgs
    else:
        return truncate(weight_in_kgs, 2)


def process_positions(df):
    df['LS'] = df['LS'].map(lambda x: parse_add_position(x))
    df['ST'] = df['ST'].map(lambda x: parse_add_position(x))
    df['RS'] = df['RS'].map(lambda x: parse_add_position(x))
    df['LW'] = df['LW'].map(lambda x: parse_add_position(x))
    df['LF'] = df['LF'].map(lambda x: parse_add_position(x))
    df['CF'] = df['CF'].map(lambda x: parse_add_position(x))
    df['RF'] = df['RF'].map(lambda x: parse_add_position(x))
    df['RW'] = df['RW'].map(lambda x: parse_add_position(x))
    df['LAM'] = df['LAM'].map(lambda x: parse_add_position(x))
    df['CAM'] = df['CAM'].map(lambda x: parse_add_position(x))
    df['RAM'] = df['RAM'].map(lambda x: parse_add_position(x))
    df['LM'] = df['LM'].map(lambda x: parse_add_position(x))
    df['LCM'] = df['LCM'].map(lambda x: parse_add_position(x))
    df['CM'] = df['CM'].map(lambda x: parse_add_position(x))
    df['RCM'] = df['RCM'].map(lambda x: parse_add_position(x))
    df['RM'] = df['RM'].map(lambda x: parse_add_position(x))
    df['LWB'] = df['LWB'].map(lambda x: parse_add_position(x))
    df['LDM'] = df['LDM'].map(lambda x: parse_add_position(x))
    df['CDM'] = df['CDM'].map(lambda x: parse_add_position(x))
    df['RDM'] = df['RDM'].map(lambda x: parse_add_position(x))
    df['RWB'] = df['RWB'].map(lambda x: parse_add_position(x))
    df['LB'] = df['LB'].map(lambda x: parse_add_position(x))
    df['LCB'] = df['LCB'].map(lambda x: parse_add_position(x))
    df['CB'] = df['CB'].map(lambda x: parse_add_position(x))
    df['RCB'] = df['RCB'].map(lambda x: parse_add_position(x))
    df['RB'] = df['RB'].map(lambda x: parse_add_position(x))

    return df


def parse_add_position(position):
    if str(position) in 'nan':
        return position
    else:
        values = str(position).split('+')
        total_position = int(values[0]) + int(values[1])
        return total_position


def process_loaned_players(df):
    df = df.dropna(subset=["Contract Valid Until"])
    df['Contract Valid Until'] = df['Contract Valid Until'].map(lambda x: remove_date(x)).astype(int)
    return df


def remove_date(contract):
    # Has date in their contract
    if len(str(contract)) > 4:
        contract_string = contract.split(', ')
        year = contract_string[1]
        return year
    return contract


def process_preferred_foot(df):
    df['Preferred Foot'] = df['Preferred Foot'].map(lambda x: make_foot_binary(x))
    return df


def make_foot_binary(foot):
    if foot == 'Left':
        return 0
    elif foot == 'Right':
        return 1


def process_face(df):
    df['Real Face'] = df['Real Face'].map(lambda x: make_face_binary(x))
    return df


def make_face_binary(face):
    if face == 'No':
        return 0
    elif face == 'Yes':
        return 1


def process_work_rate(df):
    df['Work Rate Attack'] = df['Work Rate'].map(lambda x: get_attack_work_rate(x))
    df['Work Rate Defence'] = df['Work Rate'].map(lambda x: get_defence_work_rate(x))
    return df


def get_attack_work_rate(work_rate):
    work_rate_split = work_rate.split('/ ')
    work_rate_attack = work_rate_split[0]
    return make_work_rate_integer(work_rate_attack)


def get_defence_work_rate(work_rate):
    work_rate_split = work_rate.split('/ ')
    work_rate_defence = work_rate_split[1]
    return make_work_rate_integer(work_rate_defence)


def make_work_rate_integer(work_rate):
    if work_rate == 'Low':
        return 0
    elif work_rate == 'Medium':
        return 1
    elif work_rate == 'High':
        return 2


def process_joined(df):
    df['Joined'] = df['Joined'].map(lambda x: get_joined_year(x))
    df["Joined"] = df.apply(lambda x: _process_joined_auxiliary(x), axis=1)
    df["Joined"] = df["Joined"].astype(int)

    return df


def get_joined_year(joined_date):
    if (len(str(joined_date)) > 0) and not (str(joined_date) == 'nan'):    # Has valid contract and not on loan
        joined_date_split = str(joined_date).split(', ')
        date = joined_date_split[0]
        year = joined_date_split[1]
        return year
    else:
        return joined_date

def _process_joined_auxiliary(row):
    if str(row["Joined"]) == "nan":
        new_year = str(int(row["Contract Valid Until"]) - 3)
        return new_year
    else:
        return row["Joined"]




def position_one_hot_encode(df):
    df["Position"] = pandas.Categorical(df["Position"])
    dfDummies = pandas.get_dummies(df['Position'], prefix='is_position')
    df = pandas.concat([df, dfDummies], axis=1)
    return df


def nationality_one_hot_encode(df):
    df["Nationality"] = pandas.Categorical(df["Nationality"])
    dfDummies = pandas.get_dummies(df['Nationality'], prefix='has_nationality')
    df = pandas.concat([df, dfDummies], axis=1)
    return df


def remove_zero_values(df):
    df = df.drop(df[pandas.to_numeric(df.Value, errors='coerce') == 0].index)
    return df


def process_loaned_from(df):
    df['Is_Loaned'] = df['Loaned From'].map(lambda x: is_loaned_from(x))
    df["Loaned_From_Overall"] = df.apply(lambda x: _process_loaned_from_auxiliary(x, df), axis=1)
    return df


def _process_loaned_from_auxiliary(row, df):
    # Loaned from is nan, replace with current clubs overall
    if not row["Is_Loaned"]:
        return row["Club_Overall"]

    # Find the loaned from club's overall rating and return it
    for i, search_row in df[["Club", "Club_Overall"]].drop_duplicates().iterrows():
        if search_row["Club"] == row["Loaned From"]:
            return search_row["Club_Overall"]

    return row["Club_Overall"]




def is_loaned_from(loaned_club):
    if str(loaned_club) == "nan":   # Player is not on loan
        return 0
    else:
        return 1


def remove_columns(df):
    df = df.drop(columns=["ID", "Name", "Photo", "Flag", "Club Logo", "Special", "Body Type", "Release Clause", "Club",
                          "Unnamed: 0", "Unnamed: 0.1", "Nationality", "Work Rate", "GKDiving", "GKHandling",
                          "GKPositioning", "GKReflexes", "Value", "Position", "Loaned From"])
    return df


def process_club(df):
    reduced_df = df[["Club", "Overall"]]
    club_means = reduced_df.groupby('Club', as_index=False).mean()
    df["Club_Overall"] = df.apply(lambda x: _process_club_auxiliary(x, club_means), axis=1)
    return df


def _process_club_auxiliary(row, club_means):
    if row["Club"] == "nan":
        return row["Overall"]
    else:
        club_mean_row = club_means.loc[club_means['Club'] == row["Club"]]
        result = float(club_mean_row["Overall"].values)  # Convert to single value
        return result


def normalize(df):
    # Normalize every column
    for key in df.keys():
        column = df[key]
        standard_deviation = np.std(column)
        mean = np.mean(column)
        df[key] = column.apply(lambda val: (val - mean)/standard_deviation)

    return df



if __name__ == '__main__':
    df = pre_process()
    df.to_csv(PROCESSED_OUTPUT_FILE_PATH, index=False)
