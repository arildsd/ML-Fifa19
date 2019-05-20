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
    vector_values = process_value(df)

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

    return df


def process_wage(df):
    df['Wage'] = df['Wage'].map(lambda x: parse_values(x))
    return df


def process_value(df):
    df['Value'] = df['Value'].map(lambda x: parse_values(x))
    values_vector = df['Value'].values
    return values_vector


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
    df['Contract Valid Until'] = df['Contract Valid Until'].map(lambda x: remove_date(x))
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
    return df


def get_joined_year(joined_date):
    if (len(str(joined_date)) > 0) and not (str(joined_date) == 'nan'):    # Has valid contract and not on loan
        joined_date_split = str(joined_date).split(', ')
        date = joined_date_split[0]
        year = joined_date_split[1]
        return year



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


if __name__ == '__main__':
    df = pre_process()
    df.to_csv(PROCESSED_OUTPUT_FILE_PATH)
