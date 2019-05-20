import pandas

INPUT_FILE_PATH = r"../data/data.csv"


def pre_process():
    # Read the file
    df = pandas.read_csv(INPUT_FILE_PATH, sep=";")

    # Parse wage
    df = parse_wage(df)

    # Convert height to meter
    df = convert_height(df)

    # Convert weight to kgs
    df = convert_weight(df)

    # Parse values and add them for each position
    df = process_positions(df)

    # Remove date for loaned players
    df = process_loaned_players(df)

    return df


def parse_wage(df):
    df['Wage'] = df['Wage'].map(lambda x: x.lstrip('£$€â‚¬').rstrip('kKmM'))
    return df


def convert_height(df):
    df['Height'] = df['Height'].map(lambda x: feet_to_meters(x))

    return df


def feet_to_meters(height):
    split_string = height.split("'")
    feet = split_string[0]
    inches = split_string[1]
    height_in_meters = 0.3048 * int(feet) + 0.0254 * int(inches)
    return height_in_meters


def convert_weight(df):
    df['Weight'] = df['Weight'].map(lambda x: lbs_to_kg(x))
    return df


def lbs_to_kg(weight):
    weight_string = weight.rstrip('lbsLBS')
    weight_in_kgs = 0.45359237 * int(weight_string)
    return weight_in_kgs


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
    values = position.split('+')
    total_position = int(values[0]) + int(values[1])
    return total_position


def process_loaned_players(df):
    df['Contract Valid Until'] = df['Contract Valid Until'].map(lambda x: remove_date(x))
    return df


def remove_date(contract):
    # Has date in their contract
    if len(contract) > 4:
        contract_string = contract.split(', ')
        year = contract_string[1]
        return year
    return contract
