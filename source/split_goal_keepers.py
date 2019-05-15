import pandas

INPUT_FILE_PATH = r"../data/data.csv"
GK_OUTPUT_FILE_PATH = r"../data/goal_keepers.csv"
OUT_FIELD_PLAYERS_FILE_PATH = r"../data/out_field_players.csv"
# Read the file
df = pandas.read_csv(INPUT_FILE_PATH, sep=";")

is_pos_nan = df["Position"].isna()
is_not_pos_nan = is_pos_nan.apply(lambda x: not x)
df = df[is_not_pos_nan]

# Make two separate data frames, one with GK and one without GK
is_gk = df["Position"] == "GK"
gk_df = df[is_gk]
out_field_df = df[is_gk.apply(lambda x: not x)]

# Write the dfs to files
gk_df.to_csv(GK_OUTPUT_FILE_PATH)
out_field_df.to_csv(OUT_FIELD_PLAYERS_FILE_PATH)

