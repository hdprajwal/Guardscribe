import pandas as pd


def make_dataset():

    data = [
        "./data/toxic_spans/tsd_test.csv",
    ]

    # Read the CSV files into DataFrames
    dfs = [pd.read_csv(file) for file in data]

    # Concatenate the DataFrames into a single DataFrame
    df = pd.concat(dfs, ignore_index=True)

    # Drop the 'Unnamed: 0' column if it exists
    if "Unnamed: 0" in df.columns:
        df.drop(columns=["Unnamed: 0"], inplace=True)

    def extract_spans(spans):
        # Convert the string representation of the list to an actual list
        if isinstance(spans, str):
            if len(spans) > 2:
                return 1
            else:
                return 0
        if isinstance(spans, list):
            return 0

    df["label"] = df["spans"].apply(
        lambda span: extract_spans(span)
    )

    df.drop(columns=["spans"], inplace=True)

    # Save the DataFrame to a new CSV file

    df.to_csv("./data/combine_toxic.csv", index=False)


if __name__ == "__main__":
    make_dataset()
