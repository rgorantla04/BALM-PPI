import pandas as pd


def format_protein_sequences(dataset: pd.DataFrame):
    """
    Format protein sequences for ESM-2 processing.

    Args:
        dataset (pd.DataFrame): A DataFrame containing the dataset with 'Target' and 'proteina' columns.

    Returns:
        tuple: Two lists containing the formatted protein sequences.
    """
    unique_proteins = dataset["Target"].unique().tolist()
    unique_proteina = dataset["proteina"].unique().tolist()

    # Check if there are non-string values in the unique proteina list
    non_string_proteina = sum([not isinstance(proteina, str) for proteina in unique_proteina])
    if non_string_proteina > 0:
        print("Non-string proteina sequences found:")
        print([proteina for proteina in unique_proteina if not isinstance(proteina, str)])

    # Format sequences for ESM-2 (no special formatting needed, just ensure they're strings)
    formatted_proteins = [str(seq) for seq in unique_proteins]
    formatted_proteina = [str(seq) for seq in unique_proteina]

    return formatted_proteins, formatted_proteina


def process_sequences(examples):
    """
    Process protein sequences for ESM-2.

    Args:
        examples (dict): A dictionary containing the 'Target' and 'proteina' keys for sequences.

    Returns:
        dict: A dictionary with the original sequences formatted for ESM-2.
    """
    return {
        "protein": str(examples["Target"]),
        "proteina": str(examples["proteina"]),
        "protein_ori_sequences": examples["Target"],
        "proteina_ori_sequences": examples["proteina"],
    }