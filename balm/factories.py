from tdc.multi_pred import DTI

from balm.datasets import (
    BindingDBDataset,
    LeakyPDBDataset,
    MproDataset,
    USP7Dataset,
    HIF2ADataset,
    MCL1Dataset,
    SYKDataset,
)


DATASET_MAPPING = {
    "LeakyPDB": LeakyPDBDataset,
    "BindingDB_filtered": BindingDBDataset,
    "Mpro": MproDataset,
    "USP7": USP7Dataset,
    "HIF2A": HIF2ADataset,
    "MCL1": MCL1Dataset,
    "SYK": SYKDataset,
}


def get_dataset(dataset_name, harmonize_affinities_mode=None, *args, **kwargs):
    """
    Get dataset for protein-protein interaction studies.

    Args:
        dataset_name (str): Name of the dataset to load.
        harmonize_affinities_mode (str, optional): Mode for harmonizing affinities in DTI datasets.
        *args, **kwargs: Additional arguments passed to dataset classes. Can include fine-tuning specific 
                        parameters when needed.

    Returns:
        Dataset: The loaded dataset with protein sequences ready for model processing 
                (both frozen and fine-tuning approaches).
    """
    if dataset_name.startswith("DTI_"):
        dti_dataset_name = dataset_name.replace("DTI_", "")
        dataset = DTI(name=dti_dataset_name)
        if harmonize_affinities_mode:
            dataset.harmonize_affinities(mode=harmonize_affinities_mode)
            # Convert $K_d$ to $pKd$
            dataset.convert_to_log(form="binding")
    else:
        dataset = DATASET_MAPPING[dataset_name](*args, **kwargs)

    # Ensure protein sequences are strings
    if hasattr(dataset, 'data'):
        dataset.data['Target'] = dataset.data['Target'].astype(str)
        dataset.data['proteina'] = dataset.data['proteina'].astype(str)

    return dataset