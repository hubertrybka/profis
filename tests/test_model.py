import pandas as pd
import numpy as np
import torch
from profis.utils.finger import encode
from profis.utils.modelinit import initialize_model


def test_encode():

    test_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_model = initialize_model('tests/data/test_RNN_config.ini', test_device)
    test_df = pd.read_parquet('tests/data/test_d2_dataset_KRFP.parquet')

    mus, logvars = encode(test_df, test_model, test_device, batch=2)
    assert mus.shape[0] == test_df.shape[0]
    assert mus.shape[1] == 32  # 32 is the latent space size in the test_RNN_config.ini
    assert logvars.shape[0] == test_df.shape[0]
    assert logvars.shape[1] == 32
    assert np.all(np.isfinite(mus))
    assert np.all(np.isfinite(logvars))

    return

