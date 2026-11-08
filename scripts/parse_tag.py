import re


def parse_tag(family, name, with_seed=False):
    def get_groups(pat):
        prefix = f'(\d+)-' + (r's(\d+)-' if with_seed else '')
        return list(re.fullmatch(prefix + pat, name).groups()[1+with_seed:])

    if family == 'MLP-reconstruction':
        pat = r'dims-3240-(\d+(?:-\d+)*)'
        groups = get_groups(pat)
        return {'dims': list(map(int, groups[0].split('-')))}

    elif family == 'TCN-reconstruction':
        pat = r'sep(\d+)-hid(\d+)-lat(\d+)-bl(\d+)-ker(\d+)'
        groups = get_groups(pat)
        return {
            'separable': bool(int(groups[0])),
            'hidden_channels': int(groups[1]),
            'latent_channels': int(groups[2]),
            'num_blocks': int(groups[3]),
            'kernel_size': int(groups[4]),
        }

    elif family == 'Conv-reconstruction':
        pat = r'hid(\d+(?:\.\d+)*)-lat(\d+)-ker(\d+)'
        groups = get_groups(pat)
        return {
            'hidden_channels': list(map(int, groups[0].split('.'))),
            'latent_channels': int(groups[1]),
            'kernel_size': int(groups[2]),
        }

    elif family == 'GRU-repeated-reconstruction':
        pat = r'lay(\d+)-hid(\d+)-lat(\d+)'
        groups = get_groups(pat)
        return {
            'num_layers': int(groups[0]),
            'hidden_size': int(groups[1]),
            'latent_size': int(groups[2]),
        }


    elif family == 'GRU-seq2seq-reconstruction':
        pat = r'lay(\d+)-hid(\d+)-rev(\d+)-teach(\d+\.\d+)'
        groups = get_groups(pat)
        return {
            'num_layers': int(groups[0]),
            'hidden_size': int(groups[1]),
            'reverse_target': bool(int(groups[2])),
            'teacher_forcing_ratio': float(groups[3]),
        }

    elif family == 'Transformer-reconstruction':
        pat = r'lay(\d+)-mod(\d+)-lat(\d+)-ff(\d+)'
        groups = get_groups(pat)
        return {
            'num_layers': int(groups[0]),
            'd_model': int(groups[1]),
            'latent_dim': int(groups[2]),
            'dim_feedforward': int(groups[3]),
        }
    
    elif family == 'PCA-reconstruction':
        pat = r'k(\d+)'
        groups = get_groups(pat)
        return {
            'n_components': int(groups[0]),
        }
    
    elif family == 'MLP-forecasting':
        pat = r'h(\d+)-hid(\d+(?:\.\d+)*)'
        groups = get_groups(pat)
        return {
            'horizon_size': int(groups[0]),
            'hidden_dims': list(map(int, groups[1].split('.'))),
        }
    
    elif family == 'TCN-forecasting':
        pat = r'h(\d+)-sep(\d+)-hid(\d+)-bl(\d+)-ker(\d+)-fs(\d+)'
        groups = get_groups(pat)
        return {
            'horizon_size': int(groups[0]),
            'separable': bool(int(groups[1])),
            'hidden_channels': int(groups[2]),
            'num_blocks': int(groups[3]),
            'kernel_size': int(groups[4]),
            'final_steps': int(groups[5]),
        }
    
    elif family == "TCN-light-forecasting":
        pat = r'h(\d+)-sep(\d+)-hid(\d+)-bl(\d+)-ker(\d+)-fs(\d+)-tb(\d+)-mix(\d+)'
        groups = get_groups(pat)
        return {
            'horizon_size': int(groups[0]),
            'separable': bool(int(groups[1])),
            'hidden_channels': int(groups[2]),
            'num_blocks': int(groups[3]),
            'kernel_size': int(groups[4]),
            'final_steps': int(groups[5]),
            'head_temporal_bases': int(groups[6]),
            'head_mixer_channels': int(groups[7])
        }

    elif family == 'GRU-seq2seq-forecasting':
        pat = r'hor(\d+)-lay(\d+)-hid(\d+)-rev0-teach(\d+\.\d+)'
        groups = get_groups(pat)
        return {
            'horizon_size': int(groups[0]),
            'num_layers': int(groups[1]),
            'hidden_size': int(groups[2]),
            'teacher_forcing_ratio': float(groups[3]),
        }

    else:
        raise ValueError(f"unknown family: {family}")
    
