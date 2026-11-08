"""Download raw PU bearing archives and convert MATLAB files to parquet."""

import os
import requests
import rarfile
import tqdm
import pandas as pd
from scipy.io import loadmat
import shutil

HEALTHY = ['K001', 'K002', 'K003', 'K004', 'K005', 'K006']
DAMAGED_ARTIFICIAL = ['KA01', 'KA03', 'KA04', 'KA05', 'KA06', 'KA07', 'KA08', 'KA09', 'KI01', 'KI03', 'KI05', 'KI07',
                      'KI08']
DAMAGED_NATURAL = ['KA04', 'KA15', 'KA16', 'KA22', 'KA30', 'KB23', 'KB24', 'KB27', 'KI04', 'KI14', 'KI16', 'KI17',
                   'KI18', 'KI21']
DAMAGED = DAMAGED_ARTIFICIAL + DAMAGED_NATURAL


def _parse_mat(filename):
    data = loadmat(filename, squeeze_me=True)
    keys = [k for k in data.keys() if not k.startswith('__')]
    assert len(keys) == 1
    a = data[keys[0]].item()
    dfs = dict()
    for data in a[1]:
        data = data.item()
        dfs[data[4]] = pd.DataFrame({'ts': data[2]})
    for data in a[2]:
        data = data.item()
        name = data[0]
        vals = data[2]
        df = dfs[data[4]]
        assert len(df) == len(vals)
        df[name] = vals
    assert set(dfs.keys()) == {'Mech_4kHz', 'HostService', 'Temp_1Hz'}
    assert set(dfs['Mech_4kHz'].columns.tolist()) == {'ts', 'force', 'speed', 'torque'}
    assert set(dfs['HostService'].columns.tolist()) == {'ts', 'phase_current_1', 'phase_current_2', 'vibration_1'}
    assert set(dfs['Temp_1Hz'].columns.tolist()) == {'ts', 'temp_2_bearing_module'}
    return dfs


def download_and_parse(dst="./data/PU", retries=3, silent_if_exists=True):
    base = "https://groups.uni-paderborn.de/kat/BearingDataCenter/"
    os.makedirs(dst, exist_ok=True)
    for code in HEALTHY + DAMAGED:
        cur_tmp = os.path.join(dst, 'tmp', code)
        rar_path = os.path.join(cur_tmp, f"{code}.rar")
        raw_path = os.path.join(cur_tmp, 'raw')
        parsed_path = os.path.join(dst, code)

        cnt_retried = 0
        success = False
        while True:
            need_parse = not os.path.exists(os.path.join(parsed_path, 'done'))
            need_unpack = not os.path.isdir(os.path.join(raw_path, code)) and need_parse
            need_download = not os.path.exists(rar_path) and need_unpack

            try:
                if need_download:
                    os.makedirs(cur_tmp, exist_ok=True)
                    url = base + f"{code}.rar"
                    print(f"{code}: downloading {url} ...")
                    r = requests.get(url, stream=True, timeout=5)
                    r.raise_for_status()
                    with open(rar_path, "wb") as f:
                        for chunk in tqdm.tqdm(r.iter_content(64 * 1024)):
                            if chunk:
                                f.write(chunk)
                elif not silent_if_exists:
                    print(f"{code}: {rar_path} already downloaded, skipping download")

                if need_unpack:
                    os.makedirs(raw_path, exist_ok=True)
                    print(f"{code}: unpacking to {raw_path} ...")
                    with rarfile.RarFile(rar_path) as rf:
                        rf.extractall(path=raw_path)

                elif not silent_if_exists:
                    print(f"{code}: {raw_path} already unpacked, skipping unpacking")

                if need_parse:
                    mats_path = os.path.join(raw_path, code)
                    for filename in sorted(os.listdir(mats_path)):
                        if not filename.endswith('.mat'):
                            continue
                        save_path = os.path.join(parsed_path, filename.removesuffix('.mat'))
                        if os.path.isdir(save_path):
                            if not silent_if_exists:
                                print(f"{code}/{filename}: already parsed, skipping")
                            continue
                        print(f"{code}/{filename}: parsing...")
                        try:
                            dfs = _parse_mat(os.path.join(mats_path, filename))
                        except TypeError:
                            print(f"{code}/{filename}: failed to parse, skipping")
                            continue
                        os.makedirs(save_path, exist_ok=True)
                        print(f"{code}/{filename}: saving to {save_path} ...")
                        for key, df in dfs.items():
                            float_cols = df.select_dtypes(include=["float"]).columns
                            df[float_cols] = df[float_cols].astype("float32")
                            df.to_parquet(os.path.join(save_path, f"{key}.parquet"))
                    with open(os.path.join(parsed_path, 'done'), 'w'):
                        pass
                elif not silent_if_exists:
                    print(f"{code}: {parsed_path} already parsed, skipping parsing")

                success = True
                break
            except (requests.exceptions.RequestException, rarfile.Error) as e:
                print(f"{code}: something failed: {e}")
                print(f"{code}: removing all data...")
                cnt_retried += 1
                if os.path.exists(rar_path):
                    os.remove(rar_path)
                if os.path.isdir(raw_path):
                    shutil.rmtree(raw_path)
                if os.path.isdir(parsed_path):
                    shutil.rmtree(parsed_path)

                if cnt_retried > retries:
                    print(f"{code}: No more retries, skipping")
                    break
                else:
                    print(f"{code}: Retrying ({cnt_retried})...")

        if success and need_parse:
            print(f"{code}: done!")
            if os.path.isdir(cur_tmp):
                shutil.rmtree(cur_tmp)

    tmp_path = os.path.join(dst, 'tmp')
    failed_codes = sorted(os.listdir(tmp_path)) if os.path.isdir(tmp_path) else []
    if len(failed_codes) > 0:
        print(f"Failed codes: {failed_codes}")
    elif os.path.isdir(tmp_path):
        os.rmdir(tmp_path)
