import os
import argparse
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import pandas as pd


def add_args(parser: argparse.ArgumentParser):
    """Add dataset-specific arguments (none needed for 3DRealCar)"""
    pass


def get_metadata(output_dir, **kwargs):
    """Load metadata from the output directory"""
    metadata_path = os.path.join(output_dir, 'metadata.csv')
    if not os.path.exists(metadata_path):
        raise ValueError(f'metadata.csv not found in {output_dir}')
    return pd.read_csv(metadata_path)


def download(metadata, output_dir, **kwargs):
    """
    No download needed for 3DRealCar - data is already converted
    Just return the existing metadata
    """
    return metadata[['sha256', 'local_path']]


def foreach_instance(metadata, output_dir, func, max_workers=None, desc='Processing objects') -> pd.DataFrame:
    """
    Process each instance in the metadata
    For rendering: uses the raw .obj file
    For voxelization: uses the rendered mesh.ply
    """
    metadata = metadata.to_dict('records')
    
    records = []
    max_workers = max_workers or os.cpu_count()
    
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor, \
            tqdm(total=len(metadata), desc=desc) as pbar:
            
            def worker(metadatum):
                try:
                    sha256 = metadatum['sha256']
                    local_path = metadatum['local_path']
                    
                    # For rendering: use the raw .obj file
                    # For voxelization: the function itself will look for renders/{sha256}/mesh.ply
                    file = os.path.join(output_dir, local_path)
                    
                    if not os.path.exists(file):
                        print(f"Warning: File not found: {file}")
                        pbar.update()
                        return
                    
                    record = func(file, sha256)
                    if record is not None:
                        records.append(record)
                    pbar.update()
                except Exception as e:
                    print(f"Error processing object {sha256}: {e}")
                    import traceback
                    traceback.print_exc()
                    pbar.update()
            
            executor.map(worker, metadata)
            executor.shutdown(wait=True)
    except Exception as e:
        print(f"Error happened during processing: {e}")
        import traceback
        traceback.print_exc()
    
    return pd.DataFrame.from_records(records)