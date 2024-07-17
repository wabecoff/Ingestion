import os
import shutil
import yaml
import json
import pandas as pd
import uuid
from kaggle.api.kaggle_api_extended import KaggleApi
from typing import Literal, List, Dict, Union

class KaggleDatasetLoader:
    def __init__(self, data_dir: str = './Data/Text'):
        """
        Initialize the KaggleDatasetLoader with the specified data directory.

        Args:
            data_dir (str): The directory where data will be saved. Defaults to './Data'.
        """
        self.kaggle_dir = os.path.expanduser('~/.kaggle')
        self.kaggle_file = os.path.join(self.kaggle_dir, 'kaggle.json')
        self.has_credentials = False
        self.api = None
        self.private = True
        self.num_docs = 1000  # Number of documents per file

        self.check_kaggle_json()
        if self.has_credentials:
            # Make the kaggle json private
            if self.private:
                self.ensure_private_permissions()

            self.api = KaggleApi()
            self.api.authenticate()

        self.data_dir = data_dir
        self.schema_dir = './Loader/schema'

    def check_kaggle_json(self) -> None:
        """
        Check if the kaggle.json file exists and set the has_credentials flag accordingly.
        """
        if os.path.exists(self.kaggle_file):
            print("kaggle.json file is already in ~/.kaggle")
            self.has_credentials = True
        elif os.path.exists('kaggle.json'):
            print("kaggle.json file found in the working directory, copying to ~/.kaggle")
            self.copy_kaggle_json()
            self.has_credentials = True
        else:
            print("kaggle.json file not found. Please place the kaggle.json file in your working directory.")
            self.has_credentials = False

    def ensure_private_permissions(self) -> None:
        """
        Ensure that the kaggle.json file has private permissions.
        """
        current_permissions = oct(os.stat(self.kaggle_file).st_mode & 0o777)
        desired_permissions = '0o600'

        if current_permissions != desired_permissions:
            os.chmod(self.kaggle_file, 0o600)
            print(f"Set permissions of {self.kaggle_file} to be private")
        else:
            print(f"Permissions of {self.kaggle_file} are currently private")

    def copy_kaggle_json(self, overwrite: bool = False) -> None:
        """
        Copy the kaggle.json file to the .kaggle directory.

        Args:
            overwrite (bool): Whether to overwrite the existing kaggle.json file. Defaults to False.
        """
        source = 'kaggle.json'

        if not os.path.exists(source):
            print("kaggle.json file not found in the working directory. Please provide the file.")
            return

        if not os.path.exists(self.kaggle_dir):
            os.makedirs(self.kaggle_dir)
            print(f"Created directory {self.kaggle_dir}")

        if os.path.exists(self.kaggle_file):
            if overwrite:
                shutil.copyfile(source, self.kaggle_file)
                print(f"Overwritten {self.kaggle_file}")
            else:
                print(f"{self.kaggle_file} already exists. Use overwrite=True to overwrite it.")
        else:
            shutil.copyfile(source, self.kaggle_file)
            print(f"Copied {source} to {self.kaggle_file}")

    def get_dataset(self, url: str) -> None:
        """
        Download and unzip a dataset from Kaggle.

        Args:
            url (str): The URL of the Kaggle dataset.
        """
        url = url.strip()
        args = url.split('/')

        owner, name = args[-2], args[-1]  # Extract owner and dataset name from URL

        output_dir = f'{self.data_dir}/{name}'

        os.makedirs(output_dir, exist_ok=True)

        print(f"Loading dataset into {output_dir}. This may take a few minutes")

        try:
            self.api.dataset_download_files(f'{owner}/{name}', path=output_dir, unzip=True)
        except Exception as e:
            print(f"An error occurred while downloading files: {e}")

    def check_consistency(self, data_pth: str) -> bool:
        """
        Check if all files in the directory are of the same format and at the same directory level.

        Args:
            data_pth (str): The path to the dataset.

        Returns:
            bool: Boolean indicating if the directory is consistent.
        """
        if not os.path.exists(data_pth):
            print(f"Error: Directory {data_pth} does not exist.")
            return False

        file_formats = set()
        file_depths = set()

        for root, dirs, files in os.walk(data_pth):
            for file in files:
                if not file.startswith('.'):  # Ignore hidden files
                    file_extension = os.path.splitext(file)[1].lower()
                    if file_extension:
                        file_formats.add(file_extension)

                if len(file_formats) > 1:
                    print("Error: Multiple file formats found.")
                    print(file_formats)
                    return False

        if len(file_formats) == 0:
            print("Error: No files found in the directory.")
            return False

        return True

    def load_yaml(self, schema_path: str) -> Dict:
        """
        Load a YAML schema file.

        Args:
            schema_path (str): The path to the YAML schema file.

        Returns:
            Dict: The loaded YAML schema as a dictionary.
        """
        with open(schema_path, 'r') as file:
            schema = yaml.safe_load(file)
        return schema

    def read_csv_chunks(self, file_path: str, chunk_size: int = 1000) -> pd.DataFrame:
        """
        Read a CSV file in chunks.

        Args:
            file_path (str): The path to the CSV file.
            chunk_size (int): The number of rows per chunk. Defaults to 1000.

        Returns:
            pd.DataFrame: A pandas DataFrame iterator for the CSV file chunks.
        """
        return pd.read_csv(file_path, chunksize=chunk_size)

    def read_json_file(self, file_path: str, single: bool = True) -> List[Dict]:
        """
        Read a JSON file.

        Args:
            file_path (str): The path to the JSON file.
            single (bool): Whether the file contains a single JSON object or a list of JSON objects. Defaults to True.

        Returns:
            List[Dict]: A list of dictionaries representing the JSON objects.
        """
        with open(file_path, 'r') as file:
            if single:
                return [json.load(file)]
            else:
                return json.load(file)

    def transform_and_save(self, documents: List[Dict], mapping: Dict, output_dir: str, chunk_idx: int) -> None:
        """
        Transform and save documents according to a schema mapping. Adds a unique doc_id if not present.

        Args:
            documents (List[Dict]): The list of documents to transform.
            mapping (Dict): The schema mapping for transformation.
            output_dir (str): The directory to save the transformed documents.
            chunk_idx (int): The index of the current chunk.
        """
        transformed_docs = []
        for doc in documents:
            transformed_doc = {key: doc.get(mapping[key], "") for key in mapping}
            if 'doc_id' not in transformed_doc or not transformed_doc['doc_id']:
                transformed_doc['doc_id'] = str(uuid.uuid4())  # Assign a unique doc_id
            transformed_docs.append(transformed_doc)

        output_path = os.path.join(output_dir, f'standard_{chunk_idx}.csv')
        df = pd.DataFrame(transformed_docs)
        df.to_csv(output_path, index=False)

    def remove_zip_files(self, path: str) -> None:
        """
        Remove all .zip files in the given directory and delete empty directories.

        Args:
            path (str): The directory path to clean up.
        """
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith('.zip'):
                    os.remove(os.path.join(root, file))
                    print(f"Removed .zip file: {os.path.join(root, file)}")

            # Remove empty directories
            if not os.listdir(root):
                os.rmdir(root)
                print(f"Removed empty directory: {root}")

    def format_dataset(self, dir_name: str, schema_name: str, delete: bool = False) -> None:
        """
        Format the dataset according to a schema and optionally delete the original files.

        Args:
            dir_name (str): The name of the directory containing the dataset.
            schema_name (str): The name of the schema file.
            delete (bool): Whether to delete the original files after formatting. Defaults to False.
        """
        data_pth = f'{self.data_dir}/{dir_name}'
        schema_pth = f'{self.schema_dir}/{schema_name}'

        # Remove all .zip files and delete empty directories
        self.remove_zip_files(data_pth)

        if not self.check_consistency(data_pth):
            return

        schema = self.load_yaml(schema_pth)
        file_format = schema['Format'] # in ['csv', 'single_json', 'multi_json']
        mapping = schema['Mapping'] # Dict with the values we want and their name in the dataset

        output_dir = f'{data_pth}-standard'
        os.makedirs(output_dir, exist_ok=True)

        documents = []
        chunk_idx = 0

        for root, _, files in os.walk(data_pth):
            for file in files:
                if (file_format == 'csv' and file.endswith('.csv')) or (file_format in ['single_json', 'multi_json'] and file.endswith('.json')):
                    file_path = os.path.join(root, file)
                    if file_format == 'csv':
                        for chunk in self.read_csv_chunks(file_path):
                            documents.extend(chunk.to_dict(orient='records'))
                            if len(documents) >= self.num_docs:
                                self.transform_and_save(documents[:self.num_docs], mapping, output_dir, chunk_idx)
                                documents = documents[self.num_docs:]
                                chunk_idx += 1
                    elif file_format in ['single_json', 'multi_json']:
                        single = file_format == 'single_json'
                        json_docs = self.read_json_file(file_path, single)
                        documents.extend(json_docs)
                        while len(documents) >= self.num_docs:
                            self.transform_and_save(documents[:self.num_docs], mapping, output_dir, chunk_idx)
                            documents = documents[self.num_docs:]
                            chunk_idx += 1

        # Save any remaining documents
        if documents:
            self.transform_and_save(documents, mapping, output_dir, chunk_idx)

        if delete:
            shutil.rmtree(data_pth)
            print(f"Deleted original directory: {data_pth}")
