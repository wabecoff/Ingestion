# test_kaggle_dataset_loader.py

import os
import pytest
import shutil
from Loader/.KaggleDatasetLoader import KaggleDatasetLoader

# Create a fixture for the KaggleDatasetLoader instance
@pytest.fixture
def kaggle_loader():
    return KaggleDatasetLoader(data_dir='./test_data', schema_dir='./test_schemas')

# Fixture to create test directories and files
@pytest.fixture(scope='module')
def setup_test_environment():
    os.makedirs('./test_data/example_dataset', exist_ok=True)
    os.makedirs('./test_schemas', exist_ok=True)

    # Create a sample YAML schema file
    schema_content = """
    Format: single_json
    Mapping:
      title: title
      text: text
      url: url
      author: author
    """
    with open('./test_schemas/financial.yaml', 'w') as file:
        file.write(schema_content)

    # Create a sample JSON file
    json_content = """
    {
      "title": "Sample Title",
      "text": "Sample text",
      "url": "http://example.com",
      "author": "Author Name"
    }
    """
    with open('./test_data/example_dataset/sample.json', 'w') as file:
        file.write(json_content)

    yield

    # Teardown test environment
    shutil.rmtree('./test_data')
    shutil.rmtree('./test_schemas')

# Test the check_consistency method
def test_check_consistency(kaggle_loader, setup_test_environment):
    assert kaggle_loader.check_consistency('./test_data/example_dataset', 'single_json')

# Test the remove_zip_files method
def test_remove_zip_files(kaggle_loader, setup_test_environment):
    # Create a .zip file
    with open('./test_data/example_dataset/sample.zip', 'w') as file:
        file.write('This is a test zip file.')

    kaggle_loader.remove_zip_files('./test_data/example_dataset')
    assert not os.path.exists('./test_data/example_dataset/sample.zip')

# Test the load_yaml method
def test_load_yaml(kaggle_loader, setup_test_environment):
    schema = kaggle_loader.load_yaml('./test_schemas/financial.yaml')
    assert schema['Format'] == 'single_json'
    assert schema['Mapping']['title'] == 'title'

# Test the read_json_file method
def test_read_json_file(kaggle_loader, setup_test_environment):
    documents = kaggle_loader.read_json_file('./test_data/example_dataset/sample.json', single=True)
    assert len(documents) == 1
    assert documents[0]['title'] == 'Sample Title'

# Test the transform_and_save method
def test_transform_and_save(kaggle_loader, setup_test_environment):
    documents = [
        {
            "title": "Sample Title",
            "text": "Sample text",
            "url": "http://example.com",
            "author": "Author Name"
        }
    ]
    mapping = {
        "title": "title",
        "text": "text",
        "url": "url",
        "author": "author"
    }
    output_dir = './test_data/example_dataset_transformed'
    os.makedirs(output_dir, exist_ok=True)
    kaggle_loader.transform_and_save(documents, mapping, output_dir, 0)
    assert os.path.exists(os.path.join(output_dir, 'transformed_0.csv'))

# Test the format_dataset method
def test_format_dataset(kaggle_loader, setup_test_environment):
    kaggle_loader.format_dataset('example_dataset', 'financial.yaml', delete=True)
    output_dir = './test_data/example_dataset_transformed'
    assert os.path.exists(output_dir)
    assert os.path.exists(os.path.join(output_dir, 'transformed_0.csv'))
    assert not os.path.exists('./test_data/example_dataset')
