
# Text Ingestion Pipeline Case Study



## Dirs

1. **Loader**
   - Loader deals with Kaggle permissions and api. To use the kaggle api, you need a kaggle.json file. A kaggle.json file has been provided.  Please note that initializing the KaggleDatasetLoader class will copy the kaggle.json file into your kaggle directory. Once a loader is initialized, use loader.get_dataset(url) to load in a kaggle dataset.  url can be in the format 'user/dataset' or the complete url 'https://kaggle.com/datasets/user/dataset', either is fine. This will load in all the data files.

   - Loader also deals with converting kaggle datasets into a standard format.  Since the transformation needed depends on the dataset, we use a custom yaml config file for each dataset. The relevant yamls for the football and financial datasets can be found in Loader/schema. When you call loader.format_dataset(dataset, schema) - just include the downloaded dataset dir name and the schema file name - the loader knows at which path to find these. While we can control while columns to copy data from and what to name the columns after the transform by changing the yaml, I will note that the loader can only transform csv and json.  In order to ingest other dataset formats loader class would need to be expanded.  format_dataset will create csv files in the specified format with loader.num_docs documents per csv.  If called with delete = True, format_dataset will delete the pre-transform data.

   - Refer to Loading Data Example.ipynb for usage.

2. **Chunker**
   - Chunker class will break long strings either based off of number of tokens or by sentences using nltk sentence tokenization. Max tokens is still enforced for sentence chunking. Length of chunks should depend on what your embedding model permits and how you plan to use embeddings. chunker.chunk_text(text) splits a string into a list of string chunks.

3. **Embedding Model**
   - Wrapper class for Huggingface SentenceTransformers. embeddingmodel.embed(passage_list) embeds an arbitrarily long list of strings into an (n, d) np array. 'all-mpnet-base-v2' is used as the default model, but you can find the model appropriate for your downstream task at https://huggingface.co/sentence-transformers or at https://huggingface.co/models?library=sentence-transformers

4. **TextDataset**
   - Includes the Document, Chunk, and TextDataset classes.  TextDataset ingests data in the format outputed by loader.format_dataset().
   - Functions for initializing, embedding, saving, and loading TextDatasets are found in TextDataset/utils.py

5. **App**
   - VSS class takes in a TextDataset an embedding model and TextDataset chunk embeddings to initialize a basic top_k vector similarity search program.
   - Barebones implementation but it demonstrates how the ingestion pipeline can be easily used to make a simple NLP system.
   - Refer to Ingestion Application Example.ipynb for usage of VSS and classes 2-5 more generally.

6.  **tests**
   - pytest based testing harness for chunker, embedding model, TextDataset.

## Setup and Usage

To use these scripts, ensure you have the necessary dependencies installed. You can set up your environment by running:

```sh
conda env create -f environment.yml
conda activate text-processing-env
```
To run notebooks with this environment, run
```sh
python -m ipykernel install --user --name=text-processing-env --display-name "Python (text-processing-env)"

```
