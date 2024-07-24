# Social Media Sentiment Analysis on Cryptocurrencies User Manual

## Introduction
This ReadMe provides instructions on how to run the data collection and analysis scripts for your final year project. The project consists of two main sections: data collection and data analysis. The data collection section involves scraping posts from Mastodon using Python scripts, while the data analysis section includes sentiment analysis and topic extraction scripts that process the collected data.

## Prerequisites
Before running the scripts, ensure that you have the following:

- Python 3.11 or later installed on your system
- Poetry package manager installed (You can install Poetry by following the instructions at [https://python-poetry.org/docs/#installation](https://python-poetry.org/docs/#installation))

## Setting up the Virtual Environment
Follow these steps to set up the virtual environment using Poetry:

1. Open a terminal or command prompt.
2. Navigate to the directory where the `pyproject.toml` file is located.
3. Run the following command to create a new virtual environment and install the required dependencies:
   ```bash
   poetry install
   ```
   This command will create a virtual environment and install all the dependencies specified in the `pyproject.toml` file.

4. Activate the virtual environment by running the following command:
   ```bash
   poetry shell
   ```
   You should now be in the virtual environment, and your terminal prompt should be prefixed with the name of the virtual environment.

## Data Collection
### Scraping Mastodon Posts
The `mastodonScraper.py` script is responsible for scraping posts from Mastodon based on a list of hashtags. To run the script, follow these steps:

1. Ensure that you are in the virtual environment created in the previous section.
2. Navigate to the directory where the `mastodonScraper.py` script is located.
3. Run the following command:
   ```bash
   python mastodonScraper.py
   ```
   The script will start scraping posts from Mastodon based on the specified hashtags. The scraped posts will be saved as JSON files in the `mastodon_posts_top50` directory.

### Finding Hashtags in Scraped Posts
The `hashTagFinder.py` script is used to find hashtags in the scraped Mastodon posts and calculate their frequencies. To run the script, follow these steps:

1. Ensure that you are in the virtual environment created in the previous section.
2. Navigate to the directory where the `hashTagFinder.py` script is located.
3. Run the following command:
   ```bash
   python hashTagFinder.py
   ```
   The script will process the JSON files in the `mastodon_posts` directory, find the hashtags in each post, and calculate their frequencies. The resulting hashtags and their frequencies will be saved in a CSV file named `hashtags.csv`.

## Data Analysis
### Sentiment Analysis
#### Aggregate Sentiment Over Time
The `AggrSentOverTime.py` script analyses the sentiment of the collected Mastodon posts over time. It calculates a trust score for each post based on engagement metrics and the followers/following ratio, and then aggregates the sentiment scores by date. The script produces a plot of the smoothed and clipped sentiment over time using Plotly.

To run the script, follow these steps:

1. Ensure that you are in the virtual environment created in the data collection section.
2. Navigate to the directory where the `AggrSentOverTime.py` script is located.
3. Run the following command:
   ```bash
   python AggrSentOverTime.py
   ```
   The script will process the JSON files in the `mastodon_posts_top50` directory, calculate the sentiment scores, and plot the aggregated sentiment over time.

#### All Coin Sentiment Graph
The `allCoinSentGraph.py` script analyses the sentiment of the collected Mastodon posts for each cryptocurrency and plots the sentiment alongside the corresponding cryptocurrency price over time.

To run the script, follow these steps:

1. Ensure that you are in the virtual environment created in the data collection section.
2. Navigate to the directory where the `allCoinSentGraph.py` script is located.
3. Run the following command:
   ```bash
   python allCoinSentGraph.py
   ```
   The script will process the JSON files in the `mastodon_posts_top50` directory for each cryptocurrency, calculate the sentiment scores, fetch the corresponding price data, and plot the sentiment and price over time for each cryptocurrency.

### Topic Extraction
#### LDA Processing
The `ldaProcessing.py` script performs topic extraction using Latent Dirichlet Allocation (LDA) on the collected Mastodon posts. It preprocesses the text data, applies LDA to extract topics, and assigns the dominant topic to each post. The script saves the topic assignments and word counts per week to CSV files.

To run the script, follow these steps:

1. Ensure that you are in the virtual environment created in the data collection section.
2. Navigate to the directory where the `ldaProcessing.py` script is located.
3. Run the following command:
   ```bash
   python ldaProcessing.py
   ```
   The script will process the JSON files in the `all_posts` directory, perform LDA topic extraction, and save the topic assignments and word counts to CSV files.

#### Plot LDA
The `plotLda.py` script takes the output from the LDA processing step and plots the popularity of individual words over time using a bump chart.

To run the script, follow these steps:

1. Ensure that you are in the virtual environment created in the data collection section.
2. Navigate to the directory where the `plotLda.py` script is located.
3. Run the following command:
   ```bash
   python plotLda.py
   ```
   The script will read the word counts data from the `word_counts_per_week.csv` file generated by the LDA processing step and plot the popularity of individual words over time.

#### Topic Grid Search (Optional)
The `topicGridSearch.py` script performs a grid search to find the optimal parameters for the LDA topic extraction. It searches over different numbers of topics and learning decay values to find the best combination.

To run the script, follow these steps:

1. Ensure that you are in the virtual environment created in the data collection section.
2. Navigate to the directory where the `topicGridSearch.py` script is located.
3. Run the following command:
   ```bash
   python topicGridSearch.py
   ```
   The script will process the JSON files in the `all_posts` directory, perform a grid search for the LDA parameters, and output the best model's parameters and log-likelihood score.

Note: Running the grid search is optional, as the optimal parameters have already been determined and used in the `ldaProcessing.py` script. However, if you wish to experiment with different parameters, you can use this script.

## Model Training and Inference
This section provides instructions on how to train the GPT-neo-x-20B model on the collected cryptocurrency-related social media datasets and use the trained model for generating responses to questions.

### Training the Model
To train the GPT-neo-x-20B model on the datasets, follow these steps:

1. Ensure that you have the necessary dependencies installed, including PyTorch, Transformers, PEFT, bitsandbytes, and datasets libraries. You can install them using the following command:
   ```bash
   pip install torch transformers peft bitsandbytes datasets
   ```

2. Prepare your dataset by combining the relevant JSON files from the `ExtraDataProcessed` and `MastodonProcessed` directories. Update the file paths in the notebook to point to your dataset directories.

3. Open the `bnb_4bit_training_LoRA.ipynb` notebook in Jupyter Notebook or JupyterLab.

4. Run the notebook cells in order, which will perform the following steps:
   - Load the GPT-neo-x-20B model with the specified configuration and quantisation settings.
   - Prepare the model for training using the PEFT library's `prepare_model_for_kbit_training` function.
   - Load and preprocess the datasets, applying tokenisation and truncation.
   - Set up the training arguments and initialise the Trainer object.
   - Train the model using the `trainer.train()` function.

5. After training, save the fine-tuned model using the following code:
   ```python
   model_to_save = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model
   model_to_save.save_pretrained("path/to/save/model")
   ```
   Replace `"path/to/save/model"` with the desired directory path where you want to save the fine-tuned model.

### Asking Questions and Getting Responses
To ask questions and get responses from the fine-tuned model, follow these steps:

1. Load the fine-tuned model using the PEFT library's `get_peft_model` function:
   ```python
   model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neox-20b", quantization_config=bnb_config, device_map={"":0})
   lora_config = LoraConfig.from_pretrained('path/to/saved/model')
   model = get_peft_model(model, lora_config)
   ```
   Replace `"path/to/saved/model"` with the directory path where you saved the fine-tuned model.

2. Use the model's `generate` function to generate responses to your questions:
   ```python
   question = "Your question here"
   inputs = tokenizer(question, return_tensors="pt").to("cuda")
   outputs = model.generate(**inputs, max_new_tokens=100)
   response = tokenizer.decode(outputs[0], skip_special_tokens=True)
   print(response)
   ```
   Replace `"Your question here"` with your specific question related to cryptocurrencies.

3. The model will generate a response based on the fine-tuned knowledge it acquired during training. The response will be printed in the notebook.

Note: Ensure that you have access to a GPU with sufficient memory to load and run the GPT-neo-x-20B model. The model requires a significant amount of GPU memory, so make sure your system meets the hardware requirements.

By following these steps, you can train the GPT-neo-x-20B model on your cryptocurrency-related social media datasets and use the fine-tuned model to ask questions and generate responses.