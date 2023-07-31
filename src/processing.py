import numpy as np

def get_vectors(modelname, lang, size):
    """
    Load word vectors from a trained model file.

    Args:
        modelname (str): Name of the trained model.
        lang (str): Language of the model.
        size (int): Size of training data in sentences.

    Returns:
        dict: Dictionary of word vectors, where each word is a key and the value is a numpy array.
    """

    # Construct the file path based on the modelname, language, and size
    file_name = f"trained_models/{modelname}/{modelname}_{lang}_{size}.txt"

    # Read the file and retrieve the data
    with open(file_name, 'r') as file:
        data = file.readlines()

    # Initialize an empty dictionary to store the word vectors
    vectors = dict()

    # Process each line in the data
    for line in data:
        # Split the line by whitespace
        line_split = line.split()

        # Extract the word (first element) and convert the remaining elements to a numpy array
        word = line_split[0]
        vector = np.array([float(n) for n in line_split[1:]])

        # Store the word vector in the dictionary
        vectors[word] = vector

    return vectors

import pickle

def get_mcu(lang):
    """
    Load "mcus" data based on the language.

    Args:
        lang (str): Language of the model

    Returns:
        tuple: A tuple containing two loaded datasets: mcus by_cat and random_expressions.
    """

    # Define the file paths for the mcu data
    by_categ_file = f'data/mcus/by_category/mcus_{lang}.p'
    random_expr_file = f'data/mcus/random_expressions/random_expressions_{lang}.p'

    # Load data by category
    with open(by_categ_file, 'rb') as handle:
        by_categ = pickle.load(handle)

    # Load the random expressions data
    with open(random_expr_file, 'rb') as handle:
        random_expressions = pickle.load(handle)

    return by_categ, random_expressions