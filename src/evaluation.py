import pickle
import numpy as np

def cos_sim(expressions, vectors, lang):
    # Load expressions split data based on language
    all_expressions_split = pickle.load(open(f'data/mcus/all_expressions_split.p', 'rb'))[lang]
    
    # Filter expressions based on available data and create a dictionary
    all_expressions_split = {a[0]: a[1:] for a in all_expressions_split if a[0] in expressions}
    
    # Initialize lists and dictionaries
    all_cos_sim = []
    cos_sim_dict = {}
    
    # Iterate over expressions
    for expression in expressions:
        # Process the expression by replacing spaces and hyphens with underscores
        processed_expression = expression.replace(' ', '_').replace('-', '_').replace('-', '_')
        
        # Retrieve the split words for the expression or use processed expression as words
        words = all_expressions_split.get(expression, list(processed_expression.split('_')))
        
        # Retrieve vectors for words or use 0 if not available
        w_vectors = [vectors.get(w, 0) for w in words]
        
        # Calculate the sum of word vectors
        w_sum = np.sum(w_vectors, axis=0)
        
        # Calculate cosine similarity if the processed expression vector is available
        if processed_expression in vectors:
            cos_sim = np.dot(vectors[processed_expression], w_sum) / (np.linalg.norm(vectors[processed_expression]) * np.linalg.norm(w_sum))
            all_cos_sim.append(cos_sim)
        else:
            cos_sim = 0
        
        # Store the cosine similarity in the dictionary with the original expression
        cos_sim_dict[expression] = cos_sim
    
    # Return the list of cosine similarities and the dictionary
    return all_cos_sim, cos_sim_dict
