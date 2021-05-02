# Building a ChatBot with Deep NLP
 
import numpy as np
import tensorflow as tf
import re
import time
 
########## DATA PREPROCESSING ##########

lines = open('/data/movie_lines.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n')
conversations = open('/data/movie_conversations.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n')




# Create a dictionary that map each line and its id
id2line = {}
for line in lines:
    _line = line.split(' +++$+++ ')
    if len(_line) == 5:
        id2line[_line[0]] = _line[4]
        
# Create list of all conversations
conversations_ids = []
for conversation in conversations[:-1]:
    _conversation = conversation.split(' +++$+++ ')[-1][1:-1].replace("'", "").replace(" ", "")
    conversations_ids.append(_conversation.split(","))
    
# Separate the question from the answers
questions = []
answers = []

for conversation in conversations_ids:
    for i in range(len(conversation) - 1):
        questions.append(id2line[conversation[i]])
        answers.append(id2line[conversation[i+1]])

        

def clean_text(text):
    text = text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"[-()\"#/@;:<>{}+=~|.?,]", "", text)
    return text

# Clean questions and answers
clean_questions = []
for question in questions:
    clean_questions.append(clean_text(question))
    
clean_answers = []
for answer in answers:
    clean_answers.append(clean_text(answer))

# Create dictionary that maps each word to its number of occurances
word2count = {}

for question in clean_questions:
    for word in question.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1
            
for answer in clean_answers:
    for word in answer.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1
            

# Creating two dictionaries that map the questions words and the answers words to a unique integer
# Filtering the non-frequent words

# With these dictionaries, we know which words are not the five percent least frequent word so later
threshold_questions = 20
questionswords2int = {} # map each word in all questions to a unique integer
word_number = 0
for word, count in word2count.items():
    if count >= threshold_questions: # check if the number of occurances of the word is > threshold
        questionswords2int[word] = word_number
        word_number += 1

threshold_answers = 20
answerswords2int = {} # map each word in all answers to a unique integer
word_number = 0
for word, count in word2count.items():
    if count >= threshold_answers: # check if the number of occurances of the word is > threshold
        answerswords2int[word] = word_number
        word_number += 1

# Add tokens to 'questionswords2int' and 'anwerswords2int' dictionaries
# Tokenization
# useful for the encoder and the decoder in the seq2seq model
# Start of string: encode bwith SOS; End of string: encode with EOS; 
# The 'PAD' which is important for the model because the process
# training data and the sequences in the batches should all have the same 
# length and therefore input this token in an empty position.
# OUT: corresponds to all the words that were filtered out by questionswords2int 
# and  answersword2int dictionaries. Use the dictionaries to replace all the five percent less frequent words with this 
# token 'OUT' so that these 5 percent less frequent words are all replaced by one same common token

tokens = ['<PAD>', '<EOS>', '<OUT>', '<SOS>']
for token in tokens:
    questionswords2int[token] = len(questionswords2int) + 1
    
for token in tokens:
    answerswords2int[token] = len(answerswords2int) + 1

# Create the inverse dictionary of the answersword2int dictionary.
# That's because we will need inverse mapping from the integers to the answersword2int in the implementation
# of the seq2seq model.
answersints2word = {w_i: w for w, w_i in answerswords2int.items()} 


# Adding the end of String token to the end of every answer
for i in range(len(clean_answers)):
    clean_answers[i] += ' <EOS>'

# Translating all the questions and the answers into integers and replacing 
# all the words that were filtered out by

# Sort all the questions and all the answers by their length.
# Two lists of questions and answers translated into integers.
#  optimize the training performance.
questions_into_int = []
for question in clean_questions:
    ints = []
    for word in question.split():
        if word not in questionswords2int:
            ints.append(questionswords2int['<OUT>'])
        else:
            ints.append(questionswords2int[word])
    questions_into_int.append(ints)
    
answers_into_int = []
for answer in clean_answers:
    ints = []
    for word in answer.split():
        if word not in answerswords2int:
            ints.append(answerswords2int['<OUT>'])
        else:
            ints.append(answerswords2int[word])
    answers_into_int.append(ints)
    

# Sorting question and answer by the length of questions
sorted_clean_questions = []
sorted_clean_answers = []
for length in range(1, 25 + 1):
    for i in enumerate(questions_into_int):
        if len(i[1]) == length:
            sorted_clean_questions.append(questions_into_int[i[0]])
            sorted_clean_answers.append(answers_into_int[i[0]])



 
 
########## BUILDING THE SEQ2SEQ MODEL ##########

# Create placeholder for the inputs and the target
def model_inputs():
    inputs = tf.placeholder(tf.int32, [None, None], name='input')
    targets = tf.placeholder(tf.int32, [None, None], name='target')
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    
    return inputs, targets, learning_rate, keep_prob

# Preprocessing the target
def preprocess_targets(targets, word2int, batch_size):
    """ 
        prepare a set of targets [answers]  because the 
        decoder will only accept a certain format of the targets.
        take all these answers inside batches and remove the last column of these answers.
        make a concatenation to add the start of string tokens at the beginning of the target in the batches.
    """
    # The targets must be in two batches. 
    # The recurrent neural networks of the decoder 
    # will not accept single targets that is single answers.
    # Each of the answers in the batch of Target must start with the SOS tokens.

    left_side = tf.fill([batch_size, 1], word2int['<SOS>']) # vector of elements only containing 
                                                            # the S.O.S [unique identifiers encoding 
                                                            # the SOS tokens] tokens.
            
    # all the answers in the batch of answers except the last column because we don't
    # want to get the last token (not needed for the decoder)
    right_side = tf.strided_slice(tragets, [0,0], [batch_size, -1], [1,1]) 
    preprocess_targets = tf.concat([left_side, right_side], 1)
    
    return preprocess_targets

#Encoder RNN Layer
def encoder_rnn(rnn_input, rnn_size, num_layers, keep_prob, sequence_length):
    """
    Create the Encoder RNN Layer
    
    Arguments:
            rnn_input: corresponds to the model inputs.
            rnn_size: number of input tensors of the encoder of the RNN layer
            num_layer: number of layers
            keep_prob:  control the dropout rate
            sequence_length: the list of the length of each question in the batch
    """
    
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size) 
    
    # dropout is the technique of dropping out, that is deactivating a certain 
    # rcentage of the neurons during the training iterations.
    lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob) # LSTM with dropout apply
    encoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)
    _, encoder_state = tf.nn.bidirectional_dynamic_rnn(cell_fw = encoder_cell, 
                                                       cell_bw = encoder_cell, 
                                                       sequence_length = sequence_length, 
                                                       input = rnn_input, 
                                                       dtype = tf.float32)
    return tf.contrib.rnn
                                                      
    
    

#Decode the training set   
def decode_training_set(encoder_state, 
                        decoder_cell, 
                        decoder_embedded_input, 
                        sequence_length, 
                        decoding_scope, 
                        output_function, 
                        keep_prob, batch_size):
    """ 
        Decode the training set.
        decode the observations of the training set and returned the output of the decoder.
        Observations of the training set: some observations that are going back into the neural 
        network to update the weights and improve the ability of the chatbot to talk like a human.
    """
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
    attentions_keys, attention_values, attention_score_function, attention_construct_functions = tf.contrib.seq2seq.prepare_attention(attention_states, 
                                                                                                                                      attention_option = 'bahdanau', 
                                                                                                                                      num_units = decoder_cell.output_size)
    
    training_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_train(encoder_state[0], 
                                                                              attentions_keys, 
                                                                              attention_values, 
                                                                              attention_score_function, 
                                                                              attention_construct_functions, 
                                                                              name = "attn_dec_train")
    decoder_output, _, _ = tf.contrib.seq2seq.dynamic_rnn_decorder(decoder_cell, 
                                                                   training_decoder_function, 
                                                                   decoder_embedded_input, 
                                                                   sequence_length, 
                                                                   decoding_scope)
    decoder_output_dropout = tf.nn.dropout(decoder_output, keep_prob)
    return output_function(decoder_output_dropout)


# Decode the test/validation set
def decode_test_set(encoder_state, 
                    decoder_cell, 
                    decoder_embeddings_matrix, 
                    sos_id, 
                    eos_id, 
                    maximum_length, 
                    num_words, 
                    decoding_scope, 
                    output_function, 
                    keep_prob, 
                    batch_size):
     
    """Predict the observations of the test and validation set."""
    
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states, 
                                                                                                                                    attention_option = "bahdanau", 
                                                                                                                                    num_units = decoder_cell.output_size)
    test_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_inference(output_function,
                                                                              encoder_state[0],
                                                                              attention_keys,
                                                                              attention_values,
                                                                              attention_score_function,
                                                                              attention_construct_function,
                                                                              decoder_embeddings_matrix,
                                                                              sos_id,
                                                                              eos_id,
                                                                              maximum_length,
                                                                              num_words,
                                                                              name = "attn_dec_inf")
    test_predictions, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                                    test_decoder_function,
                                                                    scope = decoding_scope)
    return test_predictions


# Create Decoder RNN
def decoder_rnn(decoder_embedded_input, 
                decoder_embeddings_matrix, 
                encoder_state, num_words, 
                sequence_length, rnn_size, 
                num_layers, word2int, 
                keep_prob, 
                batch_size):
    with tf.variable_scope("decoding") as decoding_scope:
        lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
        lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
        decoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)
        weights = tf.truncated_normal_initializer(stddev = 0.1)
        biases = tf.zeros_initializer()
        output_function = lambda x: tf.contrib.layers.fully_connected(x,
                                                                      num_words,
                                                                      None,
                                                                      scope = decoding_scope,
                                                                      weights_initializer = weights,
                                                                      biases_initializer = biases)
        training_predictions = decode_training_set(encoder_state,
                                                   decoder_cell,
                                                   decoder_embedded_input,
                                                   sequence_length,
                                                   decoding_scope,
                                                   output_function,
                                                   keep_prob,
                                                   batch_size)
        decoding_scope.reuse_variables()
        test_predictions = decode_test_set(encoder_state,
                                           decoder_cell,
                                           decoder_embeddings_matrix,
                                           word2int['<SOS>'],
                                           word2int['<EOS>'],
                                           sequence_length - 1,
                                           num_words,
                                           decoding_scope,
                                           output_function,
                                           keep_prob,
                                           batch_size)
    return training_predictions, test_predictions


# Build the seq2seq Model
def seq2seq_model(inputs, 
                  targets, 
                  keep_prob, 
                  batch_size, 
                  sequence_length, 
                  answers_num_words, 
                  questions_num_words, 
                  encoder_embedding_size, 
                  decoder_embedding_size, 
                  rnn_size, num_layers, 
                  questionswords2int):
    encoder_embedded_input = tf.contrib.layers.embed_sequence(inputs,
                                                              answers_num_words + 1,
                                                              encoder_embedding_size,
                                                              initializer = tf.random_uniform_initializer(0, 1))
    encoder_state = encoder_rnn(encoder_embedded_input, 
                                rnn_size, num_layers, 
                                keep_prob, 
                                sequence_length)
    preprocessed_targets = preprocess_targets(targets, questionswords2int, batch_size)
    decoder_embeddings_matrix = tf.Variable(tf.random_uniform([questions_num_words + 1, decoder_embedding_size], 0, 1))
    decoder_embedded_input = tf.nn.embedding_lookup(decoder_embeddings_matrix, preprocessed_targets)
    training_predictions, test_predictions = decoder_rnn(decoder_embedded_input,
                                                         decoder_embeddings_matrix,
                                                         encoder_state,
                                                         questions_num_words,
                                                         sequence_length,
                                                         rnn_size,
                                                         num_layers,
                                                         questionswords2int,
                                                         keep_prob,
                                                         batch_size)
    return training_predictions, test_predictions

 
 
