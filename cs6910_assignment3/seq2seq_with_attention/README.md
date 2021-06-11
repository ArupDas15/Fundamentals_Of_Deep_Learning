# Assignment 3: Use recurrent neural networks with attention mechanism to build a transliteration system.
----------------------------------------------------
In this project we implement attention networks to overcome the limitations of vanilla seq2seq model and visualise the interactions between different components in an RNN based model. We use wandb for hyper parameter configuration using the validation dataset and visualisation of test data. We have performed a large number of experiments to make meaningful inferences and get to our best model.

Here we use a single layered encoder and a single layered decoder and add an attention network to this basic sequence to sequence model and train the model. 

# Libraries Used: #
----------------------------------------------------
1. wandb to obtain the best model using the hyperparameter configurations.
2. matplotlib libraries were used for plotting the confusion matrix.
3. Keras and tensorflow 
4. numpy to convert the attention weights and loss values from tensor objects into numpy objects.
5. os for file operations
6. io for file operations
7. time to measure the time taken for every epoch.
8. random to select 10 random test inputs for attention heatmap generation
9. shutil to force delete a folder
10. Ipython for visualisation

# **NOTE:** 
The Hindi font file for displaying Hindi characters in the matplotlib plots [here](https://drive.google.com/file/d/11B4BahRBIujMr_jhsw_uXbxN9LF5CHaX/view?usp=sharing). A copy of the same has been upload in the [GitHub project repository](https://github.com/utsavdey/cs6910_assignment3/blob/main/seq2seq_with_attention/Nirmala.ttf). *Kindly upload the same before generating the heatmaps.* 

To run the code using wandb simply **Uncomment** *wandb.agent()* and comment the call to train(). 

Set ```use_wandb=False``` while calling train function if you do not desire to log results into wandb.

To execute the program in Google Colab one can click on the run cell option.

# Program Flow #

* At first the program downloads the [dataset](https://storage.googleapis.com/gresearch/dakshina/dakshina_dataset_v1.0.tar) and sets the train, validation and test filepath(train_file_path, vaildation_file_path, test_file_path) to the standard train, dev, test set from the folder **dakshina_dataset_v1.0/hi/lexicons/**. 
* Create Input word tensors, target word tensors, input vocabulary and output vocabulary from training set.
   * Every target word and input word(translitereated roman word) is appended with a **Start of Sequence Character** ***'\t'*** and **End of Sequence Character** ***'\n'***.
   * We create a dictionary of all the unique characters in the training dataset for both the input language(ENGLISH: inp_lang) and the target language(HINDI: targ_lang).
   * The pairs [HINDI, ENGLISH] of target word and input word(translitereated roman word) are fed into target_tensor_train and input_tensor_train respectively.
 * Begin [training](https://github.com/utsavdey/cs6910_assignment3/blob/main/seq2seq_with_attention/README.md#train).
   * Log the train loss and [validation accuracy](https://github.com/utsavdey/cs6910_assignment3/blob/main/seq2seq_with_attention/README.md#validate) if [wandb is being used](https://github.com/utsavdey/cs6910_assignment3/blob/main/seq2seq_with_attention/README.md#note).
 * Report the [test accuracy](https://github.com/utsavdey/cs6910_assignment3/blob/main/seq2seq_with_attention/README.md#validate) after the completion of the model training.
 * [Generate](https://github.com/utsavdey/cs6910_assignment3/blob/main/seq2seq_with_attention/README.md#generate_inputs) random number(=**n_test_samples**) of test inputs to evaluate the model performance.
   * [Predict](https://github.com/utsavdey/cs6910_assignment3/blob/main/seq2seq_with_attention/README.md#transliterate) the target word for each of the test inputs.
   * Generate the [attention heatmap](https://github.com/utsavdey/cs6910_assignment3/blob/main/seq2seq_with_attention/README.md#plot_attention). This shows which segments of the input word(ENGLISH) takes the model's attention while predicting the target word(HINDI).
   * [Statically visualise](https://github.com/utsavdey/cs6910_assignment3/blob/main/seq2seq_with_attention/README.md#visualize) the RNN Type ('RNN', 'LSTM', 'GRU') activations.
   * Perform [connectivity visualisation](https://github.com/utsavdey/cs6910_assignment3/blob/main/seq2seq_with_attention/README.md#connectivity) to understand which is the input(ENGLISH) character that the sequence to sequence model is looking at while decoding the i<sup>th</sup> character in the output(a HINDI character).
 
# Sequence to Sequence Model Construction #
## train() ##
Currently, the hyperparameters have been set to the [best configurations](https://github.com/utsavdey/cs6910_assignment3/blob/main/seq2seq_with_attention/README.md#best-model-configurations) we obtained during our experiments.

Once the model taining is complete we find the validation accuracy and report the test accuracy and create a folder ***prediction_attention*** with a sub-folder having the name of the hyperparameter configuration(= **run_name**) where we create two files  `success.txt` and `failure.txt`. These files contain `<input word><space><target word><space><predicted word>` of the successful and failed predictions made by the sequence to sequence to sequence.

We randomly choose 10 inputs from our test data and generate a 3*3 attention heatmaps on the same.

We then perform a [connectivity visualisation](https://distill.pub/2019/memorization-in-rnns/#appendix-autocomplete) for the predictions made by our model. Currently, the connectivity visualisation can be performed on any three romanised words. For example: here we have chosen `doctor`, `prayogshala` and `angarakshak` for the purpose of visualisation.


| Parameters | Description                                                                                                                                                                |
|------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| use_wandb  | Accepts a boolean True or boolean False. Deafult value set to True. Set ```use_wandb=False``` while calling train function if you do not desire to log results into wandb. |

Example:
* To use wandb: wandb.agent("<sweep_id>",train) (sample sweep id: "utsavdey/attention-seq-to-seq/w8pglu2i")
* To train without wandb: train(use_wandb=False)

![image](https://user-images.githubusercontent.com/37553488/119320997-ccfbbc00-bc99-11eb-9985-6f9c81af238b.png)


## inference_model() ##
The inference_model() is similar to the train_every_step(). Here the only difference is that we don't use [teacher forcing](https://machinelearningmastery.com/teacher-forcing-for-recurrent-neural-networks/). 

*   The decoder input at every time step is the prediction that it had made previously along with the hidden state and the encoder output. 
*   We stop predicting the target word when the model predicts the end token '\n'.
*   The attention weights obtained at every time step is stored and returned so that it can be used for heatmap generation, visualisation and connectivity exploration.

It takes a transliterated romanized word as input and produces the corresponding indic language word( here 'Hindi').

| Parameters | Description                                                                                                           |
|------------|-----------------------------------------------------------------------------------------------------------------------|
| input_word | Accepts string as input. Here we pass the transliterated roman word.                                                  |
| rnn_type   | Accepts string as input. Here we pass the type of RNN being used. The acceptable values are 'RNN', 'LSTM', and 'GRU'. |


## validate() ##
`validate(path_to_file,folder_name)`: Finds the accuracy of the model on the test and validation dataset. 

It creates a folder ***prediction_attention*** with a sub-folder folder_name where we create two files  `success.txt` and `failure.txt`. These files contain `<input word><space><target word><space><predicted word>` of the successful and failed predictions made by the sequence to sequence to sequence.

| Parameters   | Description                                                                                                                                                        |
|--------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| path_to_file | Accepts parameters of type string. Contains the path to validation or test dataset. from from the folder dakshina_dataset_v1.0/hi/lexicons/                        |
| folder_name  | Accepts parameters of type string. This parameter is helpful in creating subfolder inside the prediction_attention folder as per the hyperparamter configurations. |

## visualize() ##
`visualize()`: Helps to visualise what the sequence to sequence model learns with the help of attention network.
| Parameters  | Description                                                                                                                                                                                                |
|-------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| input_word  | Accepts string as input. Here we pass the transliterated roman word                                                                                                                                        |
| output_word | Accepts string as input. Here we pass the predicted output word.                                                                                                                                           |
| att_w       | Takes a list of list where each sublist denotes the attention weights learnt at a particular timestep. Each of the sublist of size equal to length of the transliterated roman word which is fed as input. |

**Example:**</br>
![Output_without_heatmap](https://user-images.githubusercontent.com/37553488/119277211-e53cee00-bc3b-11eb-9309-1fcf59ae18d0.png)

## connectivity() ##
`connectivity()`: Helps to visualise what the sequence to sequence model learns with the help of attention network.

| Parameters | Description                                                                                                                                                       |
|------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| input_word | Accepts string as input. Here we pass the transliterated roman word                                                                                               |
| rnn_type   | Accepts string as input. Here we pass the type of RNN being used. The acceptable values are 'RNN', 'LSTM', and 'GRU'.                                             |
| file_path  | Accepts a string as input. Here we pass the file location where we want to store the connectivity visualisation. The visualisation is stored with the file name: [connectivity.html](https://github.com/utsavdey/cs6910_assignment3/blob/main/seq2seq_with_attention/connectivity.html)   |

## create_file() ##
`create_file()` is used to create and store the connectivity.html file in the specified  location.

| Parameters  | Description                                                                                                                                                       |
|-------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| input_word  | Accepts string as input. Here we pass the transliterated roman word                                                                                               |
| text_colors | List of list where each sublist denotes the color to be given to every input character on mouse hover action on an output character.                              |
| rnn_type    | Accepts string as input. Here we pass the type of RNN being used. The acceptable values are 'RNN', 'LSTM', and 'GRU'.                                             |
| file_path   | Accepts a string as input. Here we pass the file location where we want to store the connectivity visualisation. The visualisation is stored with the file name: [connectivity.html](https://github.com/utsavdey/cs6910_assignment3/blob/main/seq2seq_with_attention/connectivity.html) |

## get_shade_color() ##
`get_shade_color()`: Returns a specific colour depending the value passed to it.  

| Parameters | Description                                                                                                                                                       |
|------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| value      | Expected a floating number between 0-1 denoting the attention weight of the j<sup>th</sup> input character while predicting the i<sup>th</sup>output character.  |

## transliterate() ##
`transliterate()`: Finds the predicted target word for a given tansliterated roman word, plots the attention heatmap and visualises the LSTM activations if the visual_flag is set to True.
| Parameters  | Description                                                                                                                                                                                                                                                                     |
|-------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| input_word  | Accepts string as input. Here we pass the transliterated roman word                                                                                                                                                                                                             |
| rnn_type    | Accepts string as input. Here we pass the type of RNN being used. The acceptable values are 'RNN', 'LSTM', and 'GRU'.                                                                                                                                                           |
| file_path   | Accepts a string as input. Here we pass the file location where we want to store the attention heatmap for the input word and the predicted target word. If not specified the attention heatmaps is stored in the current working directory by the name "attention_heatmap.png" |
| visual_flag | Accepts a boolean True or boolean False. If the visual_flag is set to true then the code to statically visualise the LSTM activations are called. Default value of the flag is set to "True".                                                                                   |


## generate_inputs() ##
`generate_inputs()`: Randomly chooses 10 inputs from the test dataset and calles the transliteration() to produce the predicted target input and heatmaps. It also set the visual_flag in transliteration() to True only for the first test input and False for the rest 9 test inputs.</br>
| Parameters     | Description                                                                                                                        |
|----------------|------------------------------------------------------------------------------------------------------------------------------------|
| rnn_type       | Accepts string as input. Here we pass the type of RNN being used. The acceptable values are 'RNN', 'LSTM', and 'GRU'.              |
| n_test_samples | Accepts an integer as input. Here we pass number of test inputs to be used for the heatmap generation. Default value is set to 10. |


## plot_attention() ##
`plot_attention()`: Responsible for the attention heatmap creation.
| Parameters     | Description                                                                                       |
|----------------|---------------------------------------------------------------------------------------------------|
| attention      | Accepts a list of attention weights learnt during the model inference for the test input.         |
| input_word     | Accepts string as input. Here we pass the input word (English word).                              |
| predicted_word | Accepts a string as input. Here we pass the prediction made by inference model.                   |
| file_name      | Accepts a string as input. Here we pass the path of the file where we want to store the heatmap.  |

Below is a sample heatmap generated by the code:</br>
![nazarandaaz](https://user-images.githubusercontent.com/37553488/119312982-a1280880-bc90-11eb-8297-a81067faeab4.png)


## How to debug the code? ##

Reduce the number of word pairs being generated by create_dataset() to some small value say 100. However reduce the batch size to a value less than 100 else you may encounter [StopIteration() error](https://stackoverflow.com/questions/48709839/stopiteration-generator-output-nextoutput-generator).

Also, to reduce the validation datasize reduce the number of inputs in validate() to some small number like 10 instead of len(input_words). 

Run the train() for 1 epoch. To visualise the code for 20 epochs you can restore the model parameters as provided in the [github repository](https://github.com/utsavdey/cs6910_assignment3/blob/main/seq2seq_with_attention/training_checkpoints.zip) using `checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))` after runnning the train function for one epoch.

## Best Model Configurations ##
Several experiments were performed to smartly and strategically analyse the hyperparameter configurations by logging the eprenmental results to wandb. To know a summary of the experiments performed visit [here](https://docs.google.com/spreadsheets/d/16FtG5UmiUgYG-q7mFtxHi9EzqxGpyij9UQmZE_5U9lc/edit?usp=sharing).</br>
The details of the best model configurations is given below:
1. Cell Type = 'LSTM'
2. Number of encoder/decoder layers = 1
3. Dropout = 0.2
4. Latent dimensions =  1024
5. Batch size = 64
6. Embedded dimension = 512
7. Epochs = 20
</br>The accuracy obtained on the test set is 40.26% and on validation set is 41.93%. To be noted here is that the predicted output is considered correct only if it exactly matches the reference output(actual target output).
# Acknowledgements #
1. The entire project has been developed from the lecture slides of Dr. Mitesh Khapra, Indian Institute of Technology Madras: http://cse.iitm.ac.in/~miteshk/CS6910.html#schedule
2. https://wandb.ai
3. https://github.com/
4. [Dakshina dataset](https://github.com/google-research-datasets/dakshina)
5. https://stackoverflow.com/questions/44526794/matplotlib-pyplot-labels-not-displaying-hindi-text-in-labels
6. CHARACTER-LEVEL LANGUAGE MODELING: https://arxiv.org/pdf/1506.02078.pdf
7. Write your own custom attention layer: https://towardsdatascience.com/create-your-own-custom-attention-layer-understand-all-flavours-2201b5e8be9e
8. https://keras.io/api/layers/recurrent_layers/lstm/
9. Dzmitry Bahdanau and Kyunghyun Cho and Yoshua Bengio, Neural Machine Translation by Jointly Learning to Align and Translate: https://arxiv.org/pdf/1409.0473.pdf
10. Teacher forcing: https://machinelearningmastery.com/teacher-forcing-for-recurrent-neural-networks/
11. [Intution behind attention](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)
12. [Intuition & Use-Cases of Embeddings in NLP & beyond](https://www.youtube.com/watch?v=4-QoMdSqG_I)
13. Additonal resources on attention. [1](https://stackoverflow.com/questions/62948332/how-to-add-attention-layer-to-a-bi-lstm/62949137#62949137), [2](https://androidkt.com/text-classification-using-attention-mechanism-in-keras/), [3](https://stackoverflow.com/questions/56946995/how-to-build-a-attention-model-with-keras), [4](https://datascience.stackexchange.com/questions/76444/how-can-i-build-a-self-attention-model-with-tf-keras-layers-attention), [5](https://github.com/spro/practical-pytorch/blob/master/seq2seq-translation/seq2seq-translation.ipynb)
14. [Connectivity visualisation](https://distill.pub/2019/memorization-in-rnns/#appendix-autocomplete)
15. [Statically visualise LSTM Activations in Keras](https://towardsdatascience.com/visualising-lstm-activations-in-keras-b50206da96ff).
