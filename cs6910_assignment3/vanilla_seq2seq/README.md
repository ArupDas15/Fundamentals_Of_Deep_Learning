# Vanilla Seq-to-Seq
----------------------------------------------------
 Here we have implemented the Encoder-Decoder model without the attention. We have built stacked encoder-decoder model here.
# Set up and Installation: #
----------------------------------------------------
Easiest way to try this code out is :-
1. Download the .pynb file
2. Upload it to Google Colab
3. Use the "Run All Cells" option to begin training the best model.

Accuracy on test dataset= 34.00%

# Methods #
|   | Method name     | Description                                                                                                          |
|---|-----------------|----------------------------------------------------------------------------------------------------------------------|
| 1 | data            | Prepare the data by padding the output and then tokenizing it .                                                      |
| 2 | build_model     | Create a model without compiling , as required by parameters.                                                        |
| 3 | build_inference | Modifies our trained model to build a model that is capable of decoding given input.                                          |
| 4 | decode_batch    | It is used to decode batch of inputs using inference model                                                           |
| 5 | test_accuracy   | Returns the accuracy of the model using test data. Also generates file containing succesful prediciton and failures.  |
| 6 | batch_validate  | Returns the accuracy of the model using validation data                                                              |
| 7 | train           | Train using configs sent by wandb.                                                                                   |
| 8 | manual_train    | Train using our custom configs.                                                                                      |

# Sequence of method calling #
-------------------------------------------------------
A typical sequence of method calling is shown below. Here we are sequence when we do manual training.

|   | Method name     | Sequence of Call |
|---|-----------------|------------------|
| 1 | data            | 0                |
| 2 | build_model     | 2                |
| 3 | build_inference | 3                |
| 4 | decode_batch    | 4                |
| 5 | test_accuracy   | 6                |
| 6 | batch_validate  | 5                |
| 7 | train           | -                |
| 8 | manual_train    | 1                |

# Training without WandB #
-----------------------------------------------------
As shown below manual training takes in a object of 'configuration' class.

Set wb=False


config=configuration('LSTM',32,512,3,2,.3,20,64)
manual_train(config)

- Argument 1 : Cell Type
- Argument 2 : Embedding Dimension
- Argument 3 : Hidden Units
- Argument 4 : Number of encoder layers
- Argument 5 : Number of decoder layers
- Argument 6 : Dropout
- Argument 7 : Epochs
- Argument 8 : Batch Size

# Training with WandB #
-------------------------------------------------------
Set wb=True

And provide your sweep ID.

# Best Model #
----------------------------------------------------
![model](https://user-images.githubusercontent.com/12824938/119309723-4b516180-bc8c-11eb-9d5e-4a8781bf6e82.png)



# Acknoledgement #
1. Course slides of CS6910 course by Prof. Mithesh Khapra
2. [This](https://keras.io/examples/nlp/lstm_seq2seq/)  blog was very helpful in understanding how Seq-to-Seq models are coded in Keras. 
