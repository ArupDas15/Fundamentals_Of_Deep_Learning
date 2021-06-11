# Assignment 3: Use recurrent neural networks to build a transliteration system.
----------------------------------------------------
In this [project](https://wandb.ai/miteshk/assignments/reports/Assignment-3--Vmlldzo0NjQwMDc) we implement a model sequence to sequence learning problems using Recurrent Neural Networks, compare different cells such as vanilla RNN, LSTM and GRU, implement attention networks to overcome the limitations of vanilla seq2seq model and visualise the interactions between different components in an RNN based model. We use wandb for hyper parameter configuration using the validation dataset and visualisation of test data. We have performed a large number of experiments to make meaningful inferences and get to our best model.

# Set up and Installation #
----------------------------------------------------
Both vanilla_seq2seq and seq_2_seq_with_attention has been implented in Google Colab.</br>

Visit [here](https://github.com/ArupDas15/Fundamentals_Of_Deep_Learning/tree/master/cs6910_assignment3/vanilla_seq2seq) to know more about vanilla_seq2seq without attention.</br>
Visit [here](https://github.com/ArupDas15/Fundamentals_Of_Deep_Learning/tree/master/cs6910_assignment3/seq2seq_with_attention) to know more about seq2seq with attention network.

# Wandb Report #
Visit [here](https://wandb.ai/utsavdey/seq_to_seq/reports/Assignment-3--Vmlldzo3MTk1Nzk) for the wandb Report and insightful observations made during the training.

# Further Enhancements #
* Evaluate and experiment with our model and code for different datasets. 
* Perform training on a larger dataset.

# **NOTE** 
The code implemented for both [vanilla sequence to sequence](https://github.com/ArupDas15/Fundamentals_Of_Deep_Learning/tree/master/cs6910_assignment3/vanilla_seq2seq/Vanilla_Seq_to_Seq.ipynb) and [sequence to sequence with attention](https://github.com/ArupDas15/Fundamentals_Of_Deep_Learning/tree/master/cs6910_assignment3/seq2seq_with_attention/seq2seq_with_attention.ipynb) can be used on any other indian language from the [Dakshina dataset](https://github.com/google-research-datasets/dakshina) by replacing the hi by the language of your choice while setting the train, dev, test set path from the folder **dakshina_dataset_v1.0/hi/lexicons/**. We have used Hindi for our model.
