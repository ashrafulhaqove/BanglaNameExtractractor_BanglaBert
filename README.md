# BanglaNameExtractractor_BanglaBert
## About the project
The target of this project is to extract names from the body of text using a language model. The language model is trained on a Named Entity Recognition dataset. Our target is to use the NER model to tag the named entities and then take only the person name tags to extract the names of the persons in the given text.

## Data Source
As the data source we are using this [dataset](https://raw.githubusercontent.com/banglakit/bengali-ner-data/master/main.jsonl) which is a Jsonline object right. The sentences and the list of named entities arranged in order of the words are paired and stacked as each line. This is an example of one line, 

["আফজালুর রহমান নামের এক পরীক্ষার্থী বলেন, সবার হাতে হাতে প্রশ্ন দেখে তিনি ভেবেছিলেন এটি ভুয়া প্রশ্ন।", ["B-PERSON", "L-PERSON", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O"]]

Note that the tagging format used here is the ‘BIL’ format and the tags are ordered in the order of the words.

## Preprocessing

Our first task was to collect the data from the Jsonline object and convert it into a  dataframe.


After doing that we set up a dataframe of 3545 rows of data. Where each row represented a sentence and their NER tags. 

Now the Next task will be to tokenize the sentences and change the format of the dataset which can be fed into a NER model. For Tokenization we are using [BNLP Basic Tokenizer](https://github.com/sagorbrur/bnlp/blob/main/docs/README.md#basic-tokenizer) 


After tokenization our task is to identify any error done by the tokenizer and fix those errors. For that we compared the number of tokens and the number of tags for each sentence. 


95 sentences have been wrongly tokenized. After deep inspection we found most of the cases to be related to names which has dot (".") in them and the tokenizer is wrongly tagging it to be an independent tag.
processing these data will be too much effort compared to the impact it would make. so, dropping the 95 mis tokenized.

set up a new dataset leaving out the 95 wrongly tokenized sentences.
Then convert the dataset into a dataset that a NER model can accept as training data. For that we need to explode each tag and tokens into one row for each. Here is the final format of the dataset. 



## Model

## Training

## Testing

## Output

