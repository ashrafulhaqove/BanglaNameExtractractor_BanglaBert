# BanglaNameExtractractor_BanglaBert
## About the project
The target of this project is to extract names from the body of text using a language model. The language model is trained on a Named Entity Recognition dataset. Our target is to use the NER model to tag the named entities and then take only the person name tags to extract the names of the persons in the given text.

## Data Source
As the data source we are using this [dataset](raw.githubusercontent.com/banglakit/bengali-ner-data/master/main.jsonl) which is a Jsonline object right. The sentences and the list of named entities arranged in order of the words are paired and stacked as each line. This is an example of one line, 

["আফজালুর রহমান নামের এক পরীক্ষার্থী বলেন, সবার হাতে হাতে প্রশ্ন দেখে তিনি ভেবেছিলেন এটি ভুয়া প্রশ্ন।", ["B-PERSON", "L-PERSON", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O"]]

## Preprocessing

## Model

## Training

## Testing

## Output

