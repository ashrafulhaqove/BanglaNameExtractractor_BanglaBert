# BanglaNameExtractractor_BanglaBert
## About the project
The target of this project is to extract names from the body of text using a language model. The language model is trained on a Named Entity Recognition dataset. Our target is to use the NER model to tag the named entities and then take only the person name tags to extract the names of the persons in the given text.

## Data Source
As the data source we are using this [dataset](https://raw.githubusercontent.com/banglakit/bengali-ner-data/master/main.jsonl) which is a Jsonline object right. The sentences and the list of named entities arranged in order of the words are paired and stacked as each line. This is an example of one line, 

["আফজালুর রহমান নামের এক পরীক্ষার্থী বলেন, সবার হাতে হাতে প্রশ্ন দেখে তিনি ভেবেছিলেন এটি ভুয়া প্রশ্ন।", ["B-PERSON", "L-PERSON", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O"]]

Note that the tagging format used here is the ‘BIL’ format and the tags are ordered in the order of the words.

## Preprocessing

Our first task was to collect the data from the Jsonline object and convert it into a  dataframe.

```python
import json
import pandas as pd


lines = []
with open(NerPath) as f:
    lines = f.read().splitlines()

line_dicts = [json.loads(line) for line in lines]
df_final = pd.DataFrame(line_dicts)

print(df_final)
```

After doing that we set up a dataframe of 3545 rows of data. Where each row represented a sentence and their NER tags. 

![image](https://github.com/ashrafulhaqove/BanglaNameExtractractor_BanglaBert/assets/30887866/54b8a118-b6a0-462b-af05-06652e529032)


Now the Next task will be to tokenize the sentences and change the format of the dataset which can be fed into a NER model. For Tokenization we are using [BNLP Basic Tokenizer](https://github.com/sagorbrur/bnlp/blob/main/docs/README.md#basic-tokenizer) 
```python
df_final["Tokens"] = df_final[0].apply(lambda x: basic_t.tokenize(x))
```

After tokenization our task is to identify any error done by the tokenizer and fix those errors. For that we compared the number of tokens and the number of tags for each sentence. 
```python
df_final[ (df_final['Tokens'].str.len() !=  df_final[1].str.len()) ]
```

![image](https://github.com/ashrafulhaqove/BanglaNameExtractractor_BanglaBert/assets/30887866/937cabe6-b886-42d8-9745-8de0f7c58bf7)

95 sentences have been wrongly tokenized. After deep inspection we found most of the cases to be related to names which has dot (".") in them and the tokenizer is wrongly indetifying it to be an independent token.
processing these data will be too much effort compared to the impact it would make. so, dropping the 95 mis tokenized. set up a new dataset leaving out the 95 wrongly tokenized sentences.
```python
indexAge = df_final[ (df_final['Tokens'].str.len() !=  df_final[1].str.len()) ].index
df_final_clean = df_final.copy()
df_final_clean.drop(indexAge , inplace=True)
```
Then convert the dataset into a dataset that a NER model can accept as training data. For that we need to explode each tag and tokens into one row for each. Here is the final format of the dataset. 

```python
Model_dataset_df = pd.DataFrame()

Model_dataset_df['Tokens'] = df_final_clean['Tokens'	].explode()

Model_dataset_df['Ner_tag'] = df_final_clean['Ner_tag'	].explode()

Model_dataset_df['#Sentence'] = Model_dataset_df.index

Model_dataset_df.reset_index()

Model_dataset_df.rename(columns={"#Sentence":"sentence_id","Tokens":"words","Ner_tag":"labels"}, inplace =True)

Model_dataset_df["labels"] = Model_dataset_df["labels"].str.upper()

```
![image](https://github.com/ashrafulhaqove/BanglaNameExtractractor_BanglaBert/assets/30887866/09654950-ea50-4906-8f85-b15445dad50e)



## Model
We used simple transformers to make the training and testing part of the project as simple as possible. And as the NER model we took the 'sagorsarker/bangla-bert-base' .

```python
!pip install simpletransformers
```

```python
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
```




## Training
For training we split the dataframe in 80:20 as the train and test split and follow the procedure for training the data. 
```python
X= Model_dataset_df[["sentence_id","words"]]
Y =Model_dataset_df["labels"]

x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size =0.2)

#building up train data and test data
train_data = pd.DataFrame({"sentence_id":x_train["sentence_id"],"words":x_train["words"],"labels":y_train})
test_data = pd.DataFrame({"sentence_id":x_test["sentence_id"],"words":x_test["words"],"labels":y_test})
```
After that we need to make a list of labels that will be used to train the model. 

```python
label = train_data["labels"].unique().tolist()
```
![image](https://github.com/ashrafulhaqove/BanglaNameExtractractor_BanglaBert/assets/30887866/565c0e51-2fa0-4a5d-87b6-01578a1ac59b)

Lastly fixing a set of arguments (Hyperparameters) and training the model for the first time 

```python
args = NERArgs()
args.num_train_epochs = 1
args.learning_rate = 1e-4
args.overwrite_output_dir =True
args.train_batch_size = 32
args.eval_batch_size = 32
```
```python
model = NERModel('bert', 'sagorsarker/bangla-bert-base',labels=label,args =args)
```

```python
model.train_model(train_data,eval_data = test_data,acc=accuracy_score)

```

## Testing

After training we can check the evaluation matrix by this code 
```python
result, model_outputs, preds_list = model.eval_model(test_data)
result
```
![image](https://github.com/ashrafulhaqove/BanglaNameExtractractor_BanglaBert/assets/30887866/24207fd1-dc18-4cb5-af93-3396bbd92fdf)


And we can test out the model by an example 
```python

prediction, model_output = model.predict(["আব্দুর রহিম নামের কাস্টমারকে একশ টাকা বাকি দিলাম"])
prediction
```
![image](https://github.com/ashrafulhaqove/BanglaNameExtractractor_BanglaBert/assets/30887866/e760b4dd-73fa-4b5c-b814-9ff73e31d20c)


We can see that it does identify Named Entities fairly well.

## Hyperparameter Variations
As a short project, not much of erroft has been given in the Hyperparameter optimization. We tested out with 3 different variations of Hyperparameters and selected the best performing one based on the evaluation loss. 

```python
resultId = 1

args = NERArgs()
args.num_train_epochs = 1
args.learning_rate = 1e-4
args.overwrite_output_dir =True
args.train_batch_size = 32
args.eval_batch_size = 32

model1 = NERModel('bert', 'sagorsarker/bangla-bert-base',labels=label,args =args)

args , result =  experimentWithHypParamenters(model1 , args)


resultlist.append({'id' : resultId , 'args' :args , 'result':result})

```
```python
print("result id: " ,resultlist[resultId -1]['id'] , "result : " , resultlist[resultId - 1]['result'] )
```

result id:  1 result :  {'eval_loss': 0.33157099757854847, 'precision': 0.5328596802841918, 'recall': 0.5649717514124294, 'f1_score': 0.5484460694698354}



## Output
The final solution takes a sentence as an argument and returns the names of persons in a list format . 
<img width="638" alt="withName" src="https://github.com/ashrafulhaqove/BanglaNameExtractractor_BanglaBert/assets/30887866/334ea11b-f624-4a61-98a5-37a1e4d3decc">

It can also tell if there is no person's name in the sentence. 


<img width="520" alt="withoutName" src="https://github.com/ashrafulhaqove/BanglaNameExtractractor_BanglaBert/assets/30887866/f43e7a86-ed68-4a41-af54-c488594e7164">



