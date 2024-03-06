import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def extract_df_list1(pickle_file_path):
  with open(pickle_file_path, 'rb') as file:
    dataframes_list = pickle.load(file)
  return dataframes_list

def get_array_label_list1(dataframes_list, num_features, min_change):
  array_list = []
  labels_list = []
  num_features = num_features
  k = num_features+1
  min_change = min_change
  for df in dataframes_list:
    df = df['Close'].values
    arr = np.zeros((len(df)-k, k-1))
    labels = np.full((len(df)-k, ), '', dtype=object)
    for i in range(len(df)-k):
      arr[i] = df[i:i+k-1]
      if np.abs(df[i+k]-df[i+k-1]) < min_change:
        labels[i] = 'no big change'
      elif df[i+k] > df[i+k-1]:
        labels[i] = 'increase'
      elif df[i+k] < df[i+k-1]:
        labels[i] = 'decrease'
    array_list.append(arr)
    labels_list.append(labels)
  return array_list, labels_list

def encode_label_array1(label_array):
  label_array[label_array=='decrease'] = 0
  label_array[label_array=='no big change'] = 1
  label_array[label_array=='increase'] = 2
  label_encoded = label_array.astype(int).copy()
  return label_encoded

def preprocess_data_via_close_values(pickle_file_path, num_features=32, min_change=1e-5, shuffle=True, split=True, test_size=0.2, random_state=42):
  """
  
  """
  dataframes_list = extract_df_list1(pickle_file_path)
  print("Columns of each dataframe", dataframes_list[0].columns)
  # All values are pre-normalized with respect to their columns
  array_list, labels_list = get_array_label_list1(dataframes_list, num_features, min_change)
  del(dataframes_list)
  print("Shape of each array in the list", array_list[0].shape)
  data_array = np.concatenate(array_list, axis = 0)
  label_array = np.concatenate(labels_list, axis = 0)
  del(array_list)
  del(labels_list)
  print("Label Classes", np.unique(label_array))
  print("Data available for each label - increase, decrease and no big change", sum((label_array=='increase')*1), sum((label_array=='decrease')*1), sum((label_array=='no big change')*1))
  print("Combined array shape - ", data_array.shape)
  print("Combined labels shape - ", label_array.shape)
  label_encoded = encode_label_array1(label_array)
  del(label_array)
  if shuffle:
    inds = list(range(data_array.shape[0]))
    np.random.shuffle(inds)
    data_shuff = data_array[inds, :]
    label_shuff = label_encoded[inds]
    print("Shuffled data shape - ", data_shuff.shape)
    if split:
      x_train, x_test, y_train, y_test = train_test_split(data_shuff, label_shuff, test_size=test_size, random_state=random_state)
      return x_train, x_test, y_train, y_test
    else:
      return data_shuff, label_shuff
  else:
    print("Data Array shape - ", data_shuff.shape)
    if split:
      x_train, x_test, y_train, y_test = train_test_split(data_shuff, label_shuff, test_size=test_size, random_state=random_state)
      return x_train, x_test, y_train, y_test
    else:
      return data_array, label_array


def preprocess_data_equal_division(file_path, split=True, time_steps = 10, num_stocks = 30, le=False, only_close = False, equal_split=True, min_change=1e-5):
  # num_stocks = 30 # The dataset is a list of 2000 stock dataframes. num_stocks is the number of stocks to consider for training (Matter of RAM capacity)
  # time_steps = 10 # Time steps to consider
  df_list = extract_df_list1(file_path)
  k = time_steps
  arr_list = []
  close_values = []
  if not num_stocks:
    num_stocks = len(df_list)
  if not only_close:
      for df in df_list[:num_stocks]:
        for i in range(0, len(df)-k):
          arr = df[['Open', 'High', 'Low', 'Volume']].iloc[i:i+k, :].values
          close_value = df['Close'].loc[i+k] - df['Close'].loc[i+k-1]
          arr_list.append(arr)
          close_values.append(close_value)
  else:
      for df in df_list[:num_stocks]:
        for i in range(0, len(df)-k-1):
          arr = df['Close'].loc[i:i+k].values
          close_value = df['Close'].loc[i+k+1] - df['Close'].loc[i+k]
          arr_list.append(arr)
          close_values.append(close_value)
  labels_list = []
  if equal_split:
    combined_data = list(zip(close_values, arr_list))
    combined_data = sorted(combined_data, key=lambda x: x[0])
    k = len(combined_data)//3
    labels_list = ['decrease']*k + ['no big change']*k + ['increase']*(len(combined_data)-2*k)
    features = [data[1] for data in combined_data]
    features = np.array(features)
  else:
    for i in range(len(close_values)):
      if np.abs(close_values[i]) < min_change:
        labels_list.append('no big change')
      elif close_values[i] < 0:
        labels_list.append('increase')
      elif close_values[i] > 0:
        labels_list.append('decrease')
    features = np.array(arr_list)

  labels_array = np.array(labels_list)
  labels_encoded = np.zeros((len(labels_list), 3))
  labels_encoded[labels_array=='decrease', 0] = 1
  labels_encoded[labels_array=='no big change', 1] = 1
  labels_encoded[labels_array=='increase', 2] = 1
  
  if not le:
    labels = labels_encoded.copy()
  else:
    labels_array[labels_array=='decrease'] = 0
    labels_array[labels_array=='no big change'] = 1
    labels_array[labels_array=='increase'] = 2
    labels = labels_array.astype('int').copy()
  if split:
    x_train, x_valid, y_train, y_valid = train_test_split(features, labels, test_size=0.2, random_state=42)
    return x_train, x_valid, y_train, y_valid
  else:
    return features, labels
    
def metric_calculations(predictions, true_values, str=False, set_ = ""):
  accu = sum((predictions==true_values)*1)/len(true_values)*100
  if not str:
      TP = sum((predictions[predictions==2] == true_values[predictions==2])*1)
      FP = sum((predictions[predictions==2] != true_values[predictions==2])*1)
      TN = sum((predictions[predictions!=2] == true_values[predictions!=2])*1)
      FN = sum((predictions[predictions!=2] != true_values[predictions!=2])*1)
  else:
      TP = sum((predictions[predictions=="increase"] == true_values[predictions=="increase"])*1)
      FP = sum((predictions[predictions=="increase"] != true_values[predictions=="increase"])*1)
      TN = sum((predictions[predictions!="increase"] == true_values[predictions!="increase"])*1)
      FN = sum((predictions[predictions!="increase"] != true_values[predictions!="increase"])*1)
  prec = TP/(TP + FP)*100
  recall = TP/(TP+FN)*100
  specificity = TN/(TN+FP)*100
  F1score = 2*prec*recall/(prec+recall)
  print(f"For {set_}")
  print(f"""\
  Accuracy: {accu},
  Precision: {prec},
  Recall: {recall},
  Specificity: {specificity},
  F1score: {F1score}""")
  return accu, prec, recall, specificity, F1score
  
def metric_calculations_categorical(model, x_, y_true, set_=""):
  y_pred = model.predict(x_)
  y_pred_args = np.argmax(y_pred, axis=1)
  y_true_args = np.argmax(y_true, axis=1)
  return metric_calculations(y_pred_args, y_true_args, set_ = set_)