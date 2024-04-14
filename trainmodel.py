from function import *
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard
from keras.preprocessing.sequence import pad_sequences

# Define the maximum sequence length and feature dimension
sequence_length = 50
feature_dimension = 63

label_map = {label: num for num, label in enumerate(actions)}

sequences, labels = [], []
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            # Load the array
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)), allow_pickle=True)
            
            # Debugging: Print the shape of the loaded array
            print("Shape of loaded array:", res.shape)
            
            # Check if the loaded array has the expected shape
            if res.shape != (feature_dimension,):  # Ensure expected shape
                print("Skipping array with unexpected shape:", res.shape)
                break
            
            window.append(res)
        
        else:  # Only append if the loop completed without breaking
            sequences.append(window)
            labels.append(label_map[action])

# Check if any valid sequences were found
if not sequences:
    print("No valid sequences found to create labels!")
    exit()

# Padding sequences with zeros to ensure uniform length
X = pad_sequences(sequences, padding='post', dtype='float32')

# Convert labels to categorical
y = to_categorical(labels)

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

# Build the LSTM model
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(X.shape[1], X.shape[2])))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.fit(X_train, y_train, epochs=200, callbacks=[tb_callback])
model.summary()

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

model.save('model.h5')
