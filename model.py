exec(open("/content/scripts/preprocessing.py").read())



from keras import Model, Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout, Flatten, Input
from keras.utils import to_categorical

#input_shape = (9, 9, 10)

model = Sequential()
model.add(Dense(64, activation='relu', input_shape=input_shape))
model.add(Dropout(0.4))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.4))
model.add(Flatten())

grid = Input(shape=input_shape)  # inputs
features = model(grid)  # commons features

# define one Dense layer for each of the digit we want to predict
digit_placeholders = [
    Dense(9, activation='softmax')(features)
    for i in range(81)
]

solver = Model(grid, digit_placeholders)  # build the whole model
solver.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("model is starting training")

solver.fit(
    delete_digits(Xtrain, 0),  # we don't delete any digit for now
    [ytrain[:, i, j, :] for i in range(9) for j in range(9)],  # each digit of solution
    batch_size=128,
    epochs=1,  # 1 epoch should be enough for the task
    verbose=1,
)

print("model has completed training")

solver.save("model")

print("The Trained model has been saved in model folder")

print("Hola! we Made it!")