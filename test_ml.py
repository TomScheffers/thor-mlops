import pyarrow as pa 
from thor_mlops.clean import ThorTableCleaner
from thor_mlops.ops import head

# Training data
t1 = pa.Table.from_pydict({
    'Animal': ['Falcon', 'Falcon', 'Parrot', 'Parrot', 'Parrot'],
    'Max Speed': [380., 370., None, 26., 24.],
    'Value': [2000, 1500, 10, 30, 20],
})

# Create TableCleaner
cleaner = ThorTableCleaner()
cleaner.register_numerical('Max Speed', impute='min', clip=True)
cleaner.register_label('Animal') # Categories is optional, unknown values get set to 0
cleaner.register_one_hot('Animal')

# Clean table and split into train/test
X, y = cleaner.transform(t1, label='Value')
head(X)

# Train a model + save cleaner dictionary for reuse (serialize to JSON or pickle)
cleaner_dict = cleaner.to_dict()
for c in cleaner_dict:
    print(c)

# Prediction data
t2 = pa.Table.from_pydict({
    'Animal': ['Falcon', 'Goose', 'Parrot', 'Parrot'],
    'Max Speed': [380., 10., None, 26.]
})
new_cleaner = ThorTableCleaner().from_dict(cleaner_dict)
X, _ = new_cleaner.transform(t2)
head(X)

X = new_cleaner.mutate(X)
head(X)

X_train, X_test = new_cleaner.split(table=X, perc=0.5)
head(X_train)