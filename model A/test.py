import convenience as c

c.genLabelsCSV('./preprocessed-augmented/train/specs/', './preprocessed-augmented/train/')

c.genLabelsCSV('./preprocessed-augmented/val/specs/', './preprocessed-augmented/val/')

print('done')