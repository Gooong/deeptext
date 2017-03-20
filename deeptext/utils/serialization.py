import dill

def save(obj, filename):
    with open(filename, 'wb') as f:
        f.write(dill.dumps(obj))

def restore(filename):
    with open(filename, 'rb') as f:
        return dill.loads(f.read())
