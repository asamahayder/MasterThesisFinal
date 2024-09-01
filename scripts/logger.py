import os

FILENAME = 'log.txt'

def log(*args):
    message = ''.join(map(str, args))
    with open(FILENAME, 'a') as file:
        file.write(message + '\n')

def delete_log():
    if os.path.exists(FILENAME):
        os.remove(FILENAME)
