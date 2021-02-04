from pesq import pesq
from pystoi import stoi
#from STOI import stoi

def get_pesq(ref, deg, sr):

    score = pesq(sr, ref, deg, 'wb')

    return score

def get_stoi(ref, deg, sr):

    score = stoi(ref, deg, sr, extended=False)

    return score



