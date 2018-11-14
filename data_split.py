import pickle
import sys
import random
def split(data,outfile_X = ""):
    data = pickle.load(data)
    random.shuffle(data)
    length = len(data)
    pickle.dump(data[:(length // 10)*8], open(outfile_X + '.train', 'wb'), -1)
    # pickle.dump(data[1:length // 2 ], open(outfile_Y + '.train', 'wb'), -1)
    pickle.dump(data[(length // 10)*8 : (length//10)*9 ], open(outfile_X + '.valid', 'wb'), -1)
    # pickle.dump(data[length // 2+1: (length // 4) * 3 ], open(outfile_Y + '.valid', 'wb'), -1)
    pickle.dump(data[(length//10)*9: ], open(outfile_X + '.test', 'wb'), -1)
    # pickle.dump(data[(length // 4) * 3 + 1:], open(outfile_Y + '.test', 'wb'), -1)

    cnt = set()
    for i in data[:]:
        for j in i:
            for k in j:
                cnt.add(k)

    print (len(cnt))
    return ("the code number:",data)

def find_num_per_visit(data):
    visit_num = 0
    code_num = 0
    data = pickle.load(data)
    for patient in data:
        for visit in patient:
            visit_num +=1
            for code in visit:
                code_num +=1
    print("visit_num:",visit_num)
    print("code_num:",code_num)
    return None


'''
the whole data: 7537
the vist max length: 42
the code max length: 39

'''

def find_data_shape(data):
    data = pickle.load(data)
    print("the whole data:",len(data))
    vist_mlen = 0
    code_mlen = 0
    for x in data:
        vist_mlen = max(vist_mlen,len(x))
        for y in x:
            code_mlen = max(code_mlen,len(y))
    print("the vist max length:",vist_mlen)
    print("the code max length:",code_mlen)
    return data



if __name__ == '__main__':
    data = "output.seqs"
    # out = sys.argv[2]2
    # pickle_file = open(data,"rb")
    # with open(data,"rb") as pickle_file:
    #     find_data_shape(pickle_file)
    pickle_file = open(data, "rb")
    find_num_per_visit(pickle_file)
    temp = split(pickle_file,"test2")
