import os
import pickle

def read_annotations(annot_dir, photographers2id):
	id2labels = {}
	for fileannot in os.listdir(annot_dir):
		ph = fileannot.split('.txt')[0]
		with open('./annotations/' + fileannot, 'r') as f:
			if fileannot.split('.txt')[0] not in photographers2id.keys():
				continue
		lines = f.readlines()
        for line in lines:
            fileid = line.split('.')[0]
            fileid = fileid.split('/')[-1]
            id2labels[fileid] = photo_code[ph]


if __name__ == '__main__':
	photographers2id = {'Hollming, V.':0, 'Taube, J.':1,'Helander, N.':2, 'Jänis, P.':3, 'Hedenström, O.':4, 'Suomela, E.':5, 'Kivi, K.':6, 'Uomala, V.':7, 'Nurmi, E.':8,
                                                'Sjöblom, K.':9, 'Roivainen, H.':10, 'Manninen, E.':11}
	annot_dir = '../annotations/'
	id2labels = read_annotations(annot_dir, photographers2id)
	with open('./id2labels.pkl', 'wb') as handle:
		pickle.dump(id2labels, handle, protocol=pickle.HIGHEST_PROTOCOL)
