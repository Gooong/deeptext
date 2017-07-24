#-*- coding:utf-8 -*-

import logging
import time

FORMAT = '[%(asctime)-15s] %(message)s'
logging.basicConfig(level=logging.DEBUG, format=FORMAT)

from deeptext.models.sequence_labeling.sequence_labeling import SequenceLabeling

params = {}
params["max_document_len"] = 25
params["embedding_size"] = 50
params["dropout_prob"] = 0.5
params["model_name"] = 'ner_music'
params["model_dir"] = 'data/music/model'

model = SequenceLabeling(params)

model.fit(
        steps=100, 
        batch_size=256, 
        training_data_path='data/music/music_data.txt',
        validation_data_path='data/music/music_data_validation.txt'
        )

#model = SequenceLabeling.restore(model_dir='data/music/model')
#
#
#start = time.time()
#
#model.evaluate(
#        testing_data_path='data/music/music_data_validation.txt'
#        )

#print model.predict([[u'周', u'杰', u'伦', u'的', u'歌', u'给', u'我', u'来', u'点']])
#print model.predict([[u'告', u'诉', u'我', u'周', u'杰', u'伦', u'唱', u'过', u'什', u'么', u'歌', u'曲']])
#
#while True:
#    text = raw_input("> ")   # Python 2.x
#    print model.predict([list(unicode(text, 'utf8'))])
