#-*- coding:utf-8 -*-

from deeptext.models.sequence_labeling.base_model import BaseModel as SequenceLabeling

params = {}
params["max_document_len"] = 25
params["embedding_size"] = 50
params["dropout_prob"] = 0.5
params["model_dir"] = '/Users/yy/data/kcws/20170510/models'

model = SequenceLabeling(params)
model.preprocess_fit_transform(
        training_data_path='/Users/yy/data/kcws/20170510/seg.data.head.1000'
        )

model.build_model()
model.fit(
        steps=100)
model.frozen_save()

#model = SequenceLabeling.restore(model_dir='/data/ruyi/ruyi-ml/data/sentence_trunk_service/model')
#
#import logging
#import time
#
#FORMAT = '[%(asctime)-15s] %(message)s'
#logging.basicConfig(level=logging.DEBUG, format=FORMAT)
#
#start = time.time()
#
#model.evaluate(
#        testing_token_path='/data/ruyi/ruyi-ml/data/sentence_trunk_service/x.training.data',
#        testing_label_path='/data/ruyi/ruyi-ml/data/sentence_trunk_service/y.training.data'
#        ) #logging.info(time.time() - start)
#
#print model.predict([[u'周', u'杰', u'伦', u'的', u'歌', u'给', u'我', u'来', u'点']])
#print model.predict([[u'告', u'诉', u'我', u'周', u'杰', u'伦', u'唱', u'过', u'什', u'么', u'歌', u'曲']])
#
#while True:
#    text = raw_input("> ")   # Python 2.x
#    print model.predict([list(unicode(text, 'utf8'))])
