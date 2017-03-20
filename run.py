#-*- coding:utf-8 -*-

from deeptext.models.sequence_labeling.base_model import BaseModel as SequenceLabeling

params = {}
params["max_document_len"] = 10
params["embedding_size"] = 50
params["dropout_prob"] = 0.5

model = SequenceLabeling(params)
model.preprocess_fit_transform(
        training_token_path='/data/ruyi/ruyi-ml/data/sentence_trunk_service/x.training.data',
        training_label_path='/data/ruyi/ruyi-ml/data/sentence_trunk_service/y.training.data'
        )

model.build_model(model_dir='/data/ruyi/ruyi-ml/data/sentence_trunk_service3')
model.fit(steps=10)
model.frozen_save()

model = SequenceLabeling.restore(model_dir='/data/ruyi/ruyi-ml/data/sentence_trunk_service3')

import logging
import time

FORMAT = '[%(asctime)-15s] %(message)s'
logging.basicConfig(level=logging.DEBUG, format=FORMAT)


start = time.time()

print model.predict([[u"敢", u"不", u"敢", u"放", u"点", u"轻", u"音", u"乐", u"愿", u"意", u"不"]])
logging.info(time.time() - start)
