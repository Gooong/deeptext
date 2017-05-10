#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re

rootdir=u'/Users/yy/data/nlp/2014'

symbols=u'，。'

for subdir, dirs, files in os.walk(rootdir):
    for filename in files:
        if not filename.endswith('txt'):
            print os.path.join(subdir, filename)
            continue
        with open(os.path.join(subdir, filename)) as f:
            for line in f:
                uniline = unicode(line.strip(), 'utf-8')
                tokens = []
                for pattern in re.finditer(u' ?([^/]*)\/([^ ]*)', uniline):
                    if pattern.group(1).startswith(u'['):
                        tokens.append(pattern.group(1)[1:])
                    elif u']' in pattern.group(1):
                        tokens.append(pattern.group(1)[:pattern.group(1).index(u']')])
                    elif pattern.group(1) in symbols:
                        print (u'\t'.join(list(u''.join(tokens)))).encode('utf-8')

                        label = u''
                        for token in tokens:
                            if len(label) != 0:
                                label += u'\t'
                            label += u'B' + u''.join([u'\tI'] * (len(token) - 1))
                        print label.encode('utf-8')
                        del tokens[:]
                    else:
                        tokens.append(pattern.group(1))
